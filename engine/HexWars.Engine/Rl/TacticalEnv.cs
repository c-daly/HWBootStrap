using System;
using System.Collections.Generic;

namespace HexWars.Engine.Rl
{
    /// <summary>Settings for a <see cref="TacticalEnv"/> episode: the board generator, rules, the fixed
    /// roster each side starts with (tactics-only — no economy yet), the truncation horizon, and the
    /// reward-shaping weight on position-value change.</summary>
    public sealed class EnvConfig
    {
        public BoardGenConfig BoardGen = BoardGenConfig.Default();
        public GameConfig Game = GameConfig.Default(biomesEnabled: false); // biomes off for now — terrain is mechanically inert
        public IReadOnlyList<UnitStats> Roster = DefaultRoster();
        public int MaxSteps = 600; // headroom for the higher round cap (games can run longer to a wipeout)
        public float ShapeScale = 0.01f;
        public float StepPenalty = 0.005f;       // small per-turn cost -> discourages passive play / stalemates
        public float ClosingWeight = 0.02f;      // reward per hex of distance closed to the enemy -> breaks standoffs
        public float DrawCreditWeight = 0.25f;   // partial terminal credit at the cap (lowered: don't reward coasting to a draw)
        public float PointsWeight = 0.5f;        // banked points worth less than committed force -> deploying earned bounty pays

        public static IReadOnlyList<UnitStats> DefaultRoster() => new[]
        {
            //                 H  D  Df Mv VMv R RA V VA  — mobile enough to cross the terrain
            new UnitStats(5, 3, 2, 3, 2, 1, 1, 2, 1), // bruiser
            new UnitStats(3, 5, 0, 3, 2, 2, 1, 3, 1), // striker
            new UnitStats(2, 2, 0, 4, 3, 1, 0, 5, 2), // scout
        };
    }

    /// <summary>What a <see cref="TacticalEnv.Step"/> returns: the next observation (from the learning
    /// seat's perspective), the reward, terminal/truncation flags, and the legal-action mask for the
    /// next decision (true = selectable). Mirrors the Gymnasium step tuple.</summary>
    public readonly struct StepResult
    {
        public readonly float[] Observation;
        public readonly float Reward;
        public readonly bool Terminated;
        public readonly bool Truncated;
        public readonly bool[] ActionMask;

        public StepResult(float[] observation, float reward, bool terminated, bool truncated, bool[] actionMask)
        {
            Observation = observation;
            Reward = reward;
            Terminated = terminated;
            Truncated = truncated;
            ActionMask = actionMask;
        }
    }

    /// <summary>
    /// Single-agent RL environment over the HexWars engine (tactics scope: fixed rosters, no economy).
    /// The learning agent controls one seat; the other seat is a configurable opponent the env plays
    /// automatically, so the agent always acts on its own turn. Observation/action encoding is shared
    /// with the duel env via <see cref="TacticalCoding"/>, so a trained model sees the same thing at
    /// training time and at duel time.
    /// </summary>
    public sealed class TacticalEnv
    {
        private readonly EnvConfig _cfg;
        private readonly TacticalLayout _layout;
        private readonly Func<int, IAgent> _opponentFactory;
        private readonly PlayerId _seat;
        private readonly PlayerId _foe;

        private GameState _state = null!;
        private IAgent _opponent = null!;
        private int[] _slot = Array.Empty<int>();
        private int _steps;
        private float _prevAdvantage;
        private float _armyValue;

        public TacticalEnv(Func<int, IAgent> opponentFactory, PlayerId learningSeat = PlayerId.Player0, EnvConfig? cfg = null)
        {
            _cfg = cfg ?? new EnvConfig();
            _layout = new TacticalLayout(_cfg);
            _opponentFactory = opponentFactory ?? (s => new RandomAgent(s));
            _seat = learningSeat;
            _foe = learningSeat == PlayerId.Player0 ? PlayerId.Player1 : PlayerId.Player0;
        }

        public int ActionCount => _layout.ActionCount;
        public int ObservationLength => _layout.ObservationLength;
        public GameState State => _state;

        public float[] Reset(int seed)
        {
            var (state, slot0, slot1) = _layout.NewGame(seed);
            _state = state;
            _slot = _seat == PlayerId.Player0 ? slot0 : slot1;
            _opponent = _opponentFactory(seed);
            _steps = 0;
            AdvanceToSeat();
            _prevAdvantage = Advantage();
            _armyValue = RewardShaping.PositionValue(_state, _seat, _cfg.PointsWeight);
            return TacticalCoding.Observe(_state, _seat, _layout);
        }

        public StepResult Step(int action)
        {
            var cmd = TacticalCoding.Decode(action, _state, _seat, _slot, _layout);
            float closing = 0f;
            if (cmd != null)
            {
                // active-piece closing: reward the unit you MOVE for reducing its own gap to the nearest
                // enemy, measured at your move (before the opponent replies) so it's undiluted + noise-free
                var mv = cmd as MoveUnit;
                int gapBefore = mv != null ? RewardShaping.GapOfUnit(_state, mv.UnitId, _seat, _foe) : -1;
                var r = GameEngine.Apply(_state, cmd);
                if (r.Success)
                {
                    _state = r.NewState;
                    if (mv != null && gapBefore >= 0)
                    {
                        int gapAfter = RewardShaping.GapOfUnit(_state, mv.UnitId, _seat, _foe);
                        if (gapAfter >= 0) closing = _cfg.ClosingWeight * (gapBefore - gapAfter);
                    }
                }
            }
            AdvanceToSeat();
            _steps++;

            float reward = Reward(closing);
            bool terminated = _state.IsGameOver;
            bool truncated = !terminated && _steps >= _cfg.MaxSteps;
            return new StepResult(TacticalCoding.Observe(_state, _seat, _layout), reward, terminated, truncated, LegalActionMask());
        }

        public bool[] LegalActionMask() => TacticalCoding.Mask(_state, _seat, _slot, _layout);

        private void AdvanceToSeat()
        {
            int guard = 0;
            while (!_state.IsGameOver && _state.ActivePlayer != _seat && guard++ < 4000)
            {
                var r = GameEngine.Apply(_state, _opponent.Decide(_state));
                if (r.Success) { _state = r.NewState; continue; }
                var end = GameEngine.Apply(_state, new EndTurn(_state.ActivePlayer)); // unstick an illegal pick
                if (end.Success) _state = end.NewState; else break;
            }
        }

        private float Advantage() => RewardShaping.Advantage(_state, _seat, _foe, _cfg.PointsWeight);

        private float Reward(float closing)
        {
            float adv = Advantage();
            float shaped = _cfg.ShapeScale * (adv - _prevAdvantage) + closing - _cfg.StepPenalty;
            _prevAdvantage = adv;

            if (!_state.IsGameOver) return shaped;
            if (_state.Winner == _seat) return shaped + 1f;
            if (_state.Winner == _foe) return shaped - 1f;
            return shaped + RewardShaping.DrawCredit(_state, _seat, _foe, _armyValue, _cfg.DrawCreditWeight, _cfg.PointsWeight); // cap draw
        }
    }
}
