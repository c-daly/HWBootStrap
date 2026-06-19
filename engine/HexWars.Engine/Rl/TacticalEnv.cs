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
        public GameConfig Game = GameConfig.Default();
        public IReadOnlyList<UnitStats> Roster = DefaultRoster();
        public int MaxSteps = 400;
        public float ShapeScale = 0.01f;

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
    /// A single-agent reinforcement-learning environment over the HexWars engine (tactics scope: fixed
    /// rosters, no economy). The learning agent controls one seat; the other seat is played by a
    /// configurable opponent (random / greedy / a frozen policy) that the env runs automatically, so the
    /// agent always acts on its own turn. Exposes the standard MDP surface — Reset, Step, observation,
    /// and a legal-action mask — over the deterministic engine. The .NET core a Gym bridge wraps.
    /// </summary>
    public sealed class TacticalEnv
    {
        private const int PerCell = 12; // elev, 4×terrain, myUnit, enemyUnit, myGen, enemyGen, hp, dmg, range
        private const int Globals = 5;  // myPoints, foePoints, round, myCount, foeCount

        private readonly EnvConfig _cfg;
        private readonly Func<int, IAgent> _opponentFactory;
        private readonly PlayerId _seat;
        private readonly PlayerId _foe;
        private readonly int _r;

        private readonly List<HexCoord> _cells = new List<HexCoord>();
        private readonly Dictionary<HexCoord, int> _cellIx = new Dictionary<HexCoord, int>();
        private readonly int _n;

        private GameState _state = null!;
        private IAgent _opponent = null!;
        private int[] _slotToUnitId = Array.Empty<int>();
        private int _steps;
        private float _prevAdvantage;

        public TacticalEnv(Func<int, IAgent> opponentFactory, PlayerId learningSeat = PlayerId.Player0, EnvConfig? cfg = null)
        {
            _cfg = cfg ?? new EnvConfig();
            _opponentFactory = opponentFactory ?? (s => new RandomAgent(s));
            _seat = learningSeat;
            _foe = learningSeat == PlayerId.Player0 ? PlayerId.Player1 : PlayerId.Player0;
            _r = _cfg.Roster.Count;

            for (int row = 0; row < _cfg.BoardGen.Height; row++)
                for (int col = 0; col < _cfg.BoardGen.Width; col++)
                {
                    var c = HexLayout.OffsetToAxial(col, row);
                    if (_cellIx.ContainsKey(c)) continue;
                    _cellIx[c] = _cells.Count;
                    _cells.Add(c);
                }
            _n = _cells.Count;
        }

        public int ActionCount => 1 + 2 * _r * _n;     // EndTurn + (move|attack) × slots × cells
        public int ObservationLength => PerCell * _n + Globals;
        public GameState State => _state;

        public float[] Reset(int seed)
        {
            var board = new RandomBoardGenerator(_cfg.BoardGen).Generate(seed);
            _opponent = _opponentFactory(seed);

            int nextId = 1;
            var units0 = BuildRoster(board, PlayerId.Player0, ref nextId);
            var units1 = BuildRoster(board, PlayerId.Player1, ref nextId);
            var p0 = new PlayerState(PlayerId.Player0, 0, null, units0, null);
            var p1 = new PlayerState(PlayerId.Player1, 0, null, units1, null);
            _state = new GameState(board, _cfg.Game, new[] { p0, p1 }, PlayerId.Player0, 1, nextId);

            _slotToUnitId = new int[_r];
            var mine = _state.Player(_seat).UnitsOnBoard;
            for (int i = 0; i < _r; i++) _slotToUnitId[i] = i < mine.Count ? mine[i].Id : -1;

            _steps = 0;
            AdvanceToSeat();
            _prevAdvantage = Advantage();
            return Observe();
        }

        public StepResult Step(int action)
        {
            var cmd = Decode(action);
            if (cmd != null)
            {
                var r = GameEngine.Apply(_state, cmd);
                if (r.Success) _state = r.NewState;
            }
            AdvanceToSeat();
            _steps++;

            float reward = Reward();
            bool terminated = _state.IsGameOver;
            bool truncated = !terminated && _steps >= _cfg.MaxSteps;
            return new StepResult(Observe(), reward, terminated, truncated, LegalActionMask());
        }

        public bool[] LegalActionMask()
        {
            var mask = new bool[ActionCount];
            mask[0] = true; // EndTurn is always available
            foreach (var cmd in LegalMoves.For(_state))
            {
                int ix = Encode(cmd);
                if (ix >= 0 && ix < mask.Length) mask[ix] = true;
            }
            return mask;
        }

        // ---- opponent / reward ----

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

        private float Advantage() => WinCheck.Evaluate(_state, _seat) - WinCheck.Evaluate(_state, _foe);

        private float Reward()
        {
            float adv = Advantage();
            float shaped = _cfg.ShapeScale * (adv - _prevAdvantage);
            _prevAdvantage = adv;

            if (!_state.IsGameOver) return shaped;
            if (_state.Winner == _seat) return shaped + 1f;
            if (_state.Winner == _foe) return shaped - 1f;
            return shaped; // draw
        }

        // ---- action codec ----

        private int Encode(Command cmd)
        {
            switch (cmd)
            {
                case EndTurn _:
                    return 0;
                case MoveUnit m:
                {
                    int slot = SlotOf(m.UnitId);
                    if (slot < 0 || !_cellIx.TryGetValue(m.Dest, out int cell)) return -1;
                    return 1 + slot * _n + cell;
                }
                case AttackUnit a:
                {
                    int slot = SlotOf(a.AttackerId);
                    var tc = CellOfEntity(a.TargetId);
                    if (slot < 0 || tc == null || !_cellIx.TryGetValue(tc.Value, out int cell)) return -1;
                    return 1 + _r * _n + slot * _n + cell;
                }
                default:
                    return -1; // deploy/create not in the tactical action space
            }
        }

        private Command? Decode(int action)
        {
            if (action <= 0) return new EndTurn(_seat);

            int a = action - 1;
            bool attack = a >= _r * _n;
            if (attack) a -= _r * _n;

            int slot = a / _n, cell = a % _n;
            if (slot < 0 || slot >= _r || cell < 0 || cell >= _n) return null;

            int unitId = _slotToUnitId[slot];
            if (unitId < 0 || !IsLivingSeatUnit(unitId)) return null;

            var coord = _cells[cell];
            if (!attack) return new MoveUnit(_seat, unitId, coord);

            int targetId = EnemyEntityAt(coord);
            return targetId < 0 ? null : new AttackUnit(_seat, unitId, targetId);
        }

        private int SlotOf(int unitId)
        {
            for (int i = 0; i < _r; i++) if (_slotToUnitId[i] == unitId) return i;
            return -1;
        }

        private bool IsLivingSeatUnit(int unitId)
        {
            foreach (var u in _state.Player(_seat).UnitsOnBoard)
                if (u.Id == unitId && u.IsAlive) return true;
            return false;
        }

        private HexCoord? CellOfEntity(int id)
        {
            foreach (var p in _state.Players)
            {
                foreach (var u in p.UnitsOnBoard) if (u.Id == id && u.IsAlive) return u.Cell;
                foreach (var g in p.Generators) if (g.Id == id && g.IsAlive) return g.Cell;
            }
            return null;
        }

        private int EnemyEntityAt(HexCoord coord)
        {
            var foe = _state.Player(_foe);
            foreach (var u in foe.UnitsOnBoard) if (u.IsAlive && u.Cell == coord) return u.Id;
            foreach (var g in foe.Generators) if (g.IsAlive && g.Cell == coord) return g.Id;
            return -1;
        }

        // ---- observation ----

        private float[] Observe()
        {
            var obs = new float[ObservationLength];
            var board = _state.Board;
            float maxElev = Math.Max(1, _cfg.BoardGen.MaxElevation);

            for (int i = 0; i < _n; i++)
            {
                if (!board.Contains(_cells[i])) continue;
                var t = board.TileAt(_cells[i]);
                int o = i * PerCell;
                obs[o] = Math.Min(1f, t.Elevation / maxElev);
                switch (t.Terrain)
                {
                    case TerrainType.Plains: obs[o + 1] = 1f; break;
                    case TerrainType.Forest: obs[o + 2] = 1f; break;
                    case TerrainType.Rough: obs[o + 3] = 1f; break;
                    case TerrainType.Water: obs[o + 4] = 1f; break;
                }
            }

            WriteEntities(obs, _state.Player(_seat), mine: true);
            WriteEntities(obs, _state.Player(_foe), mine: false);

            int g = _n * PerCell;
            obs[g + 0] = Math.Min(1f, _state.Player(_seat).Points / 50f);
            obs[g + 1] = Math.Min(1f, _state.Player(_foe).Points / 50f);
            obs[g + 2] = Math.Min(1f, _state.Round / (float)Math.Max(1, _cfg.Game.RoundCap));
            obs[g + 3] = Math.Min(1f, AliveUnits(_state.Player(_seat)) / (float)Math.Max(1, _r));
            obs[g + 4] = Math.Min(1f, AliveUnits(_state.Player(_foe)) / (float)Math.Max(1, _r));
            return obs;
        }

        private void WriteEntities(float[] obs, PlayerState p, bool mine)
        {
            foreach (var u in p.UnitsOnBoard)
            {
                if (!u.IsAlive || !_cellIx.TryGetValue(u.Cell, out int i)) continue;
                int o = i * PerCell;
                obs[o + (mine ? 5 : 6)] = 1f;
                obs[o + 9] = Math.Min(1f, u.CurrentHp / (float)Math.Max(1, u.Stats.Health));
                obs[o + 10] = Math.Min(1f, u.Stats.Damage / 10f);
                obs[o + 11] = Math.Min(1f, u.Stats.Range / 8f);
            }
            foreach (var gen in p.Generators)
            {
                if (!gen.IsAlive || !_cellIx.TryGetValue(gen.Cell, out int i)) continue;
                obs[i * PerCell + (mine ? 7 : 8)] = 1f;
            }
        }

        private static int AliveUnits(PlayerState p)
        {
            int c = 0;
            foreach (var u in p.UnitsOnBoard) if (u.IsAlive) c++;
            return c;
        }

        // ---- setup ----

        private IReadOnlyList<Unit> BuildRoster(Board board, PlayerId player, ref int nextId)
        {
            var zone = new List<HexCoord>(board.DeploymentZone(player));
            zone.Sort((x, y) => x.Q != y.Q ? x.Q.CompareTo(y.Q) : x.R.CompareTo(y.R));

            var units = new List<Unit>();
            int count = Math.Min(_r, zone.Count);
            for (int i = 0; i < count; i++)
            {
                var c = zone[i];
                units.Add(new Unit(nextId++, player, _cfg.Roster[i], c, board.TileAt(c).Elevation));
            }
            return units;
        }
    }
}
