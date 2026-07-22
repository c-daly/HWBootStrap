using System;
using System.Collections.Generic;
using System.Linq;

namespace HexWars.Engine.Rl
{
    public readonly struct AdaptiveDiagnostics
    {
        public int DesignCount { get; }
        public int DistinctCustomTemplatesDeployed { get; }
        public int PregameDecisions { get; }
        public int InvalidSequences { get; }
        public bool DeploymentCompleted { get; }

        public AdaptiveDiagnostics(int designCount, int distinctCustomTemplatesDeployed,
            int pregameDecisions, int invalidSequences, bool deploymentCompleted)
        {
            DesignCount = designCount;
            DistinctCustomTemplatesDeployed = distinctCustomTemplatesDeployed;
            PregameDecisions = pregameDecisions;
            InvalidSequences = invalidSequences;
            DeploymentCompleted = deploymentCompleted;
        }
    }

    /// <summary>A two-controller adaptive-v1 game. Pregame state remains in private deployment ledgers;
    /// only the atomic revealed state is retained as the replay start.</summary>
    public sealed class AdaptiveDuelEnv
    {
        private readonly AdaptiveEnvConfig _cfg;
        private readonly AdaptiveLayout _layout;
        private readonly MlContract _contract;
        private readonly List<Command> _log = new List<Command>();
        private readonly HashSet<int> _customTemplatesDeployed = new HashSet<int>();

        private AdaptiveDeployment _setup = null!;
        private GameState? _state;
        private GameState? _start;
        private AdaptiveDecisionState _decision0 = null!;
        private AdaptiveDecisionState _decision1 = null!;
        private AdaptiveUnitSlots _slots0 = null!;
        private AdaptiveUnitSlots _slots1 = null!;
        private IAgent? _controller0;
        private IAgent? _controller1;
        private PlayerId _deploymentSeat;
        private PlayerId _learner;
        private int _steps;
        private int _designCount;
        private int _pregameDecisions;
        private int _invalidSequences;
        private bool _awaitingPostRevealAdvance;

        public AdaptiveDuelEnv(AdaptiveEnvConfig? config = null)
        {
            _cfg = config ?? AdaptiveEnvConfig.Default();
            _layout = new AdaptiveLayout(_cfg);
            _contract = MlContract.CreateAdaptive(_cfg, MlEnvironmentKind.AdaptiveDuel);
        }

        public int ActionCount => _layout.ActionCount;
        public int ObservationLength => _layout.ObservationLength;
        public int ObsChannels => _layout.ObservationChannels;
        public int BoardH => _layout.BoardGen.Height;
        public int BoardW => _layout.BoardGen.Width;
        public AdaptiveEnvConfig Config => _cfg;
        public AdaptiveLayout Layout => _layout;
        public MlContract Contract => _contract;
        public bool DeploymentComplete => _state != null;
        public bool AwaitingPostRevealAdvance => _awaitingPostRevealAdvance;
        public GameState State => _state ?? throw new InvalidOperationException("deployment is not complete");
        public AdaptiveDiagnostics Diagnostics => new AdaptiveDiagnostics(
            _designCount, _customTemplatesDeployed.Count, _pregameDecisions,
            _invalidSequences, DeploymentComplete);

        /// <summary>Null controller/policy means that seat is externally driven in the corresponding
        /// phase. Scripted deployment policies receive a single seat-private snapshot and are invoked once.</summary>
        public View Reset(int seed, IAgent? controller0, IAgent? controller1,
            IDeploymentPolicy? deployment0, IDeploymentPolicy? deployment1,
            PlayerId learnerSeat = PlayerId.Player0)
        {
            var board = new RandomBoardGenerator(_cfg.BoardGen).Generate(seed);
            var errors = _cfg.Validate(board);
            if (errors.Count > 0) throw new ArgumentException(string.Join("; ", errors), nameof(_cfg));

            _setup = new AdaptiveDeployment(board, _cfg);
            _state = null;
            _start = null;
            _decision0 = new AdaptiveDecisionState(PlayerId.Player0);
            _decision1 = new AdaptiveDecisionState(PlayerId.Player1);
            _slots0 = new AdaptiveUnitSlots(_cfg.MaxControllableUnits);
            _slots1 = new AdaptiveUnitSlots(_cfg.MaxControllableUnits);
            _controller0 = controller0;
            _controller1 = controller1;
            _learner = learnerSeat;
            _deploymentSeat = PlayerId.Player0;
            _steps = 0;
            _designCount = 0;
            _pregameDecisions = 0;
            _invalidSequences = 0;
            _awaitingPostRevealAdvance = false;
            _customTemplatesDeployed.Clear();
            _log.Clear();

            ApplyScriptedDeployment(PlayerId.Player0, deployment0);
            ApplyScriptedDeployment(PlayerId.Player1, deployment1);
            CompleteRevealIfReady();
            if (!DeploymentComplete) SelectNextDeploymentSeat();
            else if (!AwaitingPostRevealAdvance) AdvancePastInternal();
            return MakeView(0f);
        }

        /// <summary>Resume scripted gameplay only after a caller has presented the atomic reveal.</summary>
        public View ContinueAfterReveal()
        {
            if (!AwaitingPostRevealAdvance)
                throw new InvalidOperationException("there is no pending post-reveal continuation");
            _awaitingPostRevealAdvance = false;
            AdvancePastInternal();
            return MakeView(TerminalReward());
        }

        /// <summary>Apply one hierarchical action for the currently exposed external seat.</summary>
        public View Step(int action)
        {
            bool wasDeployment = !DeploymentComplete;
            PlayerId seat = wasDeployment ? _deploymentSeat : State.ActivePlayer;
            var decision = Decision(seat);
            var slots = Slots(seat);
            var transition = AdaptiveCoding.ApplyAction(
                action, _state, _setup, seat, decision, _layout, slots);
            _steps++;
            if (wasDeployment) _pregameDecisions++;

            if (transition.InvalidSequence)
            {
                _invalidSequences++;
                return MakeView(0f);
            }

            float reward = transition.Intermediate && seat == _learner
                ? -_cfg.IntermediateDecisionPenalty
                : 0f;
            if (wasDeployment)
            {
                CompleteRevealIfReady();
                if (DeploymentComplete)
                {
                    reward += _cfg.DeploymentCompletionBonus;
                    if (!AwaitingPostRevealAdvance) AdvancePastInternal();
                }
                else SelectNextDeploymentSeat(prefer: seat);
                return MakeView(reward + TerminalReward());
            }

            if (transition.Command != null)
            {
                var result = GameEngine.Apply(State, transition.Command);
                if (!result.Success)
                {
                    _invalidSequences++;
                    decision.Clear(AdaptivePhase.GameplayRoot);
                    return MakeView(0f);
                }

                _state = result.NewState;
                _log.Add(transition.Command);
                RecordSuccessfulCommand(transition.Command);
                SyncSlots();
                AdvancePastInternal();
            }
            return MakeView(reward + TerminalReward());
        }

        /// <summary>The replay begins with the atomically revealed round-one armies. Hidden setup actions
        /// are intentionally absent; successful gameplay and template replacement commands follow.</summary>
        public string ToReplay()
        {
            if (_start == null) throw new InvalidOperationException("deployment is not complete");
            return ReplayFile.Write(_start, _log);
        }

        private void ApplyScriptedDeployment(PlayerId seat, IDeploymentPolicy? policy)
        {
            if (policy == null) return;
            IReadOnlyList<DeploymentPlacement>? placements;
            try { placements = policy.Choose(_setup.View(seat)); }
            catch (Exception exception)
            {
                _invalidSequences++;
                throw new InvalidOperationException(
                    $"scripted deployment policy for {seat} failed", exception);
            }
            if (placements == null)
            {
                _invalidSequences++;
                throw new InvalidOperationException(
                    $"scripted deployment policy for {seat} returned no placements");
            }

            var scratch = new AdaptiveDeployment(_setup.Board, _cfg);
            if (!TryApplyDeployment(scratch, seat, placements))
            {
                _invalidSequences++;
                throw new InvalidOperationException(
                    $"scripted deployment policy for {seat} returned an invalid placement or confirmation");
            }
            if (!TryApplyDeployment(_setup, seat, placements))
                throw new InvalidOperationException("validated scripted deployment could not be applied");
        }

        private bool TryApplyDeployment(AdaptiveDeployment target, PlayerId seat,
            IReadOnlyList<DeploymentPlacement> placements)
        {
            foreach (DeploymentPlacement placement in placements)
            {
                int expectedSlot = LowestFreeDeploymentSlot(target, seat);
                if (placement.Slot != expectedSlot
                    || !target.TryPlace(seat, placement.TemplateIndex, placement.Cell)) return false;
            }
            return target.TryConfirm(seat);
        }

        private int LowestFreeDeploymentSlot(AdaptiveDeployment target, PlayerId seat)
        {
            var occupied = new HashSet<int>(target.Placements(seat).Select(p => p.Slot));
            for (int slot = 0; slot < _cfg.StartingUnitCount; slot++)
                if (!occupied.Contains(slot)) return slot;
            return -1;
        }

        private void CompleteRevealIfReady()
        {
            if (DeploymentComplete || !_setup.IsRevealed) return;
            _state = _setup.Reveal(PlayerId.Player0);
            _start = _state;
            _decision0.Clear(AdaptivePhase.GameplayRoot);
            _decision1.Clear(AdaptivePhase.GameplayRoot);
            _awaitingPostRevealAdvance = Controller(State.ActivePlayer) != null;
            foreach (PlayerId seat in new[] { PlayerId.Player0, PlayerId.Player1 })
                foreach (DeploymentPlacement placement in _setup.Placements(seat))
                    if (placement.TemplateIndex >= _cfg.FixedTemplateCount)
                        _customTemplatesDeployed.Add(placement.TemplateIndex);
            SyncSlots();
        }

        private void SelectNextDeploymentSeat(PlayerId? prefer = null)
        {
            if (prefer.HasValue && !_setup.Confirmed(prefer.Value))
            {
                _deploymentSeat = prefer.Value;
                return;
            }
            if (!_setup.Confirmed(PlayerId.Player0)) _deploymentSeat = PlayerId.Player0;
            else if (!_setup.Confirmed(PlayerId.Player1)) _deploymentSeat = PlayerId.Player1;
        }

        private IAgent? Controller(PlayerId seat) =>
            seat == PlayerId.Player0 ? _controller0 : _controller1;

        private AdaptiveDecisionState Decision(PlayerId seat) =>
            seat == PlayerId.Player0 ? _decision0 : _decision1;

        private AdaptiveUnitSlots Slots(PlayerId seat) =>
            seat == PlayerId.Player0 ? _slots0 : _slots1;

        private void SyncSlots()
        {
            if (_state == null) return;
            _slots0.Sync(_state, PlayerId.Player0);
            _slots1.Sync(_state, PlayerId.Player1);
        }

        private void RecordSuccessfulCommand(Command command)
        {
            if (command is ReplaceTemplate) _designCount++;
            if (command is DeployUnit deploy && deploy.TemplateIndex >= _cfg.FixedTemplateCount)
                _customTemplatesDeployed.Add(deploy.TemplateIndex);
        }

        /// <summary>Retains the legacy guarded scripted-agent loop. An illegal scripted decision is
        /// diagnosed and unstuck with EndTurn; external hierarchical rejections never take this fallback.</summary>
        private void AdvancePastInternal()
        {
            if (_state == null) return;
            int guard = 0;
            while (!_state.IsGameOver && Controller(_state.ActivePlayer) != null && guard++ < 8000)
            {
                PlayerId seat = _state.ActivePlayer;
                Command command = Controller(seat)!.Decide(_state);
                var result = GameEngine.Apply(_state, command);
                if (result.Success)
                {
                    _state = result.NewState;
                    _log.Add(command);
                    RecordSuccessfulCommand(command);
                    SyncSlots();
                    continue;
                }

                _invalidSequences++;
                Decision(seat).Clear(AdaptivePhase.GameplayRoot);
                var end = GameEngine.Apply(_state, new EndTurn(seat));
                if (!end.Success) break;
                _state = end.NewState;
                _log.Add(new EndTurn(seat));
                SyncSlots();
            }
        }

        private float TerminalReward()
        {
            if (_state == null || !_state.IsGameOver) return 0f;
            if (_state.Winner == _learner) return 1f;
            if (_state.Winner.HasValue) return -1f;
            return 0f;
        }

        private View MakeView(float reward)
        {
            bool terminated = DeploymentComplete && State.IsGameOver;
            PlayerId seat = terminated ? _learner
                : DeploymentComplete ? State.ActivePlayer : _deploymentSeat;
            var decision = Decision(seat);
            var slots = Slots(seat);
            bool truncated = !terminated && _steps >= checked(_cfg.MaxSteps * 2);
            int winner = terminated && State.Winner.HasValue ? (int)State.Winner.Value : -1;
            return new View(
                AdaptiveCoding.Observe(_state, _setup, seat, decision, _layout, slots),
                AdaptiveCoding.Mask(_state, _setup, seat, decision, _layout, slots),
                (int)seat, reward, winner, terminated, truncated,
                DeploymentComplete, Diagnostics);
        }

        public readonly struct View
        {
            public readonly float[] Observation;
            public readonly bool[] ActionMask;
            public readonly int Seat;
            public readonly float Reward;
            public readonly int Winner;
            public readonly bool Terminated;
            public readonly bool Truncated;
            public readonly bool DeploymentComplete;
            public readonly AdaptiveDiagnostics Diagnostics;

            public View(float[] observation, bool[] actionMask, int seat, float reward, int winner,
                bool terminated, bool truncated, bool deploymentComplete, AdaptiveDiagnostics diagnostics)
            {
                Observation = observation;
                ActionMask = actionMask;
                Seat = seat;
                Reward = reward;
                Winner = winner;
                Terminated = terminated;
                Truncated = truncated;
                DeploymentComplete = deploymentComplete;
                Diagnostics = diagnostics;
            }
        }
    }
}
