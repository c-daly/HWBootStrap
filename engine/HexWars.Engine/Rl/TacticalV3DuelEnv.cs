using System;
using System.Collections.Generic;
using System.Linq;

namespace HexWars.Engine.Rl
{
    public sealed class TacticalV3DuelEnv
    {
        private readonly TacticalV3Config _config;
        private readonly TacticalV2Layout _layout;
        private readonly ILegalCandidateSource _candidates;
        private readonly ISeatObservationSource _observations;
        private readonly IActionResolver _resolver;
        private readonly IRewardContract _reward;
        private readonly List<DuelTransition> _transitions = new List<DuelTransition>();

        private GameState _start = null!;
        private GameState _state = null!;
        private TacticalV3DecisionFrame _frame = null!;
        private IAgent? _controller0;
        private IAgent? _controller1;
        private PlayerId _learnerSeat;
        private string _startProfileId = "standard-3v3";
        private PlayerId _referenceSeat;
        private bool _hasReset;

        public TacticalV3DuelEnv(TacticalV3Config config)
        {
            _config = SnapshotConfig(config ?? throw new ArgumentNullException(nameof(config)));
            _layout = new TacticalV2Layout(_config.Match);
            _observations = new TacticalV3SeatObservationSource(_config);
            _candidates = new TacticalV3LegalCandidateSource(_observations, _config.Capacity);
            _resolver = new TacticalV3ActionResolver();
            _reward = new TacticalV3Reward(_config.Reward);
        }

        public GameState State
        {
            get
            {
                RequireReset();
                return Snapshot(_state);
            }
        }
        public int InternalFallbackCount { get; private set; }

        public TacticalV3View Reset(
            int seed,
            IAgent? controller0,
            IAgent? controller1,
            PlayerId learnerSeat = PlayerId.Player0)
        {
            TacticalV2Start start = _layout.NewGame(seed);
            return ResetFromStart(
                start, controller0, controller1, learnerSeat, referenceSeat: learnerSeat);
        }

        internal TacticalV3View ResetSelectedProfile(
            int seed,
            IAgent? controller0,
            IAgent? controller1,
            PlayerId learnerSeat)
        {
            if (_config.Match.PlacementPolicy != "profiled-seeded-v1")
                return Reset(seed, controller0, controller1, learnerSeat);

            if (_config.Match.StartDistribution == null)
                throw new InvalidOperationException(
                    "profiled tactical-v3 reset requires a start distribution");
            string profileId = _config.Match.StartDistribution.Select(seed);
            if (!_config.Match.StartProfiles.Any(profile => profile.Id == profileId))
                throw new InvalidOperationException(
                    "selected start profile '" + profileId +
                    "' is not declared by the tactical-v3 configuration");

            return Reset(
                seed, controller0, controller1, profileId, learnerSeat, learnerSeat);
        }

        public TacticalV3View Reset(
            int seed,
            IAgent? controller0,
            IAgent? controller1,
            string startProfileId,
            PlayerId referenceSeat,
            PlayerId learnerSeat = PlayerId.Player0)
        {
            if (string.IsNullOrEmpty(startProfileId))
                throw new ArgumentException("start profile id must not be empty", nameof(startProfileId));
            TacticalV2StartProfile? profile = _config.Match.StartProfiles?
                .SingleOrDefault(item => item.Id == startProfileId);
            if (profile == null)
                throw new ArgumentException(
                    "start profile '" + startProfileId +
                    "' is not declared by this tactical-v3 configuration",
                    nameof(startProfileId));

            TacticalV2Start start = _layout.NewGame(seed, profile, referenceSeat);
            return ResetFromStart(
                start, controller0, controller1, learnerSeat, referenceSeat);
        }

        public TacticalV3View Step(long decisionId, int candidateId)
        {
            RequireReset();
            if (decisionId != _frame.DecisionId)
                throw new InvalidOperationException("tactical-v3 decision id is stale");

            if (IsFinished)
                throw new InvalidOperationException("tactical-v3 episode is already finished");
            Command command = _resolver.Resolve(
                _frame, decisionId, candidateId, _state);

            if (!TryApplyAccepted(command))
                throw new InvalidOperationException(
                    "accepted tactical-v3 candidate was rejected by the game engine");
            AdvancePastInternalControllers();
            RefreshFrame();
            return MakeView();
        }

        public string ToReplay()
        {
            RequireReset();
            return ReplayFile.Write(
                _start,
                _transitions.Select(transition => transition.Command).ToArray());
        }

        private TacticalV3View ResetFromStart(
            TacticalV2Start start,
            IAgent? controller0,
            IAgent? controller1,
            PlayerId learnerSeat,
            PlayerId referenceSeat)
        {
            RequireSeat(learnerSeat, nameof(learnerSeat));
            RequireSeat(referenceSeat, nameof(referenceSeat));

            _start = start.State;
            _state = start.State;
            _controller0 = controller0;
            _controller1 = controller1;
            _learnerSeat = learnerSeat;
            _startProfileId = start.ProfileId;
            _referenceSeat = referenceSeat;
            _transitions.Clear();
            InternalFallbackCount = 0;
            _hasReset = true;
            _reward.Reset(_state, _learnerSeat);
            RefreshFrame();
            AdvancePastInternalControllers();
            RefreshFrame();
            return MakeView();
        }

        private IAgent? Controller(PlayerId seat) =>
            seat == PlayerId.Player0 ? _controller0 : _controller1;

        private bool TryApplyAccepted(Command command)
        {
            GameState before = _state;
            Result result = GameEngine.Apply(before, command);
            if (!result.Success) return false;

            _state = result.NewState;
            _transitions.Add(new DuelTransition(before, command, _state));
            return true;
        }

        private void AdvancePastInternalControllers()
        {
            int guard = 0;
            while (!IsFinished && Controller(_state.ActivePlayer) != null && guard++ < 8000)
            {
                PlayerId seat = _state.ActivePlayer;
                Command command = Controller(seat)!.Decide(Snapshot(_state));
                if (TryApplyAccepted(command)) continue;

                if (TryApplyAccepted(new EndTurn(seat)))
                {
                    InternalFallbackCount++;
                    continue;
                }
                break;
            }

            if (guard >= 8000 && !IsFinished && Controller(_state.ActivePlayer) != null)
                throw new InvalidOperationException("tactical-v3 internal controller guard exhausted");
        }

        private void RefreshFrame()
        {
            if (IsFinished)
            {
                TacticalV3Observation observation = _observations.Observe(
                    _state, _state.ActivePlayer, EmptyObservationMemory.Instance);
                _frame = new TacticalV3DecisionFrame(
                    _state, _transitions.Count, _state.ActivePlayer, observation,
                    Array.Empty<TacticalV3Candidate>(), Array.Empty<Command>());
                return;
            }
            _frame = _candidates.CreateFrame(
                _state,
                _state.ActivePlayer,
                EmptyObservationMemory.Instance,
                _transitions.Count);
        }

        private TacticalV3View MakeView()
        {
            bool terminated = _state.IsGameOver;
            bool truncated = !terminated && _transitions.Count >= _config.Match.MaxSteps;
            int winner = terminated && _state.Winner.HasValue
                ? (int)_state.Winner.Value
                : -1;
            return new TacticalV3View(
                _frame,
                _reward.Evaluate(_state, terminated, truncated),
                _frame.Seat,
                winner,
                terminated,
                truncated,
                _startProfileId,
                _referenceSeat);
        }

        private static TacticalV3Config SnapshotConfig(TacticalV3Config source)
        {
            TacticalV2Config match = source.Match;
            var matchSnapshot = new TacticalV2Config
            {
                BoardGen = match.BoardGen,
                Game = SnapshotGameConfig(match.Game),
                Templates = Array.AsReadOnly(match.Templates.ToArray()),
                StartingUnitCount = match.StartingUnitCount,
                MaxControllableUnits = match.MaxControllableUnits,
                MaxSteps = match.MaxSteps,
                ShapeScale = match.ShapeScale,
                StepPenalty = match.StepPenalty,
                ClosingWeight = match.ClosingWeight,
                DrawCreditWeight = match.DrawCreditWeight,
                PointsWeight = match.PointsWeight,
                PlacementPolicy = match.PlacementPolicy,
                StartProfiles = Array.AsReadOnly(match.StartProfiles.ToArray()),
                StartDistribution = match.StartDistribution,
            };
            return new TacticalV3Config(matchSnapshot, source.Capacity, source.Reward);
        }

        private static GameConfig SnapshotGameConfig(GameConfig source)
        {
            var terrain = new Dictionary<TerrainType, TerrainDef>();
            foreach (TerrainType terrainType in Enum.GetValues(typeof(TerrainType)))
            {
                try
                {
                    terrain.Add(terrainType, source.Terrain(terrainType));
                }
                catch (KeyNotFoundException)
                {
                    // Sparse configs are valid when the omitted terrain never appears on their board.
                }
            }

            return new GameConfig(
                terrain,
                startingPoints: source.StartingPoints,
                bountyRate: source.BountyRate,
                generatorCost: source.GeneratorCost,
                generatorOutput: source.GeneratorOutput,
                generatorHealth: source.GeneratorHealth,
                damageFloor: source.DamageFloor,
                dmgHighGroundBonus: source.DmgHighGroundBonus,
                rangeHighGroundBonus: source.RangeHighGroundBonus,
                roundCap: source.RoundCap,
                designFee: source.DesignFee,
                deployCostMultiplier: source.DeployCostMultiplier,
                turnPolicy: source.TurnPolicy,
                biomesEnabled: source.BiomesEnabled,
                winConditions: source.WinConditions,
                captureCost: source.CaptureCost,
                economyWinThreshold: source.EconomyWinThreshold,
                scoreKills: source.ScoreKills,
                scorePoints: source.ScorePoints,
                scoreArmy: source.ScoreArmy,
                scoreTerritory: source.ScoreTerritory,
                upkeepFactor: source.UpkeepFactor,
                captureFactor: source.CaptureFactor,
                buildFactor: source.BuildFactor,
                territoryMode: source.TerritoryMode,
                claimEndsTurn: source.ClaimEndsTurn,
                buildAnywhere: source.BuildAnywhere,
                territoryIncome: source.TerritoryIncome,
                generatorsEnabled: source.GeneratorsEnabled,
                pointDecay: source.PointDecay,
                fogOfWar: source.FogOfWar,
                maxDesignPointCost: source.MaxDesignPointCost,
                fixedTemplateCount: source.FixedTemplateCount,
                templateSlotCount: source.TemplateSlotCount);
        }

        private static GameState Snapshot(GameState source)
        {
            var control = new Dictionary<HexCoord, PlayerId>();
            foreach (Tile tile in source.Board.Tiles)
            {
                PlayerId? owner = source.Board.Controller(tile.Coord);
                if (owner.HasValue) control.Add(tile.Coord, owner.Value);
            }

            var board = new Board(
                source.Board.Tiles.ToArray(),
                source.Board.DeploymentZone(PlayerId.Player0).ToArray(),
                source.Board.DeploymentZone(PlayerId.Player1).ToArray(),
                control);
            PlayerState[] players = source.Players.Select(player =>
                new PlayerState(
                    player.Id,
                    player.Points,
                    player.Barracks.ToArray(),
                    player.UnitsOnBoard.ToArray(),
                    player.Generators.ToArray(),
                    player.DestroyedValue)).ToArray();
            return new GameState(
                board,
                source.Config,
                players,
                source.ActivePlayer,
                source.Round,
                source.NextEntityId,
                source.IsGameOver,
                source.Winner,
                new HashSet<int>(source.MovedUnitIds),
                new HashSet<int>(source.AttackedUnitIds),
                new Dictionary<int, (int H, int V)>(source.MovementSpent));
        }

        private bool IsFinished =>
            _state.IsGameOver ||
            _transitions.Count >= _config.Match.MaxSteps;

        private void RequireReset()
        {
            if (!_hasReset)
                throw new InvalidOperationException("tactical-v3 environment must be reset before use");
        }

        private static void RequireSeat(PlayerId seat, string parameterName)
        {
            if (seat != PlayerId.Player0 && seat != PlayerId.Player1)
                throw new ArgumentOutOfRangeException(parameterName);
        }
    }

    public sealed class TacticalV3View
    {
        internal TacticalV3View(
            TacticalV3DecisionFrame decision,
            TacticalV3RewardBreakdown reward,
            PlayerId seat,
            int winner,
            bool terminated,
            bool truncated,
            string startProfileId,
            PlayerId referenceSeat)
        {
            Decision = decision ?? throw new ArgumentNullException(nameof(decision));
            Reward = reward ?? throw new ArgumentNullException(nameof(reward));
            Seat = seat;
            Winner = winner;
            Terminated = terminated;
            Truncated = truncated;
            StartProfileId = startProfileId ?? throw new ArgumentNullException(nameof(startProfileId));
            ReferenceSeat = referenceSeat;
        }

        public TacticalV3DecisionFrame Decision { get; }
        public TacticalV3RewardBreakdown Reward { get; }
        public PlayerId Seat { get; }
        public int Winner { get; }
        public bool Terminated { get; }
        public bool Truncated { get; }
        public string StartProfileId { get; }
        public PlayerId ReferenceSeat { get; }
    }
}
