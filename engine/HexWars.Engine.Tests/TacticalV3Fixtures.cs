using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using HexWars.Engine;
using HexWars.Engine.Rl;

namespace HexWars.Engine.Tests
{
    internal sealed class TacticalV3Fixture
    {
        public TacticalV3Fixture(GameState state, TacticalV3SeatObservationSource source,
            ILegalCandidateSource candidates, IActionResolver resolver)
        {
            State = state;
            Source = source;
            Candidates = candidates;
            Resolver = resolver;
        }

        public GameState State { get; }
        public TacticalV3SeatObservationSource Source { get; }
        public ILegalCandidateSource Candidates { get; }
        public IActionResolver Resolver { get; }

        public static TacticalV3Fixture Standard(int seed) => TacticalV3Fixtures.Standard(seed);
    }

    internal static class TacticalV3Fixtures
    {
        public static GameConfig CloneGame(
            GameConfig source,
            bool? fogOfWar = null,
            int? startingPoints = null,
            double? bountyRate = null,
            int? captureCost = null,
            bool? territoryMode = null,
            int? territoryIncome = null)
        {
            var terrain = new Dictionary<TerrainType, TerrainDef>();
            foreach (TerrainType terrainType in System.Enum.GetValues(typeof(TerrainType)))
                terrain.Add(terrainType, source.Terrain(terrainType));

            return new GameConfig(
                terrain,
                startingPoints: startingPoints ?? source.StartingPoints,
                bountyRate: bountyRate ?? source.BountyRate,
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
                captureCost: captureCost ?? source.CaptureCost,
                economyWinThreshold: source.EconomyWinThreshold,
                scoreKills: source.ScoreKills,
                scorePoints: source.ScorePoints,
                scoreArmy: source.ScoreArmy,
                scoreTerritory: source.ScoreTerritory,
                upkeepFactor: source.UpkeepFactor,
                captureFactor: source.CaptureFactor,
                buildFactor: source.BuildFactor,
                territoryMode: territoryMode ?? source.TerritoryMode,
                claimEndsTurn: source.ClaimEndsTurn,
                buildAnywhere: source.BuildAnywhere,
                territoryIncome: territoryIncome ?? source.TerritoryIncome,
                generatorsEnabled: source.GeneratorsEnabled,
                pointDecay: source.PointDecay,
                fogOfWar: fogOfWar ?? source.FogOfWar,
                maxDesignPointCost: source.MaxDesignPointCost,
                fixedTemplateCount: source.FixedTemplateCount,
                templateSlotCount: source.TemplateSlotCount);
        }

        public static TacticalV3CapacityProfile ExperimentalCapacity(
            int? maxCells = null,
            int? maxUnits = null,
            int? maxTemplates = null,
            int? maxCapabilityDefinitions = null,
            int? maxCapabilityAllocations = null,
            int? maxRules = null,
            int? maxMemoryRecords = null,
            int? maxRelations = null,
            int? maxCandidates = null) =>
            new TacticalV3CapacityProfile(
                maxCells ?? 512,
                maxUnits: maxUnits ?? 64,
                maxTemplates: maxTemplates ?? 32,
                maxCapabilityDefinitions: maxCapabilityDefinitions ?? 128,
                maxCapabilityAllocations: maxCapabilityAllocations ?? 2048,
                maxRules: maxRules ?? 128,
                maxMemoryRecords: maxMemoryRecords ?? 64,
                maxRelations: maxRelations ?? 65536,
                maxCandidates: maxCandidates ?? 32768);

        public static TacticalV2Config CloneMatch(
            TacticalV2Config source,
            BoardGenConfig? board = null,
            GameConfig? game = null,
            IReadOnlyList<TacticalV2Template>? templates = null) =>
            new TacticalV2Config
            {
                BoardGen = board ?? source.BoardGen,
                Game = game ?? source.Game,
                Templates = templates ?? source.Templates,
                StartingUnitCount = source.StartingUnitCount,
                MaxControllableUnits = source.MaxControllableUnits,
                MaxSteps = source.MaxSteps,
                ShapeScale = source.ShapeScale,
                StepPenalty = source.StepPenalty,
                ClosingWeight = source.ClosingWeight,
                DrawCreditWeight = source.DrawCreditWeight,
                PointsWeight = source.PointsWeight,
                PlacementPolicy = source.PlacementPolicy,
                StartProfiles = source.StartProfiles,
                StartDistribution = source.StartDistribution,
            };

        public static TacticalV3RewardConfig UncheckedReward(
            float terminalWin = 1f,
            float terminalNonWin = -1f,
            float materialAdjustmentBound = 0.20f,
            float timePressureBound = 0.05f,
            float pointsWeight = 0.5f)
        {
            var reward = (TacticalV3RewardConfig)RuntimeHelpers.GetUninitializedObject(
                typeof(TacticalV3RewardConfig));
            SetAutoProperty(reward, nameof(TacticalV3RewardConfig.TerminalWin), terminalWin);
            SetAutoProperty(reward, nameof(TacticalV3RewardConfig.TerminalNonWin), terminalNonWin);
            SetAutoProperty(reward, nameof(TacticalV3RewardConfig.MaterialAdjustmentBound), materialAdjustmentBound);
            SetAutoProperty(reward, nameof(TacticalV3RewardConfig.TimePressureBound), timePressureBound);
            SetAutoProperty(reward, nameof(TacticalV3RewardConfig.PointsWeight), pointsWeight);
            return reward;
        }

        private static void SetAutoProperty<T>(object target, string propertyName, T value)
        {
            FieldInfo? field = target.GetType().GetField(
                $"<{propertyName}>k__BackingField", BindingFlags.Instance | BindingFlags.NonPublic);
            if (field == null) throw new MissingFieldException(target.GetType().FullName, propertyName);
            field.SetValue(target, value);
        }

        public static TacticalV2Config Match(int width = 13, int height = 9)
        {
            TacticalV2Config match = TacticalV2Config.Default();
            match.BoardGen = new BoardGenConfig(width: width, height: height);
            return match;
        }

        public static TacticalV3Config Config(int width = 13, int height = 9) =>
            new TacticalV3Config(
                Match(width, height),
                ExperimentalCapacity(),

                new TacticalV3RewardConfig(+1f, -1f, 0.20f, 0.05f, 0.5f));

        public static TacticalV3Config CapacityBoundConfig()
        {
            TacticalV3Config source = Config();
            return new TacticalV3Config(
                source.Match,
                ExperimentalCapacity(maxUnits: 64, maxRelations: 1346, maxCandidates: 11656),
                source.Reward);
        }

        public static GameState DenseCapacityState(
            TacticalV3Config config, int totalUnits = 64, int seed = 31)
        {
            TacticalV2Start start = new TacticalV2Layout(config.Match).NewGame(seed);
            var learnerZone = new HashSet<HexCoord>(
                start.State.Board.DeploymentZone(PlayerId.Player0));
            HexCoord[] cells = start.State.Board.Tiles
                .Select(tile => tile.Coord)
                .OrderBy(cell => learnerZone.Contains(cell) ? 1 : 0)
                .ThenBy(cell => cell.Q)
                .ThenBy(cell => cell.R)
                .Take(totalUnits)
                .ToArray();
            UnitTemplate[] templates = config.Match.Templates
                .Select(template => template.Template)
                .ToArray();
            UnitStats stats = TestStates.Stats(
                health: 2, damage: 1, movement: 20, verticalMovement: 20,
                range: 20, rangeArc: 20, vision: 20, visionArc: 20);
            int learnerUnits = totalUnits / 2;
            Unit[] player0Units = Enumerable.Range(0, learnerUnits)
                .Select(index => new Unit(
                    index + 1, PlayerId.Player0, stats, cells[index],
                    start.State.Board.TileAt(cells[index]).Elevation))
                .ToArray();
            Unit[] player1Units = Enumerable.Range(learnerUnits, totalUnits - learnerUnits)
                .Select(index => new Unit(
                    index + 1, PlayerId.Player1, stats, cells[index],
                    start.State.Board.TileAt(cells[index]).Elevation))
                .ToArray();
            var players = new[]
            {
                new PlayerState(PlayerId.Player0, 1000, templates, player0Units),
                new PlayerState(PlayerId.Player1, 1000, templates, player1Units),
            };
            return new GameState(
                start.State.Board, config.Match.Game, players,
                PlayerId.Player0, round: 1, nextEntityId: totalUnits + 1);
        }


        public static TacticalV3DuelEnv Env(TacticalV3Config? config = null) =>
            new TacticalV3DuelEnv(config ?? Config());

        public static TacticalV3Config ProfiledConfig(string selectedProfileId = "standard-3v3")
        {
            TacticalV2Config match = Match();
            match.PlacementPolicy = "profiled-seeded-v1";
            match.StartProfiles = TacticalV2StartCatalog.ProfiledSeededV1();
            match.StartDistribution = new TacticalV2StartDistribution(match.StartProfiles.Select(profile =>
                new TacticalV2StartWeight(profile.Id, profile.Id == selectedProfileId ? 10000 : 0)));
            return new TacticalV3Config(match, ExperimentalCapacity(),
                new TacticalV3RewardConfig(+1f, -1f, 0.20f, 0.05f, 0.5f));
        }

        public static TacticalV3Fixture Standard(int seed)
        {
            TacticalV3Config config = Config();
            TacticalV2Layout layout = new TacticalV2Layout(config.Match);
            TacticalV3SeatObservationSource source = new TacticalV3SeatObservationSource(config);
            var projector = new TacticalV3CandidateProjector();
            return new TacticalV3Fixture(layout.NewGame(seed).State, source,
                new TacticalV3LegalCandidateSource(source, projector, config.Capacity),
                new TacticalV3ActionResolver());
        }
        public static GameState RewardStart(int unitCost = 10, int round = 1) =>
            RewardStart(TestStates.Cost(unitCost), round);

        public static GameState RewardStart(UnitStats stats, int round = 1)
        {
            var board = new Board(new[]
            {
                new Tile(new HexCoord(0, 0), 0, TerrainType.Plains),
                new Tile(new HexCoord(1, 0), 0, TerrainType.Plains),
            });
            return new GameState(board, GameConfig.Default(), new[]
            {
                new PlayerState(PlayerId.Player0, 0, unitsOnBoard: new[]
                {
                    new Unit(1, PlayerId.Player0, stats, new HexCoord(0, 0), 0),
                }),
                new PlayerState(PlayerId.Player1, 0, unitsOnBoard: new[]
                {
                    new Unit(2, PlayerId.Player1, stats, new HexCoord(1, 0), 0),
                }),
            }, PlayerId.Player0, round, 3);
        }

        public static GameState Terminal(PlayerId? winner)
        {
            GameState start = RewardStart();
            return new GameState(start.Board, start.Config, start.Players, start.ActivePlayer,
                start.Round, start.NextEntityId, isGameOver: true, winner: winner);
        }

        public static GameState WithDamage(GameState source, PlayerId player, int damage)
        {
            PlayerState original = source.Player(player);
            Unit[] units = new Unit[original.UnitsOnBoard.Count];
            for (int index = 0; index < units.Length; index++)
                units[index] = original.UnitsOnBoard[index].WithDamage(damage);
            var replacement = new PlayerState(player, original.Points, original.Barracks, units,
                original.Generators, original.DestroyedValue);
            var players = new[] { source.Player(PlayerId.Player0), source.Player(PlayerId.Player1) };
            players[(int)player] = replacement;
            return new GameState(source.Board, source.Config, players, source.ActivePlayer, source.Round,
                source.NextEntityId, source.IsGameOver, source.Winner, source.MovedUnitIds,
                source.AttackedUnitIds, source.MovementSpent);
        }

        public static GameState AtRound(GameState source, int round) =>
            new GameState(source.Board, source.Config, source.Players, source.ActivePlayer, round,
                source.NextEntityId, source.IsGameOver, source.Winner, source.MovedUnitIds,
                source.AttackedUnitIds, source.MovementSpent);

        public static GameState WithTerminal(GameState source, PlayerId? winner) =>
            new GameState(source.Board, source.Config, source.Players, source.ActivePlayer, source.Round,
                source.NextEntityId, isGameOver: true, winner: winner,
                movedUnitIds: source.MovedUnitIds, attackedUnitIds: source.AttackedUnitIds,
                movementSpent: source.MovementSpent);

        public static TacticalV3Reward Tracker(GameState initialState, PlayerId learnerSeat)
        {
            var tracker = new TacticalV3Reward(Config().Reward);
            tracker.Reset(initialState, learnerSeat);
            return tracker;
        }

    }
}
