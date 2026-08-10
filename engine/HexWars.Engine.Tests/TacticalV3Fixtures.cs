using System.Collections.Generic;
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
        public static GameConfig CloneGame(GameConfig source, bool? fogOfWar = null)
        {
            var terrain = new Dictionary<TerrainType, TerrainDef>();
            foreach (TerrainType terrainType in System.Enum.GetValues(typeof(TerrainType)))
                terrain.Add(terrainType, source.Terrain(terrainType));

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
                fogOfWar: fogOfWar ?? source.FogOfWar,
                maxDesignPointCost: source.MaxDesignPointCost,
                fixedTemplateCount: source.FixedTemplateCount,
                templateSlotCount: source.TemplateSlotCount);
        }

        public static TacticalV3CapacityProfile ExperimentalCapacity(
            int? maxCells = null, int? maxCandidates = null) =>
            new TacticalV3CapacityProfile(
                maxCells ?? 512,
                maxUnits: 64,
                maxTemplates: 32,
                maxCapabilityDefinitions: 128,
                maxCapabilityAllocations: 2048,
                maxRules: 128,
                maxMemoryRecords: 64,
                maxRelations: 65536,
                maxCandidates: maxCandidates ?? 32768);

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
        public static GameState RewardStart(int unitCost = 10, int round = 1)

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
                    new Unit(1, PlayerId.Player0, TestStates.Cost(unitCost), new HexCoord(0, 0), 0),
                }),
                new PlayerState(PlayerId.Player1, 0, unitsOnBoard: new[]
                {
                    new Unit(2, PlayerId.Player1, TestStates.Cost(unitCost), new HexCoord(1, 0), 0),
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
