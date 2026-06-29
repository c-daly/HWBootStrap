using System.Collections.Generic;

namespace HexWars.Engine
{
    /// <summary>
    /// All tunable rules values — balancing is data, not code. Placeholder defaults; every value
    /// is provisional (balance and fun win). Vertical movement / vertical reach cost a flat 1 per
    /// level by design (uniform 1pt/hex), so they are not tunables here.
    /// </summary>
    public sealed class GameConfig
    {
        private readonly IReadOnlyDictionary<TerrainType, TerrainDef> _terrain;

        /// <summary>Points each player starts with.</summary>
        public int StartingPoints { get; }

        /// <summary>Fraction of a destroyed unit's build cost paid to the killer's owner.</summary>
        public double BountyRate { get; }

        public int GeneratorCost { get; }
        public int GeneratorOutput { get; }
        public int GeneratorHealth { get; }

        /// <summary>Minimum damage a hit deals after defense (e.g. 0 or 1).</summary>
        public int DamageFloor { get; }

        /// <summary>Bonus damage per level of attacker high-ground advantage.</summary>
        public int DmgHighGroundBonus { get; }

        /// <summary>Bonus horizontal range per level of attacker high-ground advantage.</summary>
        public int RangeHighGroundBonus { get; }

        /// <summary>Round at which the stalemate backstop ends the game by total value.</summary>
        public int RoundCap { get; }

        /// <summary>Points to add a unit design to the barracks (default 0 = free; configurable).</summary>
        public int DesignFee { get; }

        /// <summary>Multiplier on a unit's PointCost when deploying a clone of a barracks template.</summary>
        public double DeployCostMultiplier { get; }

        /// <summary>The turn-structure rule (default <see cref="AllUnitsPolicy"/>).</summary>
        public ITurnPolicy TurnPolicy { get; }

        /// <summary>When false, every terrain looks identical (flat plains: move 1, no defense/concealment,
        /// passable) regardless of the tile's biome — i.e. biomes are mechanically off. The board can still
        /// generate/render varied terrain; it just has no gameplay effect. Default true.</summary>
        public bool BiomesEnabled { get; }

        /// <summary>Which win conditions are active (any combination). Default: annihilation only.</summary>
        public WinBy WinConditions { get; }
        /// <summary>Flat cost to capture a hex (Phase 1; value-scaling comes with generators in Phase 2).</summary>
        public int CaptureCost { get; }
        /// <summary>Banked-points threshold for an Economy win.</summary>
        public int EconomyWinThreshold { get; }
        /// <summary>Score-composite weights (see WinCheck.Score).</summary>
        public int ScoreKills { get; }
        public int ScorePoints { get; }
        public int ScoreArmy { get; }
        public int ScoreTerritory { get; }

        /// <summary>Per-turn upkeep as a fraction of a player's generator income.</summary>
        public double UpkeepFactor { get; }
        /// <summary>Capture cost on a generator hex = max(CaptureCost, round(CaptureFactor × that generator's income)).</summary>
        public double CaptureFactor { get; }
        /// <summary>Build cost of a generator = round(BuildFactor × GeneratorOutput × strength).</summary>
        public double BuildFactor { get; }

        /// <summary>When true, the territory rules apply: deploy/build require control, and (with
        /// ClaimEndsTurn) claiming is a turn-exclusive opening action. Default false = base game.</summary>
        public bool TerritoryMode { get; }
        /// <summary>When true (and TerritoryMode), a claim must be the turn's first army action and
        /// immediately ends the turn — the vulnerable-as-you-expand tempo. Default true.</summary>
        public bool ClaimEndsTurn { get; }

        /// <summary>When true (TerritoryMode), a generator may be built on ANY hex you control, not only the
        /// hex a unit stands on. Default false = build only under a unit.</summary>
        public bool BuildAnywhere { get; }
        /// <summary>Passive income per controlled hex per turn (the no-generator economy). Default 0 = income
        /// comes only from generators.</summary>
        public int TerritoryIncome { get; }
        /// <summary>When false, generators can't be built (the pure passive-income economy). Default true.</summary>
        public bool GeneratorsEnabled { get; }
        /// <summary>Fraction of a player's banked points that decays at the start of each of their turns
        /// (use-it-or-lose-it: spend on army or lose it). Self-targeting — big hoards bleed, small balances
        /// barely notice. Default 0 = no decay.</summary>
        public double PointDecay { get; }

        // Uniform terrain used for every tile when BiomesEnabled is false.
        private static readonly TerrainDef FlatTerrain = new TerrainDef(moveCost: 1, concealment: 0, defense: 0, passable: true);

        public GameConfig(
            IReadOnlyDictionary<TerrainType, TerrainDef> terrain,
            int startingPoints = 12,
            double bountyRate = 0.5,
            int generatorCost = 2,
            int generatorOutput = 1,
            int generatorHealth = 3,
            int damageFloor = 0,
            int dmgHighGroundBonus = 1,
            int rangeHighGroundBonus = 1,
            int roundCap = 100, // backstop only — games are meant to end by annihilation, so give them room
            int designFee = 0,
            double deployCostMultiplier = 1.0,
            ITurnPolicy? turnPolicy = null,
            bool biomesEnabled = true,
            WinBy winConditions = WinBy.Annihilation,
            int captureCost = 3,
            int economyWinThreshold = 200,
            int scoreKills = 1,
            int scorePoints = 1,
            int scoreArmy = 1,
            int scoreTerritory = 1,
            double upkeepFactor = 0.25,
            double captureFactor = 4.0,
            double buildFactor = 4.0,
            bool territoryMode = false,
            bool claimEndsTurn = true,
            bool buildAnywhere = false,
            int territoryIncome = 0,
            bool generatorsEnabled = true,
            double pointDecay = 0.0)
        {
            _terrain = terrain;
            StartingPoints = startingPoints;
            BountyRate = bountyRate;
            GeneratorCost = generatorCost;
            GeneratorOutput = generatorOutput;
            GeneratorHealth = generatorHealth;
            DamageFloor = damageFloor;
            DmgHighGroundBonus = dmgHighGroundBonus;
            RangeHighGroundBonus = rangeHighGroundBonus;
            RoundCap = roundCap;
            DesignFee = designFee;
            DeployCostMultiplier = deployCostMultiplier;
            TurnPolicy = turnPolicy ?? new AllUnitsPolicy();
            BiomesEnabled = biomesEnabled;
            WinConditions = winConditions;
            CaptureCost = captureCost;
            EconomyWinThreshold = economyWinThreshold;
            ScoreKills = scoreKills;
            ScorePoints = scorePoints;
            ScoreArmy = scoreArmy;
            ScoreTerritory = scoreTerritory;
            UpkeepFactor = upkeepFactor;
            CaptureFactor = captureFactor;
            BuildFactor = buildFactor;
            TerritoryMode = territoryMode;
            ClaimEndsTurn = claimEndsTurn;
            BuildAnywhere = buildAnywhere;
            TerritoryIncome = territoryIncome;
            GeneratorsEnabled = generatorsEnabled;
            PointDecay = pointDecay;
        }

        /// <summary>Modifier table for the given terrain. With biomes off, every tile reads as flat plains.</summary>
        public TerrainDef Terrain(TerrainType type) => BiomesEnabled ? _terrain[type] : FlatTerrain;

        /// <summary>Default ruleset (placeholder values, all tunable). Pass <paramref name="biomesEnabled"/>
        /// = false to make terrain mechanically inert (all tiles flat plains), and <paramref name="turnPolicy"/>
        /// to override the turn structure (null = the default <see cref="AllUnitsPolicy"/>).</summary>
        public static GameConfig Default(bool biomesEnabled = true, ITurnPolicy? turnPolicy = null,
            WinBy winConditions = WinBy.Annihilation, int captureCost = 3, int economyWinThreshold = 200,
            int scoreKills = 1, int scorePoints = 1, int scoreArmy = 1, int scoreTerritory = 1,
            double upkeepFactor = 0.25, double captureFactor = 4.0, double buildFactor = 4.0,
            int generatorOutput = 1,
            int startingPoints = 12,
            int damageFloor = 0,
            bool territoryMode = false,
            bool claimEndsTurn = true,
            bool buildAnywhere = false,
            int territoryIncome = 0,
            bool generatorsEnabled = true,
            double pointDecay = 0.0) =>
            new GameConfig(new Dictionary<TerrainType, TerrainDef>
        {
            { TerrainType.Plains, new TerrainDef(moveCost: 1, concealment: 0, defense: 0, passable: true) },
            { TerrainType.Forest, new TerrainDef(moveCost: 2, concealment: 2, defense: 1, passable: true) },
            { TerrainType.Rough,  new TerrainDef(moveCost: 2, concealment: 1, defense: 1, passable: true) },
            { TerrainType.Water,  new TerrainDef(moveCost: 3, concealment: 0, defense: 0, passable: true) },
        }, turnPolicy: turnPolicy, biomesEnabled: biomesEnabled, winConditions: winConditions,
           captureCost: captureCost, economyWinThreshold: economyWinThreshold,
           scoreKills: scoreKills, scorePoints: scorePoints, scoreArmy: scoreArmy, scoreTerritory: scoreTerritory,
           upkeepFactor: upkeepFactor, captureFactor: captureFactor, buildFactor: buildFactor,
           generatorOutput: generatorOutput, startingPoints: startingPoints, damageFloor: damageFloor,
           territoryMode: territoryMode, claimEndsTurn: claimEndsTurn,
           buildAnywhere: buildAnywhere, territoryIncome: territoryIncome,
           generatorsEnabled: generatorsEnabled, pointDecay: pointDecay);
    }
}
