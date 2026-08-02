using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Security.Cryptography;
using System.Text;

namespace HexWars.Engine.Rl
{
    /// <summary>A tactical-v2 catalog entry: a stable identifier (see <see cref="TacticalV2TemplateIds"/>)
    /// paired with the underlying barracks template it was derived from. Immutable.</summary>
    public sealed class TacticalV2Template
    {
        public TacticalV2Template(string id, UnitTemplate template)
        {
            Id = id;
            Template = template;
        }

        public string Id { get; }
        public UnitTemplate Template { get; }
    }

    /// <summary>Derives stable, deterministic catalog ids for tactical-v2 templates. An id is a
    /// lowercase ASCII slug of the template's sanitized name, followed by an eight-hex SHA-256
    /// prefix computed over the sanitized name plus all nine stats — so two templates that share a
    /// name but differ in any stat still get distinct ids, and the same name+stats always round-trip
    /// to the same id (contract stability across processes/runs).</summary>
    public static class TacticalV2TemplateIds
    {
        public static string From(UnitTemplate template)
        {
            string sanitizedName = UnitTemplate.Sanitize(template.Name);
            string slug = Slugify(sanitizedName);
            string suffix = HashSuffix(sanitizedName, template.Stats);
            return slug.Length == 0 ? suffix : slug + "-" + suffix;
        }

        /// <summary>Lowercases the sanitized name and collapses every run of non [a-z0-9] characters
        /// into a single '-', trimming leading/trailing hyphens.</summary>
        private static string Slugify(string sanitizedName)
        {
            var slug = new StringBuilder(sanitizedName.Length);
            foreach (char ch in sanitizedName.ToLowerInvariant())
            {
                bool alphanumeric = (ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9');
                if (alphanumeric) slug.Append(ch);
                else if (slug.Length > 0 && slug[slug.Length - 1] != '-') slug.Append('-');
            }
            while (slug.Length > 0 && slug[slug.Length - 1] == '-') slug.Length--;
            return slug.ToString();
        }

        private static string HashSuffix(string sanitizedName, UnitStats stats)
        {
            string payload = sanitizedName + "|" + string.Join(",", new[]
            {
                stats.Health, stats.Damage, stats.Defense, stats.Movement, stats.VerticalMovement,
                stats.Range, stats.RangeArc, stats.Vision, stats.VisionArc,
            });

            using var sha = SHA256.Create();
            byte[] hash = sha.ComputeHash(Encoding.UTF8.GetBytes(payload));
            var text = new StringBuilder(8);
            for (int i = 0; i < 4; i++) text.Append(hash[i].ToString("x2", CultureInfo.InvariantCulture));
            return text.ToString();
        }
    }

    /// <summary>Versioned separation labels understood by the profiled tactical-v2 start
    /// constructor. The distance-band mechanics live in <see cref="TacticalV2Layout"/>; these
    /// values are contract identities rather than free-form scenario tags.</summary>
    public static class TacticalV2StartSeparations
    {
        public const string LegacyMirrored = "legacy-mirrored";
        public const string Near = "near";
        public const string Medium = "medium";
        public const string Far = "far";

        public static bool IsKnown(string separation) =>
            separation == LegacyMirrored || separation == Near || separation == Medium || separation == Far;
    }

    /// <summary>An immutable, learner-relative starting-state profile. Unit counts describe live
    /// units, not observation/action slot capacity.</summary>
    public sealed class TacticalV2StartProfile
    {
        public TacticalV2StartProfile(
            string id,
            int learnerUnitCount,
            int opponentUnitCount,
            string separation)
        {
            Id = id ?? string.Empty;
            LearnerUnitCount = learnerUnitCount;
            OpponentUnitCount = opponentUnitCount;
            Separation = separation ?? string.Empty;
        }

        public string Id { get; }
        public int LearnerUnitCount { get; }
        public int OpponentUnitCount { get; }
        public string Separation { get; }
    }

    /// <summary>The exact profile catalog declared by the profiled-seeded-v1 placement contract.
    /// Return a fresh read-only collection so callers cannot mutate shared contract authority.</summary>
    public static class TacticalV2StartCatalog
    {
        public static IReadOnlyList<TacticalV2StartProfile> ProfiledSeededV1() => Array.AsReadOnly(new[]
        {
            new TacticalV2StartProfile("standard-3v3", 3, 3, TacticalV2StartSeparations.LegacyMirrored),
            new TacticalV2StartProfile("conversion-3v1-near", 3, 1, TacticalV2StartSeparations.Near),
            new TacticalV2StartProfile("conversion-3v1-medium", 3, 1, TacticalV2StartSeparations.Medium),
            new TacticalV2StartProfile("conversion-3v1-far", 3, 1, TacticalV2StartSeparations.Far),
            new TacticalV2StartProfile("conversion-2v1-near", 2, 1, TacticalV2StartSeparations.Near),
            new TacticalV2StartProfile("conversion-2v1-medium", 2, 1, TacticalV2StartSeparations.Medium),
            new TacticalV2StartProfile("conversion-2v1-far", 2, 1, TacticalV2StartSeparations.Far),
            new TacticalV2StartProfile("conversion-1v1-near", 1, 1, TacticalV2StartSeparations.Near),
            new TacticalV2StartProfile("conversion-1v1-medium", 1, 1, TacticalV2StartSeparations.Medium),
            new TacticalV2StartProfile("conversion-1v1-far", 1, 1, TacticalV2StartSeparations.Far),
        });
    }

    /// <summary>An immutable basis-point weight attached to a declared start profile.</summary>
    public sealed class TacticalV2StartWeight
    {
        public TacticalV2StartWeight(string profileId, int basisPoints)
        {
            ProfileId = profileId ?? string.Empty;
            BasisPoints = basisPoints;
        }

        public string ProfileId { get; }
        public int BasisPoints { get; }
    }

    /// <summary>Seeded profile-selection weights. Selection sorts by ordinal profile ID, so the
    /// result cannot depend on dictionary or JSON property iteration order.</summary>
    public sealed class TacticalV2StartDistribution
    {
        public static TacticalV2StartDistribution Empty { get; } =
            new TacticalV2StartDistribution(Array.Empty<TacticalV2StartWeight>());

        public TacticalV2StartDistribution(IEnumerable<TacticalV2StartWeight> weights)
        {
            if (weights == null) throw new ArgumentNullException(nameof(weights));
            Weights = Array.AsReadOnly(weights.ToArray());
        }

        public IReadOnlyList<TacticalV2StartWeight> Weights { get; }

        public IReadOnlyList<string> Validate(IEnumerable<string> declaredProfileIds)
        {
            if (declaredProfileIds == null) throw new ArgumentNullException(nameof(declaredProfileIds));

            var errors = new List<string>();
            var declared = new HashSet<string>(declaredProfileIds, StringComparer.Ordinal);
            var seen = new HashSet<string>(StringComparer.Ordinal);
            long sum = 0;

            foreach (TacticalV2StartWeight weight in Weights)
            {
                if (string.IsNullOrEmpty(weight.ProfileId))
                {
                    errors.Add("start distribution profile ids must not be empty");
                }
                else
                {
                    if (!seen.Add(weight.ProfileId))
                        errors.Add($"duplicate start distribution weight for profile '{weight.ProfileId}'");
                    if (!declared.Contains(weight.ProfileId))
                        errors.Add($"weight references undeclared start profile '{weight.ProfileId}'");
                }

                if (weight.BasisPoints < 0 || weight.BasisPoints > 10000)
                    errors.Add($"weight for start profile '{weight.ProfileId}' must be between 0 and 10000 basis points");
                sum += weight.BasisPoints;
            }

            foreach (string profileId in declared.OrderBy(id => id, StringComparer.Ordinal))
            {
                if (!seen.Contains(profileId))
                    errors.Add($"start distribution is missing declared profile '{profileId}'");
            }

            if (sum != 10000)
                errors.Add("start distribution weights must sum to 10000 basis points");

            return errors;
        }

        /// <summary>Select a profile identifier with a stable integer mixer rather than
        /// System.Random, whose implementation is not an engine contract across runtimes.</summary>
        public string Select(int seed)
        {
            var seen = new HashSet<string>(StringComparer.Ordinal);
            long sum = 0;
            foreach (TacticalV2StartWeight weight in Weights)
            {
                if (string.IsNullOrEmpty(weight.ProfileId) || !seen.Add(weight.ProfileId) ||
                    weight.BasisPoints < 0 || weight.BasisPoints > 10000)
                {
                    throw new InvalidOperationException("cannot select from an invalid start distribution");
                }
                sum += weight.BasisPoints;
            }
            if (sum != 10000)
                throw new InvalidOperationException("cannot select from an invalid start distribution");

            int roll = (int)(Mix(unchecked((uint)seed) ^ 0xC0A57A17u) % 10000u);
            int cumulative = 0;
            foreach (TacticalV2StartWeight weight in Weights.OrderBy(
                item => item.ProfileId, StringComparer.Ordinal))
            {
                cumulative += weight.BasisPoints;
                if (roll < cumulative) return weight.ProfileId;
            }

            throw new InvalidOperationException("valid start distribution did not select a profile");
        }

        private static uint Mix(uint value)
        {
            value += 0x9E3779B9u;
            value = (value ^ (value >> 16)) * 0x85EBCA6Bu;
            value = (value ^ (value >> 13)) * 0xC2B2AE35u;
            return value ^ (value >> 16);
        }
    }

    /// <summary>Versioned tactical-v2 template catalog: board/game config, a stable-id template list,
    /// roster limits, and reward shaping — plus seeded, symmetric starting-army sampling. Later tasks
    /// build layout, observation/action coding, and the training environments on top of this.</summary>
    public sealed class TacticalV2Config
    {
        private static GameConfig DefaultTacticalGame() => GameConfig.Default(
            biomesEnabled: false,
            captureCost: int.MaxValue,
            generatorsEnabled: false,
            fixedTemplateCount: BarracksCatalog.DefaultTemplates.Count,
            templateSlotCount: BarracksCatalog.DefaultTemplates.Count);

        public BoardGenConfig BoardGen { get; set; } = BoardGenConfig.Default();
        public GameConfig Game { get; set; } = DefaultTacticalGame();
        public IReadOnlyList<TacticalV2Template> Templates { get; set; } = Array.Empty<TacticalV2Template>();
        public int StartingUnitCount { get; set; }
        public int MaxControllableUnits { get; set; }
        public int MaxSteps { get; set; }
        public float ShapeScale { get; set; }
        public float StepPenalty { get; set; }
        public float ClosingWeight { get; set; }
        public float DrawCreditWeight { get; set; }
        public float PointsWeight { get; set; }
        public string PlacementPolicy { get; set; } = "symmetric-random-v1";
        public IReadOnlyList<TacticalV2StartProfile> StartProfiles { get; set; } =
            Array.Empty<TacticalV2StartProfile>();
        public TacticalV2StartDistribution StartDistribution { get; set; } =
            TacticalV2StartDistribution.Empty;

        /// <summary>Canonical catalog: the five default barracks templates, a three-unit starting
        /// army/roster cap, and the standard tactical reward shaping.</summary>
        public static TacticalV2Config Default() => new TacticalV2Config
        {
            BoardGen = BoardGenConfig.Default(),
            Game = DefaultTacticalGame(),
            Templates = Array.AsReadOnly(BarracksCatalog.DefaultTemplates
                .Select(template => new TacticalV2Template(TacticalV2TemplateIds.From(template), template))
                .ToArray()),
            StartingUnitCount = 3,
            MaxControllableUnits = 3,
            MaxSteps = DefaultMaxSteps(3, GameConfig.DefaultRoundCap),
            ShapeScale = 0.01f,
            StepPenalty = 0.005f,
            ClosingWeight = 0.02f,
            DrawCreditWeight = 0.25f,
            PointsWeight = 0.5f,
            PlacementPolicy = "symmetric-random-v1",
        };

        /// <summary>RL actions (each move/attack/deploy/end-turn call counts as one) both seats together
        /// can spend in a single round: one action per starting-unit slot plus an end-turn, per seat.
        /// This is the "~26 actions/round" a 12-unit tactical-v2 army spends, matching the 600-step
        /// tactical-v2 default that was tuned for the old 3-unit army and silently truncated (faking a
        /// draw) long before annihilation or the engine's own round cap for larger armies.</summary>
        public static int ActionsPerRound(int startingUnitCount) => 2 * (startingUnitCount + 1);

        /// <summary>The fewest RL-action MaxSteps that lets a <paramref name="startingUnitCount"/>-unit
        /// tactical-v2 army play out every round up to <paramref name="roundCap"/> before the RL step
        /// budget can truncate the episode first. Below this, the RL layer pre-empts the engine's own
        /// backstop and reports a truncated (not terminal) draw the game itself never reached.</summary>
        public static int MinimumMaxSteps(int startingUnitCount, int roundCap) =>
            ActionsPerRound(startingUnitCount) * roundCap;

        /// <summary>The recommended MaxSteps default for a new tactical-v2 scenario: the bare minimum
        /// needed to reach <paramref name="roundCap"/> (see <see cref="MinimumMaxSteps"/>), plus one
        /// extra round's worth of actions as headroom so the engine's own terminal check — not the RL
        /// step budget — always decides how the game ends.</summary>
        public static int DefaultMaxSteps(int startingUnitCount, int roundCap) =>
            MinimumMaxSteps(startingUnitCount, roundCap) + ActionsPerRound(startingUnitCount);

        public IReadOnlyList<string> Validate()
        {
            var errors = new List<string>();

            if (Templates == null || Templates.Count == 0)
            {
                errors.Add("template catalog must not be empty");
            }
            else
            {
                var seenIds = new HashSet<string>(StringComparer.Ordinal);
                foreach (TacticalV2Template item in Templates)
                {
                    if (string.IsNullOrEmpty(item.Id)) errors.Add("template ids must not be empty");
                    else if (!seenIds.Add(item.Id)) errors.Add($"duplicate template id '{item.Id}'");
                }
            }

            if (PlacementPolicy == "symmetric-random-v1")
            {
                if (StartingUnitCount < 1 || StartingUnitCount > 12)
                    errors.Add("starting unit count must be between 1 and 12");
                if (MaxControllableUnits != StartingUnitCount)
                    errors.Add("max controllable units must equal starting unit count");
            }
            else if (PlacementPolicy == "profiled-seeded-v1")
            {
                if (StartingUnitCount != 3 || MaxControllableUnits != 3)
                {
                    errors.Add(
                        "profiled-seeded-v1 requires starting unit count and max controllable units to equal 3");
                }

                if (StartProfiles == null)
                {
                    errors.Add("profiled-seeded-v1 start profile catalog must not be null");
                }
                else
                {
                    var seenProfileIds = new HashSet<string>(StringComparer.Ordinal);
                    foreach (TacticalV2StartProfile profile in StartProfiles)
                    {
                        if (string.IsNullOrEmpty(profile.Id))
                            errors.Add("profile ids must not be empty");
                        else if (!seenProfileIds.Add(profile.Id))
                            errors.Add($"duplicate start profile id '{profile.Id}'");

                        if (profile.LearnerUnitCount < 1 || profile.LearnerUnitCount > MaxControllableUnits)
                        {
                            errors.Add(
                                $"start profile '{profile.Id}' learner unit count must be between 1 and max controllable units");
                        }
                        if (profile.OpponentUnitCount < 1 || profile.OpponentUnitCount > MaxControllableUnits)
                        {
                            errors.Add(
                                $"start profile '{profile.Id}' opponent unit count must be between 1 and max controllable units");
                        }
                        if (!TacticalV2StartSeparations.IsKnown(profile.Separation))
                            errors.Add($"start profile '{profile.Id}' has unknown separation '{profile.Separation}'");
                    }

                    if (!HasExactProfiledSeededV1Catalog(StartProfiles))
                    {
                        errors.Add(
                            "profiled-seeded-v1 requires the exact versioned start profile catalog");
                    }

                    if (StartDistribution == null)
                    {
                        errors.Add("profiled-seeded-v1 start distribution must not be null");
                    }
                    else
                    {
                        errors.AddRange(StartDistribution.Validate(StartProfiles.Select(profile => profile.Id)));
                    }
                }
            }
            else
            {
                errors.Add(
                    "placement policy must be 'symmetric-random-v1' or 'profiled-seeded-v1'");
            }

            return errors;
        }

        private static bool HasExactProfiledSeededV1Catalog(
            IReadOnlyList<TacticalV2StartProfile> actual)
        {
            IReadOnlyList<TacticalV2StartProfile> expected = TacticalV2StartCatalog.ProfiledSeededV1();
            if (actual.Count != expected.Count) return false;

            for (int index = 0; index < expected.Count; index++)
            {
                TacticalV2StartProfile left = actual[index];
                TacticalV2StartProfile right = expected[index];
                if (left.Id != right.Id ||
                    left.LearnerUnitCount != right.LearnerUnitCount ||
                    left.OpponentUnitCount != right.OpponentUnitCount ||
                    left.Separation != right.Separation)
                {
                    return false;
                }
            }
            return true;
        }

        /// <summary>Samples <see cref="StartingUnitCount"/> templates with replacement, seeded so both
        /// seats (given the same seed) draw an identical, symmetric starting army.</summary>
        public IReadOnlyList<TacticalV2Template> SampleStartingArmy(int seed)
        {
            var result = new List<TacticalV2Template>(StartingUnitCount);
            var rng = new Random(seed ^ 0x5A17);
            while (result.Count < StartingUnitCount)
                result.Add(Templates[rng.Next(Templates.Count)]);
            return result;
        }
    }
}
