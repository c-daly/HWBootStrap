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

    /// <summary>Versioned tactical-v2 template catalog: board/game config, a stable-id template list,
    /// roster limits, and reward shaping — plus seeded, symmetric starting-army sampling. Later tasks
    /// build layout, observation/action coding, and the training environments on top of this.</summary>
    public sealed class TacticalV2Config
    {
        public BoardGenConfig BoardGen { get; set; } = BoardGenConfig.Default();
        public GameConfig Game { get; set; } = GameConfig.Default(biomesEnabled: false);
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

        /// <summary>Canonical catalog: the five default barracks templates, a three-unit starting
        /// army/roster cap, and the standard tactical reward shaping.</summary>
        public static TacticalV2Config Default() => new TacticalV2Config
        {
            BoardGen = BoardGenConfig.Default(),
            Game = GameConfig.Default(biomesEnabled: false),
            Templates = Array.AsReadOnly(BarracksCatalog.DefaultTemplates
                .Select(template => new TacticalV2Template(TacticalV2TemplateIds.From(template), template))
                .ToArray()),
            StartingUnitCount = 3,
            MaxControllableUnits = 3,
            MaxSteps = 600,
            ShapeScale = 0.01f,
            StepPenalty = 0.005f,
            ClosingWeight = 0.02f,
            DrawCreditWeight = 0.25f,
            PointsWeight = 0.5f,
            PlacementPolicy = "symmetric-random-v1",
        };

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

            if (StartingUnitCount < 1 || StartingUnitCount > 12)
                errors.Add("starting unit count must be between 1 and 12");
            if (MaxControllableUnits != StartingUnitCount)
                errors.Add("max controllable units must equal starting unit count");
            if (PlacementPolicy != "symmetric-random-v1")
                errors.Add("placement policy must be 'symmetric-random-v1'");

            return errors;
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
