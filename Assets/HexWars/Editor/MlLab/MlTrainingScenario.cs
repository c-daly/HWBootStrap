using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using HexWars.Presentation;
using UnityEngine;

namespace HexWars.Presentation.EditorTools.MlLab
{
    public sealed class MlTrainingScenario
    {
        public int SchemaVersion { get; set; }
        public string Id { get; set; } = string.Empty;
        public string Name { get; set; } = string.Empty;
        public MlEnvironmentContract Environment { get; set; }
        public MlTrainingBoard Board { get; set; }
        public MlTrainingRules Rules { get; set; }
        public MlTrainingEpisode Episode { get; set; }
        public MlTacticalReward TacticalReward { get; set; }
        public MlAdaptiveReward AdaptiveReward { get; set; }
        public MlTrainingAdaptive Adaptive { get; set; }

        public IReadOnlyList<string> Validate()
        {
            var errors = new List<string>();
            if (SchemaVersion != 1) errors.Add("schema version must be 1");
            if (string.IsNullOrWhiteSpace(Id)) errors.Add("id must be a non-empty string");
            if (string.IsNullOrWhiteSpace(Name)) errors.Add("name must be a non-empty string");

            ValidateBoard(errors);
            ValidateRules(errors);
            if (Episode == null) errors.Add("episode section is required");
            else if (Episode.MaxSteps <= 0) errors.Add("max steps must be positive");

            if (Environment == MlEnvironmentContract.TacticalV1)
            {
                if (TacticalReward == null)
                    errors.Add("tactical-v1 requires a tactical reward section");
                else if (!TacticalReward.IsFinite())
                    errors.Add("tactical reward values must be finite");
                if (AdaptiveReward != null)
                    errors.Add("adaptive reward section is not valid for tactical-v1");
                if (Adaptive != null)
                    errors.Add("adaptive section is not valid for tactical-v1");
            }
            else if (Environment == MlEnvironmentContract.AdaptiveV1)
            {
                if (AdaptiveReward == null)
                    errors.Add("adaptive-v1 requires an adaptive reward section");
                else if (!AdaptiveReward.IsFinite())
                    errors.Add("adaptive reward values must be finite");
                if (TacticalReward != null)
                    errors.Add("tactical reward section is not valid for adaptive-v1");
                ValidateAdaptive(errors);
            }
            else
            {
                errors.Add("environment must be tactical-v1 or adaptive-v1");
            }

            return errors;
        }

        public MlTrainingScenario Clone() =>
            MlTrainingScenarioFile.Parse(MlTrainingScenarioFile.Serialize(this), "scenario clone");

        void ValidateBoard(List<string> errors)
        {
            if (Board == null)
            {
                errors.Add("board section is required");
                return;
            }

            if (Board.Width <= 0) errors.Add("board width must be positive");
            if (Board.Height <= 0) errors.Add("board height must be positive");
            if (Board.MaxElevation <= 0) errors.Add("board max elevation must be positive");
            if (Board.ZoneDepth <= 0) errors.Add("board zone depth must be positive");
            if (Board.Width > 0 && Board.ZoneDepth > Board.Width - Board.ZoneDepth)
                errors.Add("deployment zones overlap");
            if (!IsFinite(Board.FlatChance) || Board.FlatChance < 0 || Board.FlatChance > 1)
                errors.Add("board flat chance must be within [0,1]");
            if (Board.PlainsWeight < 0) errors.Add("plains weight must be non-negative");
            if (Board.ForestWeight < 0) errors.Add("forest weight must be non-negative");
            if (Board.RoughWeight < 0) errors.Add("rough weight must be non-negative");
            if (Board.WaterWeight < 0) errors.Add("water weight must be non-negative");
            if ((long)Board.PlainsWeight + Board.ForestWeight + Board.RoughWeight +
                Board.WaterWeight <= 0)
                errors.Add("terrain weight sum must be positive");
        }

        void ValidateRules(List<string> errors)
        {
            if (Rules == null)
            {
                errors.Add("rules section is required");
                return;
            }

            if (Rules.ActionsPerTurn < 0) errors.Add("actions per turn must be non-negative");
            if (Rules.RoundCap <= 0) errors.Add("round cap must be positive");
            if (!IsFinite(Rules.BountyRate)) errors.Add("bounty rate must be finite");
            if (!IsFinite(Rules.DeployCostMultiplier))
                errors.Add("deploy cost multiplier must be finite");
        }

        void ValidateAdaptive(List<string> errors)
        {
            if (Adaptive == null)
            {
                errors.Add("adaptive-v1 requires an adaptive section");
                return;
            }

            if (Adaptive.StartingUnitCount < 1 || Adaptive.StartingUnitCount > 24)
                errors.Add("adaptive starting unit count must be between 1 and 24");
            if (Board != null && Board.Height > 0 && Board.ZoneDepth > 0 &&
                (long)Board.Height * Board.ZoneDepth < Adaptive.StartingUnitCount)
                errors.Add("adaptive deployment cells must cover the starting unit count");
            if (Adaptive.StartingArmyBudget < 20L * Adaptive.StartingUnitCount)
                errors.Add("adaptive starting army budget is insufficient for the starting unit count");
            if (Adaptive.MaxDesignPointCost <= 0)
                errors.Add("adaptive max design point cost must be positive");
        }

        static bool IsFinite(double value) => !double.IsNaN(value) && !double.IsInfinity(value);
    }

    public sealed class MlTrainingBoard
    {
        public int Width { get; set; }
        public int Height { get; set; }
        public int MaxElevation { get; set; }
        public int ZoneDepth { get; set; }
        public double FlatChance { get; set; }
        public int PlainsWeight { get; set; }
        public int ForestWeight { get; set; }
        public int RoughWeight { get; set; }
        public int WaterWeight { get; set; }
    }

    public sealed class MlTrainingRules
    {
        public int ActionsPerTurn { get; set; }
        public int RoundCap { get; set; }
        public int StartingPoints { get; set; }
        public bool FogOfWar { get; set; }
        public bool BiomesEnabled { get; set; }
        public double BountyRate { get; set; }
        public double DeployCostMultiplier { get; set; }
        public int GeneratorCost { get; set; }
        public int GeneratorOutput { get; set; }
        public int GeneratorHealth { get; set; }
    }

    public sealed class MlTrainingEpisode
    {
        public int MaxSteps { get; set; }
    }

    public sealed class MlTacticalReward
    {
        public float ShapeScale { get; set; }
        public float StepPenalty { get; set; }
        public float ClosingWeight { get; set; }
        public float DrawCreditWeight { get; set; }
        public float PointsWeight { get; set; }

        internal bool IsFinite() =>
            IsFiniteValue(ShapeScale) &&
            IsFiniteValue(StepPenalty) &&
            IsFiniteValue(ClosingWeight) &&
            IsFiniteValue(DrawCreditWeight) &&
            IsFiniteValue(PointsWeight);

        static bool IsFiniteValue(float value) =>
            !float.IsNaN(value) && !float.IsInfinity(value);
    }

    public sealed class MlAdaptiveReward
    {
        public float IntermediateDecisionPenalty { get; set; }
        public float DeploymentCompletionBonus { get; set; }

        internal bool IsFinite() =>
            !float.IsNaN(IntermediateDecisionPenalty) &&
            !float.IsInfinity(IntermediateDecisionPenalty) &&
            !float.IsNaN(DeploymentCompletionBonus) &&
            !float.IsInfinity(DeploymentCompletionBonus);
    }

    public sealed class MlTrainingAdaptive
    {
        public int StartingUnitCount { get; set; }
        public int StartingArmyBudget { get; set; }
        public int MaxDesignPointCost { get; set; }
    }

    public sealed class MlTrainingScenarioLibrary
    {
        readonly List<MlTrainingScenario> _templates;

        MlTrainingScenarioLibrary(List<MlTrainingScenario> templates)
        {
            _templates = templates;
        }

        public static MlTrainingScenarioLibrary Load(string path)
        {
            MlTrainingScenarioLibraryWire wire;
            try
            {
                string json = File.ReadAllText(path);
                MlStrictScenarioJson.ValidateLibrary(json, path);
                wire = JsonUtility.FromJson<MlTrainingScenarioLibraryWire>(
                    json);
            }
            catch (Exception error) when (
                error is ArgumentException || error is IOException ||
                error is UnauthorizedAccessException)
            {
                throw new InvalidDataException(path + ": invalid scenario library", error);
            }

            if (wire == null) throw new InvalidDataException(path + ": library must be an object");
            if (wire.schema_version != 1)
                throw new InvalidDataException("library schema version must be 1");
            if (wire.templates == null)
                throw new InvalidDataException("library templates must be an array");

            var templates = new List<MlTrainingScenario>();
            var ids = new HashSet<string>(StringComparer.Ordinal);
            foreach (var item in wire.templates)
            {
                var scenario = MlTrainingScenarioFile.FromWire(item, path);
                if (!ids.Add(scenario.Id))
                    throw new InvalidDataException(
                        "duplicate template id '" + scenario.Id + "'");
                string expectedPrefix = scenario.Environment == MlEnvironmentContract.AdaptiveV1
                    ? "adaptive-"
                    : "tactical-";
                if (!scenario.Id.StartsWith(expectedPrefix, StringComparison.Ordinal))
                    throw new InvalidDataException(
                        "template '" + scenario.Id + "' does not match its environment '" +
                        MlEnvironmentContracts.CliValue(scenario.Environment) + "'");
                templates.Add(scenario);
            }
            return new MlTrainingScenarioLibrary(templates);
        }

        public IReadOnlyList<MlTrainingScenario> Templates => _templates;

        public IReadOnlyList<MlTrainingScenario> Filter(MlEnvironmentContract environment) =>
            _templates.Where(item => item.Environment == environment).ToArray();

        internal static string Serialize(IReadOnlyList<MlTrainingScenario> templates)
        {
            var documents = templates.Select(MlTrainingScenarioFile.Serialize);
            return "{\n  \"schema_version\": 1,\n  \"templates\": [\n" +
                string.Join(",\n", documents.Select(Indent)) +
                "\n  ]\n}\n";
        }

        static string Indent(string json) =>
            "    " + json.Trim().Replace("\n", "\n    ");
    }

    public static class MlTrainingScenarioFile
    {
        public static MlTrainingScenario Load(string path)
        {
            try
            {
                return Parse(File.ReadAllText(path), path);
            }
            catch (InvalidDataException)
            {
                throw;
            }
            catch (Exception error) when (
                error is ArgumentException || error is IOException ||
                error is UnauthorizedAccessException)
            {
                throw new InvalidDataException(path + ": invalid training scenario", error);
            }
        }

        internal static MlTrainingScenario Parse(string json, string source)
        {
            MlTrainingScenarioWire wire;
            try
            {
                MlStrictScenarioJson.ValidateScenario(json, source);
                wire = JsonUtility.FromJson<MlTrainingScenarioWire>(json);
            }
            catch (ArgumentException error)
            {
                throw new InvalidDataException(source + ": invalid JSON", error);
            }
            return FromWire(wire, source);
        }

        internal static MlTrainingScenario FromWire(
            MlTrainingScenarioWire wire, string source)
        {
            if (wire == null)
                throw new InvalidDataException(source + ": scenario must be an object");
            MlTrainingScenario scenario;
            try
            {
                scenario = MlTrainingScenarioWire.ToScenario(wire);
            }
            catch (ArgumentException error)
            {
                throw new InvalidDataException(source + ": " + error.Message, error);
            }

            IReadOnlyList<string> errors = scenario.Validate();
            if (errors.Count > 0)
                throw new InvalidDataException(
                    source + ": " + string.Join("; ", errors));
            return scenario;
        }

        internal static string Serialize(MlTrainingScenario scenario)
        {
            ThrowIfInvalid(scenario);
            return scenario.Environment == MlEnvironmentContract.AdaptiveV1
                ? JsonUtility.ToJson(MlAdaptiveScenarioWire.FromScenario(scenario), true)
                : JsonUtility.ToJson(MlTacticalScenarioWire.FromScenario(scenario), true);
        }

        internal static void ThrowIfInvalid(MlTrainingScenario scenario)
        {
            if (scenario == null) throw new InvalidDataException("scenario is required");
            IReadOnlyList<string> errors = scenario.Validate();
            if (errors.Count > 0)
                throw new InvalidDataException(string.Join("; ", errors));
        }
    }

    public static class MlTrainingScenarioStore
    {
        public static string WriteSessionScenario(
            string projectRoot, MlTrainingScenario scenario)
        {
            if (string.IsNullOrWhiteSpace(projectRoot))
                throw new ArgumentException("Project root is required.", nameof(projectRoot));
            string directory = Path.Combine(projectRoot, "Library", "HexWars", "MLLab");
            Directory.CreateDirectory(directory);
            string path = Path.Combine(directory, "scenario.json");
            WriteScenarioAtomically(path, scenario);
            return path;
        }

        public static void SaveAsTemplate(
            string path, MlTrainingScenario scenario, bool overwrite)
        {
            MlTrainingScenarioFile.ThrowIfInvalid(scenario);
            var templates = File.Exists(path)
                ? MlTrainingScenarioLibrary.Load(path).Templates
                    .Select(item => item.Clone()).ToList()
                : new List<MlTrainingScenario>();
            int existingIndex = templates.FindIndex(
                item => string.Equals(item.Id, scenario.Id, StringComparison.Ordinal));
            if (existingIndex >= 0 && !overwrite)
                throw new InvalidOperationException(
                    "Template id '" + scenario.Id + "' already exists.");
            if (existingIndex >= 0) templates[existingIndex] = scenario.Clone();
            else templates.Add(scenario.Clone());

            string directory = Path.GetDirectoryName(Path.GetFullPath(path));
            Directory.CreateDirectory(directory);
            string temp = path + ".tmp";
            File.WriteAllText(temp, MlTrainingScenarioLibrary.Serialize(templates));
            MlTrainingScenarioLibrary.Load(temp);
            ReplaceOrMove(temp, path);
        }

        static void WriteScenarioAtomically(string path, MlTrainingScenario scenario)
        {
            string temp = path + ".tmp";
            File.WriteAllText(temp, MlTrainingScenarioFile.Serialize(scenario) + "\n");
            MlTrainingScenarioFile.Load(temp);
            ReplaceOrMove(temp, path);
        }

        static void ReplaceOrMove(string temp, string target)
        {
            if (File.Exists(target)) File.Replace(temp, target, null);
            else File.Move(temp, target);
        }
    }

    static class MlStrictScenarioJson
    {
        static readonly string[] LibraryKeys = { "schema_version", "templates" };
        static readonly string[] ScenarioKeys =
        {
            "schema_version", "id", "name", "environment", "board", "rules",
            "episode", "reward",
        };
        static readonly string[] AdaptiveScenarioKeys =
            ScenarioKeys.Concat(new[] { "adaptive" }).ToArray();
        static readonly string[] BoardKeys =
        {
            "width", "height", "max_elevation", "zone_depth", "flat_chance",
            "plains_weight", "forest_weight", "rough_weight", "water_weight",
        };
        static readonly string[] RulesKeys =
        {
            "actions_per_turn", "round_cap", "starting_points", "fog_of_war",
            "biomes_enabled", "bounty_rate", "deploy_cost_multiplier",
            "generator_cost", "generator_output", "generator_health",
        };
        static readonly string[] EpisodeKeys = { "max_steps" };
        static readonly string[] TacticalRewardKeys =
        {
            "shape_scale", "step_penalty", "closing_weight",
            "draw_credit_weight", "points_weight",
        };
        static readonly string[] AdaptiveRewardKeys =
        {
            "intermediate_decision_penalty", "deployment_completion_bonus",
        };
        static readonly string[] AdaptiveKeys =
        {
            "starting_unit_count", "starting_army_budget", "max_design_point_cost",
        };

        public static void ValidateLibrary(string json, string source)
        {
            JsonNode root = Parse(json, source);
            Dictionary<string, JsonNode> library = RequireObject(root, "library");
            ExactKeys(library, LibraryKeys, "library");
            RequireSchemaVersion(library["schema_version"], "library.schema_version");
            List<JsonNode> templates = RequireArray(library["templates"], "library.templates");
            for (int i = 0; i < templates.Count; i++)
                ValidateScenarioNode(templates[i], "library.templates[" + i + "]");
        }

        public static void ValidateScenario(string json, string source)
        {
            ValidateScenarioNode(Parse(json, source), "scenario");
        }

        static void ValidateScenarioNode(JsonNode node, string path)
        {
            Dictionary<string, JsonNode> scenario = RequireObject(node, path);
            string environment = RequireString(
                Required(scenario, "environment", path), path + ".environment");
            if (string.IsNullOrWhiteSpace(environment))
                Fail(path + ".environment must be a non-empty string");
            bool adaptive = string.Equals(
                environment, "adaptive-v1", StringComparison.Ordinal);
            if (!adaptive && !string.Equals(
                    environment, "tactical-v1", StringComparison.Ordinal))
                Fail(path + ".environment must be tactical-v1 or adaptive-v1");

            ExactKeys(
                scenario, adaptive ? AdaptiveScenarioKeys : ScenarioKeys, path);
            RequireSchemaVersion(scenario["schema_version"], path + ".schema_version");
            RequireNonEmptyString(scenario["id"], path + ".id");
            RequireNonEmptyString(scenario["name"], path + ".name");

            ValidateBoard(scenario["board"], path + ".board");
            ValidateRules(scenario["rules"], path + ".rules");
            ValidateEpisode(scenario["episode"], path + ".episode");
            ValidateReward(
                scenario["reward"], path + ".reward",
                adaptive ? AdaptiveRewardKeys : TacticalRewardKeys);
            if (adaptive) ValidateAdaptive(scenario["adaptive"], path + ".adaptive");
        }

        static void ValidateBoard(JsonNode node, string path)
        {
            Dictionary<string, JsonNode> value = RequireObject(node, path);
            ExactKeys(value, BoardKeys, path);
            foreach (string key in new[]
                     {
                         "width", "height", "max_elevation", "zone_depth",
                         "plains_weight", "forest_weight", "rough_weight", "water_weight",
                     })
                RequireInteger(value[key], path + "." + key);
            RequireNumber(value["flat_chance"], path + ".flat_chance");
        }

        static void ValidateRules(JsonNode node, string path)
        {
            Dictionary<string, JsonNode> value = RequireObject(node, path);
            ExactKeys(value, RulesKeys, path);
            foreach (string key in new[]
                     {
                         "actions_per_turn", "round_cap", "starting_points",
                         "generator_cost", "generator_output", "generator_health",
                     })
                RequireInteger(value[key], path + "." + key);
            RequireBoolean(value["fog_of_war"], path + ".fog_of_war");
            RequireBoolean(value["biomes_enabled"], path + ".biomes_enabled");
            RequireNumber(value["bounty_rate"], path + ".bounty_rate");
            RequireNumber(
                value["deploy_cost_multiplier"], path + ".deploy_cost_multiplier");
        }

        static void ValidateEpisode(JsonNode node, string path)
        {
            Dictionary<string, JsonNode> value = RequireObject(node, path);
            ExactKeys(value, EpisodeKeys, path);
            RequireInteger(value["max_steps"], path + ".max_steps");
        }

        static void ValidateReward(
            JsonNode node, string path, IReadOnlyList<string> expectedKeys)
        {
            Dictionary<string, JsonNode> value = RequireObject(node, path);
            ExactKeys(value, expectedKeys, path);
            foreach (string key in expectedKeys)
                RequireNumber(value[key], path + "." + key);
        }

        static void ValidateAdaptive(JsonNode node, string path)
        {
            Dictionary<string, JsonNode> value = RequireObject(node, path);
            ExactKeys(value, AdaptiveKeys, path);
            foreach (string key in AdaptiveKeys)
                RequireInteger(value[key], path + "." + key);
        }

        static JsonNode Parse(string json, string source)
        {
            try
            {
                return new JsonParser(json).Parse();
            }
            catch (InvalidDataException error)
            {
                throw new InvalidDataException(source + ": " + error.Message, error);
            }
        }

        static JsonNode Required(
            Dictionary<string, JsonNode> value, string key, string path)
        {
            JsonNode node;
            if (!value.TryGetValue(key, out node))
                Fail("missing " + path + "." + key);
            return node;
        }

        static Dictionary<string, JsonNode> RequireObject(JsonNode node, string path)
        {
            if (node == null || node.Kind != JsonKind.Object)
                Fail(path + " must be an object");
            return node.ObjectValue;
        }

        static List<JsonNode> RequireArray(JsonNode node, string path)
        {
            if (node == null || node.Kind != JsonKind.Array)
                Fail(path + " must be an array");
            return node.ArrayValue;
        }

        static string RequireString(JsonNode node, string path)
        {
            if (node == null || node.Kind != JsonKind.String)
                Fail(path + " must be a string");
            return node.StringValue;
        }

        static void RequireNonEmptyString(JsonNode node, string path)
        {
            if (string.IsNullOrWhiteSpace(RequireString(node, path)))
                Fail(path + " must be a non-empty string");
        }

        static void RequireInteger(JsonNode node, string path)
        {
            if (node == null || node.Kind != JsonKind.Integer)
                Fail(path + " must be an integer");
        }

        static void RequireNumber(JsonNode node, string path)
        {
            if (node == null ||
                (node.Kind != JsonKind.Integer && node.Kind != JsonKind.Number))
                Fail(path + " must be a number");
        }

        static void RequireBoolean(JsonNode node, string path)
        {
            if (node == null || node.Kind != JsonKind.Boolean)
                Fail(path + " must be a boolean");
        }

        static void RequireSchemaVersion(JsonNode node, string path)
        {
            RequireInteger(node, path);
            if (!string.Equals(node.NumberText, "1", StringComparison.Ordinal))
                Fail(path + " must be 1");
        }

        static void ExactKeys(
            Dictionary<string, JsonNode> value,
            IReadOnlyList<string> expected,
            string path)
        {
            var allowed = new HashSet<string>(expected, StringComparer.Ordinal);
            string[] missing = expected.Where(key => !value.ContainsKey(key)).ToArray();
            string[] extra = value.Keys.Where(key => !allowed.Contains(key))
                .OrderBy(key => key, StringComparer.Ordinal).ToArray();
            var errors = new List<string>();
            if (missing.Length > 0)
                errors.Add("missing " + string.Join(
                    ", ", missing.Select(key => path + "." + key)));
            if (extra.Length > 0)
                errors.Add("unexpected " + string.Join(
                    ", ", extra.Select(key => path + "." + key)));
            if (errors.Count > 0) Fail(string.Join("; ", errors));
        }

        static void Fail(string message)
        {
            throw new InvalidDataException(message);
        }

        enum JsonKind
        {
            Object,
            Array,
            String,
            Integer,
            Number,
            Boolean,
            Null,
        }

        sealed class JsonNode
        {
            public JsonKind Kind;
            public Dictionary<string, JsonNode> ObjectValue;
            public List<JsonNode> ArrayValue;
            public string StringValue;
            public string NumberText;
        }

        sealed class JsonParser
        {
            const int MaxDepth = 64;
            readonly string _json;
            int _index;

            public JsonParser(string json)
            {
                _json = json ?? string.Empty;
            }

            public JsonNode Parse()
            {
                SkipWhitespace();
                JsonNode value = ParseValue(0);
                SkipWhitespace();
                if (_index != _json.Length) Error("unexpected trailing JSON content");
                return value;
            }

            JsonNode ParseValue(int depth)
            {
                if (depth > MaxDepth) Error("JSON nesting exceeds 64 levels");
                if (_index >= _json.Length) Error("unexpected end of JSON");
                char current = _json[_index];
                if (current == '{') return ParseObject(depth + 1);
                if (current == '[') return ParseArray(depth + 1);
                if (current == '"')
                    return new JsonNode
                    {
                        Kind = JsonKind.String,
                        StringValue = ParseString(),
                    };
                if (current == 't')
                {
                    ReadLiteral("true");
                    return new JsonNode { Kind = JsonKind.Boolean };
                }
                if (current == 'f')
                {
                    ReadLiteral("false");
                    return new JsonNode { Kind = JsonKind.Boolean };
                }
                if (current == 'n')
                {
                    ReadLiteral("null");
                    return new JsonNode { Kind = JsonKind.Null };
                }
                if (current == '-' || (current >= '0' && current <= '9'))
                    return ParseNumber();
                Error("unexpected character '" + current + "'");
                return null;
            }

            JsonNode ParseObject(int depth)
            {
                Expect('{');
                var fields = new Dictionary<string, JsonNode>(StringComparer.Ordinal);
                SkipWhitespace();
                if (TryRead('}'))
                    return new JsonNode { Kind = JsonKind.Object, ObjectValue = fields };
                while (true)
                {
                    SkipWhitespace();
                    if (_index >= _json.Length || _json[_index] != '"')
                        Error("object property name must be a string");
                    string key = ParseString();
                    SkipWhitespace();
                    Expect(':');
                    SkipWhitespace();
                    fields[key] = ParseValue(depth);
                    SkipWhitespace();
                    if (TryRead('}')) break;
                    Expect(',');
                }
                return new JsonNode { Kind = JsonKind.Object, ObjectValue = fields };
            }

            JsonNode ParseArray(int depth)
            {
                Expect('[');
                var values = new List<JsonNode>();
                SkipWhitespace();
                if (TryRead(']'))
                    return new JsonNode { Kind = JsonKind.Array, ArrayValue = values };
                while (true)
                {
                    SkipWhitespace();
                    values.Add(ParseValue(depth));
                    SkipWhitespace();
                    if (TryRead(']')) break;
                    Expect(',');
                }
                return new JsonNode { Kind = JsonKind.Array, ArrayValue = values };
            }

            JsonNode ParseNumber()
            {
                int start = _index;
                if (TryRead('-') && _index >= _json.Length)
                    Error("incomplete JSON number");
                if (TryRead('0'))
                {
                    if (_index < _json.Length &&
                        _json[_index] >= '0' && _json[_index] <= '9')
                        Error("JSON number cannot have a leading zero");
                }
                else
                {
                    ReadDigits(required: true);
                }

                bool fractional = false;
                if (TryRead('.'))
                {
                    fractional = true;
                    ReadDigits(required: true);
                }
                if (_index < _json.Length &&
                    (_json[_index] == 'e' || _json[_index] == 'E'))
                {
                    fractional = true;
                    _index++;
                    if (_index < _json.Length &&
                        (_json[_index] == '+' || _json[_index] == '-'))
                        _index++;
                    ReadDigits(required: true);
                }
                return new JsonNode
                {
                    Kind = fractional ? JsonKind.Number : JsonKind.Integer,
                    NumberText = _json.Substring(start, _index - start),
                };
            }

            string ParseString()
            {
                Expect('"');
                var value = new System.Text.StringBuilder();
                while (_index < _json.Length)
                {
                    char current = _json[_index++];
                    if (current == '"') return value.ToString();
                    if (current < 0x20) Error("unescaped control character in string");
                    if (current != '\\')
                    {
                        value.Append(current);
                        continue;
                    }
                    if (_index >= _json.Length) Error("incomplete string escape");
                    char escaped = _json[_index++];
                    switch (escaped)
                    {
                        case '"': value.Append('"'); break;
                        case '\\': value.Append('\\'); break;
                        case '/': value.Append('/'); break;
                        case 'b': value.Append('\b'); break;
                        case 'f': value.Append('\f'); break;
                        case 'n': value.Append('\n'); break;
                        case 'r': value.Append('\r'); break;
                        case 't': value.Append('\t'); break;
                        case 'u': value.Append(ParseUnicodeEscape()); break;
                        default: Error("invalid string escape \\" + escaped); break;
                    }
                }
                Error("unterminated string");
                return null;
            }

            char ParseUnicodeEscape()
            {
                if (_index + 4 > _json.Length) Error("incomplete unicode escape");
                int value = 0;
                for (int i = 0; i < 4; i++)
                {
                    int digit = HexValue(_json[_index++]);
                    if (digit < 0) Error("invalid unicode escape");
                    value = value * 16 + digit;
                }
                return (char)value;
            }

            void ReadLiteral(string literal)
            {
                if (_index + literal.Length > _json.Length ||
                    !string.Equals(
                        _json.Substring(_index, literal.Length),
                        literal, StringComparison.Ordinal))
                    Error("invalid JSON literal");
                _index += literal.Length;
            }

            void ReadDigits(bool required)
            {
                int start = _index;
                while (_index < _json.Length &&
                       _json[_index] >= '0' && _json[_index] <= '9')
                    _index++;
                if (required && start == _index) Error("JSON number requires a digit");
            }

            void SkipWhitespace()
            {
                while (_index < _json.Length)
                {
                    char current = _json[_index];
                    if (current != ' ' && current != '\t' &&
                        current != '\r' && current != '\n')
                        break;
                    _index++;
                }
            }

            bool TryRead(char expected)
            {
                if (_index >= _json.Length || _json[_index] != expected) return false;
                _index++;
                return true;
            }

            void Expect(char expected)
            {
                if (!TryRead(expected))
                    Error("expected '" + expected + "'");
            }

            void Error(string message)
            {
                throw new InvalidDataException(
                    message + " at character " + _index);
            }

            static int HexValue(char value)
            {
                if (value >= '0' && value <= '9') return value - '0';
                if (value >= 'a' && value <= 'f') return value - 'a' + 10;
                if (value >= 'A' && value <= 'F') return value - 'A' + 10;
                return -1;
            }
        }
    }

    [Serializable]
    sealed class MlTrainingScenarioLibraryWire
    {
        public int schema_version;
        public MlTrainingScenarioWire[] templates;
    }

    [Serializable]
    sealed class MlTrainingScenarioWire
    {
        public int schema_version;
        public string id;
        public string name;
        public string environment;
        public MlTrainingBoardWire board;
        public MlTrainingRulesWire rules;
        public MlTrainingEpisodeWire episode;
        public MlTrainingRewardWire reward;
        public MlTrainingAdaptiveWire adaptive;

        public static MlTrainingScenario ToScenario(MlTrainingScenarioWire wire)
        {
            bool adaptiveEnvironment;
            if (string.Equals(wire.environment, "adaptive-v1", StringComparison.Ordinal))
                adaptiveEnvironment = true;
            else if (string.Equals(wire.environment, "tactical-v1", StringComparison.Ordinal))
                adaptiveEnvironment = false;
            else
                throw new ArgumentException(
                    "environment must be tactical-v1 or adaptive-v1");
            return new MlTrainingScenario
            {
                SchemaVersion = wire.schema_version,
                Id = wire.id ?? string.Empty,
                Name = wire.name ?? string.Empty,
                Environment = adaptiveEnvironment
                    ? MlEnvironmentContract.AdaptiveV1
                    : MlEnvironmentContract.TacticalV1,
                Board = wire.board?.ToModel(),
                Rules = wire.rules?.ToModel(),
                Episode = wire.episode?.ToModel(),
                TacticalReward = adaptiveEnvironment ? null : wire.reward?.ToTactical(),
                AdaptiveReward = adaptiveEnvironment ? wire.reward?.ToAdaptive() : null,
                Adaptive = adaptiveEnvironment ? wire.adaptive?.ToModel() : null,
            };
        }
    }

    [Serializable]
    sealed class MlTacticalScenarioWire
    {
        public int schema_version;
        public string id;
        public string name;
        public string environment;
        public MlTrainingBoardWire board;
        public MlTrainingRulesWire rules;
        public MlTrainingEpisodeWire episode;
        public MlTacticalRewardWire reward;

        public static MlTacticalScenarioWire FromScenario(MlTrainingScenario scenario) =>
            new MlTacticalScenarioWire
            {
                schema_version = scenario.SchemaVersion,
                id = scenario.Id,
                name = scenario.Name,
                environment = "tactical-v1",
                board = MlTrainingBoardWire.FromModel(scenario.Board),
                rules = MlTrainingRulesWire.FromModel(scenario.Rules),
                episode = MlTrainingEpisodeWire.FromModel(scenario.Episode),
                reward = MlTacticalRewardWire.FromModel(scenario.TacticalReward),
            };
    }

    [Serializable]
    sealed class MlAdaptiveScenarioWire
    {
        public int schema_version;
        public string id;
        public string name;
        public string environment;
        public MlTrainingBoardWire board;
        public MlTrainingRulesWire rules;
        public MlTrainingEpisodeWire episode;
        public MlAdaptiveRewardWire reward;
        public MlTrainingAdaptiveWire adaptive;

        public static MlAdaptiveScenarioWire FromScenario(MlTrainingScenario scenario) =>
            new MlAdaptiveScenarioWire
            {
                schema_version = scenario.SchemaVersion,
                id = scenario.Id,
                name = scenario.Name,
                environment = "adaptive-v1",
                board = MlTrainingBoardWire.FromModel(scenario.Board),
                rules = MlTrainingRulesWire.FromModel(scenario.Rules),
                episode = MlTrainingEpisodeWire.FromModel(scenario.Episode),
                reward = MlAdaptiveRewardWire.FromModel(scenario.AdaptiveReward),
                adaptive = MlTrainingAdaptiveWire.FromModel(scenario.Adaptive),
            };
    }

    [Serializable]
    sealed class MlTrainingBoardWire
    {
        public int width;
        public int height;
        public int max_elevation;
        public int zone_depth;
        public double flat_chance;
        public int plains_weight;
        public int forest_weight;
        public int rough_weight;
        public int water_weight;

        public MlTrainingBoard ToModel() => new MlTrainingBoard
        {
            Width = width,
            Height = height,
            MaxElevation = max_elevation,
            ZoneDepth = zone_depth,
            FlatChance = flat_chance,
            PlainsWeight = plains_weight,
            ForestWeight = forest_weight,
            RoughWeight = rough_weight,
            WaterWeight = water_weight,
        };

        public static MlTrainingBoardWire FromModel(MlTrainingBoard model) =>
            new MlTrainingBoardWire
            {
                width = model.Width,
                height = model.Height,
                max_elevation = model.MaxElevation,
                zone_depth = model.ZoneDepth,
                flat_chance = model.FlatChance,
                plains_weight = model.PlainsWeight,
                forest_weight = model.ForestWeight,
                rough_weight = model.RoughWeight,
                water_weight = model.WaterWeight,
            };
    }

    [Serializable]
    sealed class MlTrainingRulesWire
    {
        public int actions_per_turn;
        public int round_cap;
        public int starting_points;
        public bool fog_of_war;
        public bool biomes_enabled;
        public double bounty_rate;
        public double deploy_cost_multiplier;
        public int generator_cost;
        public int generator_output;
        public int generator_health;

        public MlTrainingRules ToModel() => new MlTrainingRules
        {
            ActionsPerTurn = actions_per_turn,
            RoundCap = round_cap,
            StartingPoints = starting_points,
            FogOfWar = fog_of_war,
            BiomesEnabled = biomes_enabled,
            BountyRate = bounty_rate,
            DeployCostMultiplier = deploy_cost_multiplier,
            GeneratorCost = generator_cost,
            GeneratorOutput = generator_output,
            GeneratorHealth = generator_health,
        };

        public static MlTrainingRulesWire FromModel(MlTrainingRules model) =>
            new MlTrainingRulesWire
            {
                actions_per_turn = model.ActionsPerTurn,
                round_cap = model.RoundCap,
                starting_points = model.StartingPoints,
                fog_of_war = model.FogOfWar,
                biomes_enabled = model.BiomesEnabled,
                bounty_rate = model.BountyRate,
                deploy_cost_multiplier = model.DeployCostMultiplier,
                generator_cost = model.GeneratorCost,
                generator_output = model.GeneratorOutput,
                generator_health = model.GeneratorHealth,
            };
    }

    [Serializable]
    sealed class MlTrainingEpisodeWire
    {
        public int max_steps;

        public MlTrainingEpisode ToModel() => new MlTrainingEpisode { MaxSteps = max_steps };

        public static MlTrainingEpisodeWire FromModel(MlTrainingEpisode model) =>
            new MlTrainingEpisodeWire { max_steps = model.MaxSteps };
    }

    [Serializable]
    sealed class MlTrainingRewardWire
    {
        public float shape_scale;
        public float step_penalty;
        public float closing_weight;
        public float draw_credit_weight;
        public float points_weight;
        public float intermediate_decision_penalty;
        public float deployment_completion_bonus;

        public MlTacticalReward ToTactical() => new MlTacticalReward
        {
            ShapeScale = shape_scale,
            StepPenalty = step_penalty,
            ClosingWeight = closing_weight,
            DrawCreditWeight = draw_credit_weight,
            PointsWeight = points_weight,
        };

        public MlAdaptiveReward ToAdaptive() => new MlAdaptiveReward
        {
            IntermediateDecisionPenalty = intermediate_decision_penalty,
            DeploymentCompletionBonus = deployment_completion_bonus,
        };
    }

    [Serializable]
    sealed class MlTacticalRewardWire
    {
        public float shape_scale;
        public float step_penalty;
        public float closing_weight;
        public float draw_credit_weight;
        public float points_weight;

        public static MlTacticalRewardWire FromModel(MlTacticalReward model) =>
            new MlTacticalRewardWire
            {
                shape_scale = model.ShapeScale,
                step_penalty = model.StepPenalty,
                closing_weight = model.ClosingWeight,
                draw_credit_weight = model.DrawCreditWeight,
                points_weight = model.PointsWeight,
            };
    }

    [Serializable]
    sealed class MlAdaptiveRewardWire
    {
        public float intermediate_decision_penalty;
        public float deployment_completion_bonus;

        public static MlAdaptiveRewardWire FromModel(MlAdaptiveReward model) =>
            new MlAdaptiveRewardWire
            {
                intermediate_decision_penalty = model.IntermediateDecisionPenalty,
                deployment_completion_bonus = model.DeploymentCompletionBonus,
            };
    }

    [Serializable]
    sealed class MlTrainingAdaptiveWire
    {
        public int starting_unit_count;
        public int starting_army_budget;
        public int max_design_point_cost;

        public MlTrainingAdaptive ToModel() => new MlTrainingAdaptive
        {
            StartingUnitCount = starting_unit_count,
            StartingArmyBudget = starting_army_budget,
            MaxDesignPointCost = max_design_point_cost,
        };

        public static MlTrainingAdaptiveWire FromModel(MlTrainingAdaptive model) =>
            new MlTrainingAdaptiveWire
            {
                starting_unit_count = model.StartingUnitCount,
                starting_army_budget = model.StartingArmyBudget,
                max_design_point_cost = model.MaxDesignPointCost,
            };
    }
}
