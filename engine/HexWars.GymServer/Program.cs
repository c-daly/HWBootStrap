using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using System.Text.Json;
using HexWars.Engine;
using HexWars.Engine.Rl;
using HexWars.GymServer;

// Headless RL bridge: wraps a TacticalEnv and speaks one JSON object per line over stdin/stdout, so a
// Python gymnasium.Env can drive it as a subprocess. Cross-platform (.NET 8) ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â built for WSL2/Linux.
//
//   stdin  {"cmd":"spaces"}                 -> {"obs_len":N,"n_actions":M}
//          {"cmd":"reset","seed":123}       -> {"obs":[...],"mask":[...]}
//          {"cmd":"step","action":5}        -> {"obs":[...],"reward":r,"terminated":b,"truncated":b,"mask":[...]}
//          {"cmd":"close"}                  -> (exits)
//
// Args: --opponent greedy|random   --seat 0|1   --environment tactical-v1|adaptive-v1
string opponent = "greedy";
int seat = 0;
string environment = "tactical-v1";
string? scenarioFile = null;
for (int i = 0; i < args.Length; i++)
{
    if (args[i] == "--opponent" && i + 1 < args.Length) opponent = args[++i];
    else if (args[i] == "--seat" && i + 1 < args.Length) seat = int.Parse(args[++i]);
    else if (args[i] == "--environment") environment = i + 1 < args.Length ? args[++i] : "";
    else if (args[i] == "--scenario-file")
    {
        if (scenarioFile != null || i + 1 >= args.Length || args[i + 1].StartsWith("--", StringComparison.Ordinal))
            throw new InvalidDataException("--scenario-file requires exactly one path");
        scenarioFile = args[++i];
    }
}
if (environment != "tactical-v1" && environment != "adaptive-v1" &&
    environment != "tactical-v2" && environment != MlContract.TacticalV3Version)
{
    Console.Error.WriteLine($"unsupported environment '{environment}'");
    Environment.ExitCode = 2;
    return;
}
if (environment == MlContract.TacticalV3Version && scenarioFile == null)
    throw new InvalidDataException("tactical-v3 requires --scenario-file");

TrainingScenario scenario = scenarioFile == null
    ? TrainingScenario.CreateStandard(environment)
    : ScenarioJson.Load(scenarioFile);
if (!string.Equals(scenario.Environment, environment, StringComparison.Ordinal))
    throw new InvalidDataException("scenario environment does not match --environment");

EnvConfig? tacticalConfig = environment == "tactical-v1" ? scenario.BuildTactical() : null;
AdaptiveEnvConfig? adaptiveConfig = environment == "adaptive-v1" ? scenario.BuildAdaptive() : null;
TacticalV2Config? tacticalV2Config = environment == "tactical-v2" ? scenario.BuildTacticalV2() : null;
TacticalV3Config? tacticalV3Config = environment == MlContract.TacticalV3Version
    ? scenario.BuildTacticalV3()
    : null;

Func<int, IAgent> opponentFactory = opponent == "random"
    ? (s => new RandomAgent(s))
    : (s => new GreedyAgent(s));

PlayerId learningSeat = seat == 1 ? PlayerId.Player1 : PlayerId.Player0;
TacticalEnv? env = environment == "tactical-v1" ? new TacticalEnv(opponentFactory, learningSeat, tacticalConfig) : null;
AdaptiveTacticalEnv? adaptiveEnv = environment == "adaptive-v1"
    ? new AdaptiveTacticalEnv(opponentFactory,
        opponent == "random" ? s => new RandomDeploymentPolicy(s) : s => new CombinedArmsDeploymentPolicy(s),
        learningSeat, adaptiveConfig)
    : null;
TacticalV2Env? tacticalV2Env = environment == "tactical-v2"
    ? new TacticalV2Env(opponentFactory, learningSeat, tacticalV2Config!)
    : null;
TacticalV3Env? tacticalV3Env = tacticalV3Config == null
    ? null
    : new TacticalV3Env(opponentFactory, learningSeat, tacticalV3Config);
var tacticalV2Trace = new BufferedDuelTransitionSink();
var tacticalV2Demonstrations = new BufferedTacticalV2DemonstrationSink();
var tacticalV2Dagger = new BufferedTacticalV2DaggerSink();
DuelEnv? duel = null; // created on first duel_* command (two external controllers)
AdaptiveDuelEnv? adaptiveDuel = null;
TacticalV2DuelEnv? tacticalV2Duel = null;var tacticalV2Preflight = new BufferedOraclePreflightBenchmarkSink();
TacticalV3DuelEnv? tacticalV3Duel = null;
GreedyAgent? tacticalV3GreedyTeacher = null;
bool tacticalV3DuelHasReset = false;
OracleEvidenceSession? evidenceSession = null;
bool evidenceGameOpen = false;
const string EvidenceOracleCodeSha256 = "5f03a7c8d0fda16497a9e6a2f1ad1ba4fcb920957b7a4b5fbc2545e0ae893061";
string evidenceScenarioSha256 = scenarioFile == null ? OracleEvidenceSession.Sha256(Encoding.UTF8.GetBytes(scenario.Id + "|" + scenario.SchemaVersion)) : OracleEvidenceSession.Sha256(File.ReadAllBytes(scenarioFile));
OracleEvidenceRuntimeIdentity EvidenceRuntime()
{
    MlContract contract = MlContract.CreateTacticalV2(tacticalV2Config!, MlEnvironmentKind.Duel);
    return new OracleEvidenceRuntimeIdentity(environment, evidenceScenarioSha256, contract.ContractHash, contract.EncodingHash, EvidenceOracleCodeSha256);
}
void InstallEvidenceObserver(OracleEvidenceScheduleItem item)
{
    tacticalV2Duel ??= new TacticalV2DuelEnv(tacticalV2Config!, tacticalV2Trace, tacticalV2Demonstrations);
    tacticalV2Dagger.Enabled = true;
    tacticalV2Duel.DecisionObserver = new SelectiveDaggerObserver(new OraclePreflightActionOracle(
        new BoundedSearchActionOracle(tacticalV2Config!.Game, item.Oracle.ExpansionBudget, item.Oracle.Depth, BoundedSearchAgent.HeuristicIdentity), tacticalV2Preflight, Stopwatch.GetTimestamp, Stopwatch.Frequency), tacticalV2Dagger);
}

var output = Console.Out;

void Send(object payload)
{
    output.WriteLine(JsonSerializer.Serialize(payload));
    output.Flush();
}

void RequireExactDaggerConfigureFields(JsonElement element)
{
    string[] expected =
    {
        "cmd", "enabled", "depth", "expansion_budget", "use_heuristic",
    };
    string[] actual = element.EnumerateObject()
        .Select(property => property.Name)
        .OrderBy(name => name, StringComparer.Ordinal)
        .ToArray();
    if (!actual.SequenceEqual(expected.OrderBy(name => name, StringComparer.Ordinal)))
        throw new InvalidDataException(
            "duel DAgger configure must contain exactly cmd, enabled, depth, " +
            "expansion_budget, and use_heuristic");
}

bool RequireDaggerBoolean(JsonElement element, string field, string label)
{
    JsonElement value = element.GetProperty(field);
    if (value.ValueKind != JsonValueKind.True && value.ValueKind != JsonValueKind.False)
        throw new InvalidDataException($"duel DAgger {label} must be boolean");
    return value.GetBoolean();
}

int RequirePositiveDaggerInteger(JsonElement element, string field, string label)
{
    JsonElement value = element.GetProperty(field);
    if (value.ValueKind != JsonValueKind.Number || !value.TryGetInt32(out int parsed)
        || parsed < 1)
        throw new InvalidDataException(
            $"duel DAgger {label} must be a positive integer");
    return parsed;
}

void RequireTacticalV3FieldValue(JsonProperty property, string? command)
{
    string label = command == null ? "tactical-v3" : $"tactical-v3 {command}";
    switch (property.Name)
    {
        case "cmd":
        case "p0":
        case "p1":
        case "start_profile":
        case "path":
        case "heuristic_identity":
        case "teacher_identity":
            if (property.Value.ValueKind != JsonValueKind.String ||
                string.IsNullOrEmpty(property.Value.GetString()))
                throw new InvalidDataException(
                    $"{label} field '{property.Name}' must be a non-empty string");
            break;
        case "seed":
        case "learner":
        case "reference_seat":
        case "candidate_id":
        case "search_depth":
        case "expansion_budget":
            if (property.Value.ValueKind != JsonValueKind.Number ||
                !property.Value.TryGetInt32(out _))
                throw new InvalidDataException(
                    $"{label} field '{property.Name}' must be an Int32 number");
            break;
        case "decision_id":
            if (property.Value.ValueKind != JsonValueKind.Number ||
                !property.Value.TryGetInt64(out _))
                throw new InvalidDataException(
                    $"{label} field 'decision_id' must be an Int64 number");
            break;
    }
}

string RequireTacticalV3Command(JsonElement element)
{
    if (element.ValueKind != JsonValueKind.Object)
        throw new InvalidDataException("tactical-v3 request must be an object");

    JsonProperty[] properties = element.EnumerateObject().ToArray();
    JsonProperty[] commandProperties = properties
        .Where(property => property.Name == "cmd").ToArray();
    if (commandProperties.Length == 0)
        throw new InvalidDataException("tactical-v3 request is missing cmd field");
    if (commandProperties.Length != 1)
        throw new InvalidDataException("tactical-v3 request has duplicate cmd field");
    RequireTacticalV3FieldValue(commandProperties[0], command: null);
    string command = commandProperties[0].Value.GetString()!;
    string[]? allowed = command switch
    {
        "spaces" => new[] { "cmd" },
        "reset" => new[] { "cmd", "seed" },
        "step" => new[] { "cmd", "decision_id", "candidate_id" },
        "duel_spaces" => new[] { "cmd" },
        "duel_reset" => new[]
        {
            "cmd", "seed", "p0", "p1", "learner", "start_profile", "reference_seat",
        },
        "duel_step" => new[] { "cmd", "decision_id", "candidate_id" },
        "duel_oracle_step" => new[]
        {
            "cmd", "decision_id", "search_depth", "expansion_budget",
            "heuristic_identity",
        },
        "duel_oracle_query" => new[]
        {
            "cmd", "decision_id", "search_depth", "expansion_budget",
            "heuristic_identity",
        },
        "duel_greedy_step" => new[]
        {
            "cmd", "decision_id", "teacher_identity",
        },
        "duel_dagger_inspect" => new[]
        {
            "cmd", "decision_id", "candidate_id",
        },
        "duel_status" => new[] { "cmd" },
        "duel_save" => new[] { "cmd", "path" },
        "close" => new[] { "cmd" },
        _ => null,
    };
    if (allowed == null)
    {
        string[] tacticalV2Only =
        {
            "duel_trace_enable", "duel_trace_drain",
            "duel_demo_enable", "duel_demo_drain",
            "duel_dagger_configure", "duel_dagger_drain",
            "duel_evidence_begin", "duel_evidence_game_close", "duel_evidence_end",
        };
        if (tacticalV2Only.Contains(command, StringComparer.Ordinal)) return command;
        throw new InvalidDataException($"tactical-v3 unknown command '{command}'");
    }

    string? duplicate = properties.GroupBy(property => property.Name, StringComparer.Ordinal)
        .Where(group => group.Count() > 1)
        .Select(group => group.Key)
        .FirstOrDefault();
    if (duplicate != null)
        throw new InvalidDataException(
            $"tactical-v3 {command} has duplicate field '{duplicate}'");
    if (properties.Any(property =>
            !allowed.Contains(property.Name, StringComparer.Ordinal)))
        throw new InvalidDataException(
            $"tactical-v3 {command} has unknown or missing fields");
    foreach (JsonProperty property in properties)
        if (property.Name != "cmd")
            RequireTacticalV3FieldValue(property, command);

    string[] required = command switch
    {
        "step" => new[] { "cmd", "decision_id", "candidate_id" },
        "duel_step" => new[] { "cmd", "decision_id", "candidate_id" },
        "duel_oracle_step" => new[]
        {
            "cmd", "decision_id", "search_depth", "expansion_budget",
            "heuristic_identity",
        },
        "duel_oracle_query" => new[]
        {
            "cmd", "decision_id", "search_depth", "expansion_budget",
            "heuristic_identity",
        },
        "duel_greedy_step" => new[]
        {
            "cmd", "decision_id", "teacher_identity",
        },
        "duel_dagger_inspect" => new[]
        {
            "cmd", "decision_id", "candidate_id",
        },
        _ => new[] { "cmd" },
    };
    if (required.Any(field =>
            !properties.Any(property => property.Name == field)))
        throw new InvalidDataException(
            $"tactical-v3 {command} has unknown or missing fields");
    if (command == "duel_oracle_step" || command == "duel_oracle_query")
    {
        if (element.GetProperty("search_depth").GetInt32() != 4)
            throw new InvalidDataException(
                $"tactical-v3 {command} search_depth must be 4");
        if (element.GetProperty("expansion_budget").GetInt32() != 512)
            throw new InvalidDataException(
                $"tactical-v3 {command} expansion_budget must be 512");
        if (element.GetProperty("heuristic_identity").GetString() !=
            BoundedSearchAgent.HeuristicIdentity)
            throw new InvalidDataException(
                $"tactical-v3 {command} heuristic_identity is unsupported");
    }
    if (command == "duel_greedy_step" &&
        element.GetProperty("teacher_identity").GetString() != "greedy-one-ply-v1")
        throw new InvalidDataException(
            "tactical-v3 duel_greedy_step teacher_identity is unsupported");
    return command;
}


TacticalV3View SelectTacticalV3(
    JsonElement element,
    string command,
    Func<long, int, TacticalV3View> select)
{
    string[] expected = { "cmd", "decision_id", "candidate_id" };
    string[] actual = element.EnumerateObject()
        .Select(property => property.Name)
        .OrderBy(name => name, StringComparer.Ordinal)
        .ToArray();
    if (!actual.SequenceEqual(expected.OrderBy(name => name, StringComparer.Ordinal)))
        throw new InvalidDataException(
            $"tactical-v3 {command} has unknown or missing fields");

    long decisionId = element.GetProperty("decision_id").GetInt64();
    int candidateId = element.GetProperty("candidate_id").GetInt32();
    try
    {
        return select(decisionId, candidateId);
    }
    catch (ArgumentOutOfRangeException exception)
        when (exception.ParamName == "candidateId")
    {
        throw new InvalidDataException(
            "tactical-v3 candidate id is out of range", exception);
    }
}

IAgent? MakeController(string? spec, int agentSeed)
{
    if (spec == "greedy") return new GreedyAgent(agentSeed);
    if (spec == "random") return new RandomAgent(agentSeed);
    if (spec == "bounded-search")
        return new BoundedSearchAgent(
            BoundedSearchAgent.DefaultExpansionBudget,
            BoundedSearchAgent.DefaultDepth,
            useHeuristic: true);
    return null; // "external" / unset -> caller supplies this seat's actions
}
void RequireTacticalV3ControllerSpec(string? spec, string field)
{
    if (spec == null || spec == "external" || spec == "greedy" ||
        spec == "random" || spec == "bounded-search")
        return;
    throw new InvalidDataException(
        $"tactical-v3 duel_reset {field} controller '{spec}' is unsupported");
}


IDeploymentPolicy? MakeDeployment(string? spec, int deploymentSeed)
{
    if (spec == "greedy") return new CombinedArmsDeploymentPolicy(deploymentSeed);
    if (spec == "random") return new RandomDeploymentPolicy(deploymentSeed);
    return null;
}

// Handshake: obs/action sizes + the spatial obs shape (so Python reshapes the board part to (C,H,W))
// + the env config (recorded into each run's params for reproducibility).
object Spaces(int obsLen, int nActions, int channels, int boardH, int boardW, EnvConfig c, MlEnvironmentKind environmentKind)
{
    var contract = MlContract.Create(c, environmentKind);
    return new
    {
        scenario_id = scenario.Id,
        scenario_schema_version = scenario.SchemaVersion,
        contract_version = contract.Version,
        contract_hash = contract.ContractHash,
        encoding_hash = contract.EncodingHash,
        environment_kind = contract.EnvironmentKind,
        obs_len = obsLen,
        n_actions = nActions,
        channels,
        board_h = boardH,
        board_w = boardW,
        globals = TacticalCoding.Globals,
        board = contract.Board,
        roster = c.Roster.Count,
        contract_roster = contract.Roster,
        reward = contract.Reward,
        biomes = c.Game.BiomesEnabled,
        round_cap = c.Game.RoundCap,
        max_steps = contract.Board["max_steps"],
        shape_scale = c.ShapeScale,
        step_penalty = c.StepPenalty,
        closing_weight = c.ClosingWeight,
        draw_credit_weight = c.DrawCreditWeight,
        points_weight = c.PointsWeight,
    };
}

object AdaptiveSpaces(AdaptiveLayout layout, AdaptiveEnvConfig config, MlContract contract)
{
    return new
    {
        scenario_id = scenario.Id,
        scenario_schema_version = scenario.SchemaVersion,
        contract_version = contract.Version,
        contract_hash = contract.ContractHash,
        encoding_hash = contract.EncodingHash,
        environment_kind = contract.EnvironmentKind,
        obs_len = layout.ObservationLength,
        n_actions = layout.ActionCount,
        channels = layout.ObservationChannels,
        board_h = layout.BoardGen.Height,
        board_w = layout.BoardGen.Width,
        globals = layout.ObservationGlobals,
        board = contract.Board,
        roster = config.Templates.Count,
        contract_roster = contract.Roster,
        reward = contract.Reward,
        biomes = config.Game.BiomesEnabled,
        round_cap = config.Game.RoundCap,
        max_steps = contract.Board["max_steps"],
        adaptive = contract.Semantics,
        action_regions = contract.Semantics["action_regions"],
        observation_channels = contract.Semantics["observation_channels"],
        phases = contract.Semantics["phases"],
    };
}

// Handshake for tactical-v2: the generic contract fields (mirroring Spaces/AdaptiveSpaces) plus the
// full v2 semantics document (catalog, slot/template split, action regions, observation channels) so
// Python never has to hardcode a tactical-v2 offset that could drift from the C# layout.
object TacticalV2Spaces(TacticalV2Layout layout, TacticalV2Config config, MlEnvironmentKind environmentKind)
{
    var contract = MlContract.CreateTacticalV2(config, environmentKind);
    return new
    {
        scenario_id = scenario.Id,
        scenario_schema_version = scenario.SchemaVersion,
        contract_version = contract.Version,
        contract_hash = contract.ContractHash,
        encoding_hash = contract.EncodingHash,
        environment_kind = contract.EnvironmentKind,
        obs_len = layout.ObservationLength,
        n_actions = layout.ActionCount,
        channels = layout.ObservationChannels,
        board_h = layout.BoardGen.Height,
        board_w = layout.BoardGen.Width,
        globals = layout.ObservationGlobals,
        board = contract.Board,
        roster = config.Templates.Count,
        contract_roster = contract.Roster,
        reward = contract.Reward,
        biomes = config.Game.BiomesEnabled,
        round_cap = config.Game.RoundCap,
        max_steps = contract.Board["max_steps"],
        tactical_v2 = contract.Semantics,
        action_regions = contract.Semantics["action_regions"],
        observation_channels = contract.Semantics["observation_channels"],
    };
}

object Diagnostics(AdaptiveDiagnostics value) => new
{
    design_count = value.DesignCount,
    distinct_custom_templates_deployed = value.DistinctCustomTemplatesDeployed,
    pregame_decisions = value.PregameDecisions,
    invalid_sequences = value.InvalidSequences,
    deployment_completed = value.DeploymentCompleted,
};

AdaptiveDuelEnv.View ContinueHeadlessReveal(AdaptiveDuelEnv.View view)
{
    if (adaptiveDuel == null || !adaptiveDuel.AwaitingPostRevealAdvance) return view;
    var continued = adaptiveDuel.ContinueAfterReveal();
    return new AdaptiveDuelEnv.View(
        continued.Observation, continued.ActionMask, continued.Seat,
        view.Reward + continued.Reward, continued.Winner,
        continued.Terminated, continued.Truncated,
        continued.DeploymentComplete, continued.Diagnostics);
}

string? line;
while ((line = Console.ReadLine()) != null)
{
    if (string.IsNullOrWhiteSpace(line)) continue;

    using var doc = JsonDocument.Parse(line);
    var root = doc.RootElement;
    string cmd = environment == MlContract.TacticalV3Version
        ? RequireTacticalV3Command(root) : root.GetProperty("cmd").GetString() ?? "";

    switch (cmd)
    {
        case "spaces":
            if (env != null)
                Send(Spaces(env.ObservationLength, env.ActionCount, env.ObsChannels, env.BoardH, env.BoardW, env.Config, MlEnvironmentKind.Tactical));
            else if (adaptiveEnv != null)
                Send(AdaptiveSpaces(adaptiveEnv.Layout, adaptiveEnv.Config, adaptiveEnv.Contract));
            else if (tacticalV3Env != null)
                Send(TacticalV3Wire.Spaces(scenario, TacticalV3Contract.Create(tacticalV3Config!, MlEnvironmentKind.Tactical)));
            else
                Send(TacticalV2Spaces(tacticalV2Env!.Layout, tacticalV2Env.Config, MlEnvironmentKind.Tactical));
            break;

        case "reset":
        {
            int seed = root.TryGetProperty("seed", out var s) ? s.GetInt32() : 0;
            if (env != null)
            {
                var obs = env.Reset(seed);
                Send(new { obs, mask = env.LegalActionMask() });
            }
            else if (adaptiveEnv != null)
            {
                var obs = adaptiveEnv.Reset(seed);
                Send(new
                {
                    obs,
                    mask = adaptiveEnv.LegalActionMask(),
                    deployment_complete = adaptiveEnv.DeploymentComplete,
                    diagnostics = Diagnostics(adaptiveEnv.Diagnostics),
                });
            }
            else if (tacticalV3Env != null)
                Send(TacticalV3Wire.View(tacticalV3Env.Reset(seed), tacticalV3Config!.Capacity));
            else
            {
                var obs = tacticalV2Env!.Reset(seed);
                Send(new { obs, mask = tacticalV2Env.LegalActionMask() });
            }
            break;
        }

        case "step":
        {
            int action = tacticalV3Env == null ? root.GetProperty("action").GetInt32() : 0;
            if (tacticalV3Env != null)
            {
                TacticalV3View view = SelectTacticalV3(root, "step", tacticalV3Env.Step);
                Send(TacticalV3Wire.View(view, tacticalV3Config!.Capacity));
            }
            else if (env != null)
            {
                var r = env.Step(action);
                Send(new { obs = r.Observation, reward = r.Reward, terminated = r.Terminated, truncated = r.Truncated, mask = r.ActionMask });
            }
            else if (adaptiveEnv != null)
            {
                var r = adaptiveEnv.Step(action);
                Send(new
                {
                    obs = r.Observation,
                    reward = r.Reward,
                    terminated = r.Terminated,
                    truncated = r.Truncated,
                    mask = r.ActionMask,
                    deployment_complete = adaptiveEnv.DeploymentComplete,
                    diagnostics = Diagnostics(adaptiveEnv.Diagnostics),
                });
            }
            else
            {
                var r = tacticalV2Env!.Step(action);
                Send(new { obs = r.Observation, reward = r.Reward, terminated = r.Terminated, truncated = r.Truncated, mask = r.ActionMask });
            }
            break;
        }

        case "duel_spaces":
            if (environment == "tactical-v1")
            {
                duel ??= new DuelEnv(tacticalConfig);
                Send(Spaces(duel.ObservationLength, duel.ActionCount, duel.ObsChannels, duel.BoardH, duel.BoardW, duel.Config, MlEnvironmentKind.Duel));
            }
            else if (environment == "adaptive-v1")
            {
                adaptiveDuel ??= new AdaptiveDuelEnv(adaptiveConfig);
                Send(AdaptiveSpaces(adaptiveDuel.Layout, adaptiveDuel.Config, adaptiveDuel.Contract));
            }
            else if (environment == MlContract.TacticalV3Version)
            {
                tacticalV3Duel ??= new TacticalV3DuelEnv(tacticalV3Config!);
                Send(TacticalV3Wire.Spaces(scenario,
                    TacticalV3Contract.Create(tacticalV3Config!, MlEnvironmentKind.Duel)));
            }
            else
            {
                tacticalV2Duel ??= new TacticalV2DuelEnv(
                    tacticalV2Config!, tacticalV2Trace, tacticalV2Demonstrations);
                Send(TacticalV2Spaces(tacticalV2Duel.Layout, tacticalV2Duel.Config, MlEnvironmentKind.Duel));
            }
            break;

        case "duel_reset":
        {
            int seed = root.TryGetProperty("seed", out var s) ? s.GetInt32() : 0;
            string? p0 = root.TryGetProperty("p0", out var a) ? a.GetString() : null; // "external"(default)/greedy/random
            string? p1 = root.TryGetProperty("p1", out var b) ? b.GetString() : null;
            int learner = root.TryGetProperty("learner", out var lr) ? lr.GetInt32() : 0; // reward perspective
            bool hasStartProfile = root.TryGetProperty("start_profile", out var sp);
            string? startProfile = hasStartProfile
                ? sp.GetString()
                : null;
            bool hasReferenceSeat = root.TryGetProperty("reference_seat", out var rs);
            int referenceSeat = hasReferenceSeat ? rs.GetInt32() : 0;
            if (learner is < 0 or > 1)
                throw new InvalidDataException("duel_reset learner must be 0 or 1");
            if (hasReferenceSeat && referenceSeat is < 0 or > 1)
                throw new InvalidDataException("duel_reset reference_seat must be 0 or 1");
            if (environment != "tactical-v2" &&
                environment != MlContract.TacticalV3Version &&
                (startProfile != null || hasReferenceSeat))
                throw new InvalidDataException("duel_reset start_profile/reference_seat are supported only for tactical-v2/tactical-v3");
            if ((environment == MlContract.TacticalV3Version
                    ? hasStartProfile : startProfile != null) && !hasReferenceSeat)
                throw new InvalidDataException("duel_reset reference_seat is required when start_profile is supplied");
            if (environment == MlContract.TacticalV3Version &&
                !hasStartProfile && hasReferenceSeat)
                throw new InvalidDataException("tactical-v3 duel_reset reference_seat requires start_profile");
            if (environment == MlContract.TacticalV3Version)
            {
                RequireTacticalV3ControllerSpec(p0, "p0");
                RequireTacticalV3ControllerSpec(p1, "p1");
                if (startProfile != null && !tacticalV3Config!.Match.StartProfiles.Any(
                        profile => profile.Id == startProfile))
                {
                    Send(new { error =
                        $"tactical-v3 duel_reset start_profile '{startProfile}' is not declared" });
                    break;
                }
            }
            if (evidenceSession != null && !evidenceSession.Ended)
            {
                if (evidenceGameOpen)
                    throw new InvalidDataException("evidence game is already open; close it before reset");
                OracleEvidenceScheduleItem expectedEvidence = evidenceSession.Expected ?? throw new InvalidDataException("evidence schedule is complete");
                OracleEvidenceScheduledDuel scheduled = expectedEvidence.Duel;
                if (p0 != "external" || p1 != "external" || seed != scheduled.EpisodeSeed || learner != scheduled.LearnerSeat || startProfile != scheduled.Profile || !hasReferenceSeat || referenceSeat != scheduled.ReferenceSeat)
                    throw new InvalidDataException("duel_reset does not match the expected evidence schedule");
                tacticalV2Preflight.Drain();
                tacticalV2Dagger.Drain();
            }            if (evidenceSession != null && !evidenceSession.Ended) evidenceGameOpen = true;
            if (environment == "tactical-v1")
            {
                duel ??= new DuelEnv(tacticalConfig);
                var v = duel.Reset(seed, MakeController(p0, seed * 2 + 1), MakeController(p1, seed * 2 + 2),
                                   learner == 1 ? PlayerId.Player1 : PlayerId.Player0);
                Send(new { obs = v.Observation, mask = v.ActionMask, seat = v.Seat, reward = v.Reward, winner = v.Winner, terminated = v.Terminated, truncated = v.Truncated });
            }
            else if (environment == "adaptive-v1")
            {
                adaptiveDuel ??= new AdaptiveDuelEnv(adaptiveConfig);
                var v = adaptiveDuel.Reset(seed,
                    MakeController(p0, seed * 2 + 1), MakeController(p1, seed * 2 + 2),
                    MakeDeployment(p0, seed * 2 + 1), MakeDeployment(p1, seed * 2 + 2),
                    learner == 1 ? PlayerId.Player1 : PlayerId.Player0);
                v = ContinueHeadlessReveal(v);
                Send(new
                {
                    obs = v.Observation, mask = v.ActionMask, seat = v.Seat, reward = v.Reward,
                    winner = v.Winner, terminated = v.Terminated, truncated = v.Truncated,
                    deployment_complete = v.DeploymentComplete, diagnostics = Diagnostics(v.Diagnostics),
                });
            }
            else if (environment == MlContract.TacticalV3Version)
            {
                tacticalV3Duel ??= new TacticalV3DuelEnv(tacticalV3Config!);
                IAgent? controller0 = MakeController(p0, seed * 2 + 1);
                IAgent? controller1 = MakeController(p1, seed * 2 + 2);
                PlayerId learnerSeat = learner == 1 ? PlayerId.Player1 : PlayerId.Player0;
                TacticalV3View view = startProfile == null
                    ? tacticalV3Duel.Reset(seed, controller0, controller1, learnerSeat)
                    : tacticalV3Duel.Reset(
                        seed,
                        controller0,
                        controller1,
                        startProfile,
                        referenceSeat == 1 ? PlayerId.Player1 : PlayerId.Player0,
                        learnerSeat);
                tacticalV3GreedyTeacher = new GreedyAgent(
                    seed * 2 + (learner == 1 ? 2 : 1));
                tacticalV3DuelHasReset = true;
                Send(TacticalV3Wire.View(view, tacticalV3Config!.Capacity));
            }
            else
            {
                tacticalV2Duel ??= new TacticalV2DuelEnv(
                    tacticalV2Config!, tacticalV2Trace, tacticalV2Demonstrations);
                IAgent? controller0 = MakeController(p0, seed * 2 + 1);
                IAgent? controller1 = MakeController(p1, seed * 2 + 2);
                PlayerId learnerSeat = learner == 1 ? PlayerId.Player1 : PlayerId.Player0;
                var v = startProfile == null
                    ? tacticalV2Duel.Reset(seed, controller0, controller1, learnerSeat)
                    : tacticalV2Duel.Reset(
                        seed,
                        controller0,
                        controller1,
                        startProfile,
                        referenceSeat == 1 ? PlayerId.Player1 : PlayerId.Player0,
                        learnerSeat);
                Send(new
                {
                    obs = v.Observation, mask = v.ActionMask, seat = v.Seat, reward = v.Reward,
                    winner = v.Winner, terminated = v.Terminated, truncated = v.Truncated,
                    start_profile = v.StartProfileId,
                    reference_seat = (int)v.ReferenceSeat,
                });
            }
            break;
        }

        case "duel_step":
        {
            int action = environment == MlContract.TacticalV3Version
                ? 0
                : root.GetProperty("action").GetInt32();
            if (environment == MlContract.TacticalV3Version)
            {
                tacticalV3Duel ??= new TacticalV3DuelEnv(tacticalV3Config!);
                Send(TacticalV3Wire.View(
                    SelectTacticalV3(root, "duel_step", tacticalV3Duel.Step),
                    tacticalV3Config!.Capacity));
            }
            else if (environment == "tactical-v1")
            {
                duel ??= new DuelEnv(tacticalConfig);
                var v = duel.Step(action);
                Send(new { obs = v.Observation, mask = v.ActionMask, seat = v.Seat, reward = v.Reward, winner = v.Winner, terminated = v.Terminated, truncated = v.Truncated });
            }
            else if (environment == "adaptive-v1")
            {
                adaptiveDuel ??= new AdaptiveDuelEnv(adaptiveConfig);
                var v = ContinueHeadlessReveal(adaptiveDuel.Step(action));
                Send(new
                {
                    obs = v.Observation, mask = v.ActionMask, seat = v.Seat, reward = v.Reward,
                    winner = v.Winner, terminated = v.Terminated, truncated = v.Truncated,
                    deployment_complete = v.DeploymentComplete, diagnostics = Diagnostics(v.Diagnostics),
                });
            }
            else
            {
                tacticalV2Duel ??= new TacticalV2DuelEnv(
                    tacticalV2Config!, tacticalV2Trace, tacticalV2Demonstrations);
                var v = tacticalV2Duel.Step(action);
                Send(new
                {
                    obs = v.Observation, mask = v.ActionMask, seat = v.Seat, reward = v.Reward,
                    winner = v.Winner, terminated = v.Terminated, truncated = v.Truncated,
                    start_profile = v.StartProfileId,
                    reference_seat = (int)v.ReferenceSeat,
                });
            }
            break;
        }

        case "duel_oracle_step":
        {
            if (!tacticalV3DuelHasReset)
                throw new InvalidDataException(
                    "tactical-v3 duel_oracle_step requires a successful duel_reset");
            long decisionId = root.GetProperty("decision_id").GetInt64();
            TacticalV3TeacherSelection selection = tacticalV3Duel!.SelectTeacherCandidate(
                new BoundedSearchAgent(512, 4, useHeuristic: true));
            if (selection.DecisionId != decisionId)
            {
                Send(new { error = "tactical-v3 decision id is stale" });
                break;
            }
            TacticalV3View next = tacticalV3Duel.Step(
                selection.DecisionId, selection.CandidateId);
            Send(TacticalV3Wire.OracleStep(
                selection, next, tacticalV3Config!.Capacity));
            break;
        }

        case "duel_greedy_step":
        {
            if (!tacticalV3DuelHasReset || tacticalV3GreedyTeacher == null)
                throw new InvalidDataException(
                    "tactical-v3 duel_greedy_step requires a successful duel_reset");
            long decisionId = root.GetProperty("decision_id").GetInt64();
            TacticalV3TeacherSelection selection =
                tacticalV3Duel!.SelectGreedyTeacherCandidate(tacticalV3GreedyTeacher);
            if (selection.DecisionId != decisionId)
            {
                Send(new { error = "tactical-v3 decision id is stale" });
                break;
            }
            TacticalV3View next = tacticalV3Duel.Step(
                selection.DecisionId, selection.CandidateId);
            Send(TacticalV3Wire.OracleStep(
                selection, next, tacticalV3Config!.Capacity));
            break;
        }

        case "duel_oracle_query":
        {
            if (!tacticalV3DuelHasReset)
                throw new InvalidDataException(
                    "tactical-v3 duel_oracle_query requires a successful duel_reset");
            long decisionId = root.GetProperty("decision_id").GetInt64();
            TacticalV3TeacherSelection selection = tacticalV3Duel!.SelectTeacherCandidate(
                new BoundedSearchAgent(512, 4, useHeuristic: true));
            if (selection.DecisionId != decisionId)
            {
                Send(new { error = "tactical-v3 decision id is stale" });
                break;
            }
            Send(TacticalV3Wire.OracleQuery(selection));
            break;
        }

        case "duel_dagger_inspect":
        {
            if (!tacticalV3DuelHasReset)
                throw new InvalidDataException(
                    "tactical-v3 duel_dagger_inspect requires a successful duel_reset");
            long decisionId = root.GetProperty("decision_id").GetInt64();
            int candidateId = root.GetProperty("candidate_id").GetInt32();
            TacticalV3SelectiveDaggerInspection inspection =
                tacticalV3Duel!.InspectSelectiveDagger(decisionId, candidateId);
            Send(TacticalV3Wire.DaggerInspection(inspection));
            break;
        }

        case "duel_status":
            if (!tacticalV3DuelHasReset)
                throw new InvalidDataException(
                    "tactical-v3 duel_status requires a successful duel_reset");
            Send(new { internal_fallback_count = tacticalV3Duel!.InternalFallbackCount });
            break;

        case "duel_trace_enable":
        {            if (evidenceSession != null && !evidenceSession.Ended)
                throw new InvalidDataException("evidence session owns buffers until game close");
            if (environment != "tactical-v2")
                throw new InvalidDataException("duel trace is supported only for tactical-v2");
            bool enabled = root.GetProperty("enabled").GetBoolean();
            tacticalV2Trace.Enabled = enabled;
            if (!enabled) tacticalV2Trace.Drain();
            Send(new { enabled });
            break;
        }

        case "duel_trace_drain":
        {            if (evidenceSession != null && !evidenceSession.Ended)
                throw new InvalidDataException("evidence session owns buffers until game close");
            if (environment != "tactical-v2")
                throw new InvalidDataException("duel trace is supported only for tactical-v2");
            var transitions = tacticalV2Trace.Drain()
                .Select(TacticalEvaluationTrace.Project)
                .ToArray();
            Send(new { schema_version = 1, transitions });
            break;
        }

        case "duel_demo_enable":
        {            if (evidenceSession != null && !evidenceSession.Ended)
                throw new InvalidDataException("evidence session owns buffers until game close");
            if (environment != "tactical-v2")
                throw new InvalidDataException(
                    "duel demonstrations are supported only for tactical-v2");
            bool enabled = root.GetProperty("enabled").GetBoolean();
            tacticalV2Demonstrations.Enabled = enabled;
            if (!enabled) tacticalV2Demonstrations.Drain();
            Send(new { enabled });
            break;
        }

        case "duel_demo_drain":
        {            if (evidenceSession != null && !evidenceSession.Ended)
                throw new InvalidDataException("evidence session owns buffers until game close");
            if (environment != "tactical-v2")
                throw new InvalidDataException(
                    "duel demonstrations are supported only for tactical-v2");
            Send(new { schema_version = 1, decisions = tacticalV2Demonstrations.Drain() });
            break;
        }

        case "duel_dagger_configure":
        {            if (evidenceSession != null && !evidenceSession.Ended)
                throw new InvalidDataException("evidence session owns buffers until game close");
            if (environment != "tactical-v2")
                throw new InvalidDataException(
                    "duel DAgger is supported only for tactical-v2");
            RequireExactDaggerConfigureFields(root);
            bool enabled = RequireDaggerBoolean(root, "enabled", "enabled flag");
            int depth = RequirePositiveDaggerInteger(root, "depth", "depth");
            int expansionBudget = RequirePositiveDaggerInteger(
                root, "expansion_budget", "expansion budget");
            bool useHeuristic = RequireDaggerBoolean(
                root, "use_heuristic", "heuristic choice");
            if (!useHeuristic)
                throw new InvalidDataException(
                    "duel DAgger heuristic choice must be true");

            tacticalV2Duel ??= new TacticalV2DuelEnv(
                tacticalV2Config!, tacticalV2Trace, tacticalV2Demonstrations);
            if (enabled)
            {
                if (tacticalV2Config!.Game.FogOfWar)
                    throw new InvalidDataException(
                        "duel DAgger requires fog_of_war=false");
                tacticalV2Dagger.Enabled = true;
                tacticalV2Duel.DecisionObserver = new SelectiveDaggerObserver(
                    new BoundedSearchActionOracle(
                        tacticalV2Config.Game,
                        expansionBudget,
                        depth,
                        BoundedSearchAgent.HeuristicIdentity),
                    tacticalV2Dagger);
            }
            else
            {
                tacticalV2Dagger.Enabled = false;
                tacticalV2Dagger.Drain();
                tacticalV2Duel.DecisionObserver = null;
            }

            Send(new
            {
                enabled,
                depth,
                expansion_budget = expansionBudget,
                use_heuristic = useHeuristic,
            });
            break;
        }

        case "duel_dagger_drain":
        {            if (evidenceSession != null && !evidenceSession.Ended)
                throw new InvalidDataException("evidence session owns buffers until game close");
            if (environment != "tactical-v2")
                throw new InvalidDataException(
                    "duel DAgger is supported only for tactical-v2");
            Send(new { schema_version = 1, decisions = tacticalV2Dagger.Drain() });
            break;
        }


        case "duel_evidence_begin":
        {
            if (environment != "tactical-v2") throw new InvalidDataException("evidence sessions are supported only for tactical-v2");
            if (evidenceSession != null) throw new InvalidDataException("an evidence session is already active");
            OracleEvidenceBeginResponse begin = OracleEvidenceSession.Begin(OracleEvidenceSession.ParseBegin(root), EvidenceRuntime());
            evidenceSession = begin.Session;
            tacticalV2Trace.Enabled = true; tacticalV2Trace.Drain(); tacticalV2Dagger.Drain(); tacticalV2Preflight.Drain();
            InstallEvidenceObserver(evidenceSession.Expected!);
            Send(begin);
            break;
        }

        case "duel_evidence_game_close":
        {
            if (environment == MlContract.TacticalV3Version) throw new InvalidDataException("evidence sessions are supported only for tactical-v2");
            if (evidenceSession == null || evidenceSession.Ended) throw new InvalidDataException("no active evidence session");
            string[] fields = { "cmd", "schema_version", "session_id", "nonce", "candidate_index", "game_index" };
            if (!root.EnumerateObject().Select(p => p.Name).OrderBy(x => x, StringComparer.Ordinal).SequenceEqual(fields.OrderBy(x => x, StringComparer.Ordinal))) throw new InvalidDataException("evidence game close has unknown or missing fields");
            if (root.GetProperty("schema_version").GetInt32() != 1 || !evidenceGameOpen || tacticalV2Duel == null || !tacticalV2Duel.State.IsGameOver) throw new InvalidDataException("evidence game close requires a terminal game");
            OracleEvidenceScheduleItem item = evidenceSession.Expected ?? throw new InvalidDataException("evidence schedule is complete");
            IReadOnlyList<DuelTransition> transitions = tacticalV2Trace.Drain();
            IReadOnlyList<OraclePreflightBenchmarkRecord> benchmarks = tacticalV2Preflight.Drain();
            byte[] trace = JsonSerializer.SerializeToUtf8Bytes(new { schema_version = 1, transitions = transitions.Select(TacticalEvaluationTrace.Project).ToArray() });
            byte[] replay = Encoding.UTF8.GetBytes(tacticalV2Duel.ToReplay());
            byte[] benchmark = JsonSerializer.SerializeToUtf8Bytes(new { schema_version = 1, records = benchmarks });
            int expansions = benchmarks.Sum(row => row.First.ActualExpansionCount + row.Second.ActualExpansionCount);
            string outcome = !tacticalV2Duel.State.Winner.HasValue ? "draw" : tacticalV2Duel.State.Winner.Value == (item.Duel.LearnerSeat == 1 ? PlayerId.Player1 : PlayerId.Player0) ? "win" : "loss";
            var context = new OracleEvidenceGameContext(root.GetProperty("session_id").GetString() ?? "", root.GetProperty("nonce").GetString() ?? "", root.GetProperty("candidate_index").GetInt32(), root.GetProperty("game_index").GetInt32(), outcome, tacticalV2Duel.State.Winner.HasValue ? (int)tacticalV2Duel.State.Winner.Value : null, transitions.Count, benchmarks.Count, expansions);
            OracleEvidenceGameResponse closed = evidenceSession.CloseGame(context, trace, replay, benchmark);
            evidenceGameOpen = false;
            JsonElement receipt = JsonDocument.Parse(closed.Receipt.Utf8).RootElement.Clone();
            Send(new { receipt, receipt_sha256 = closed.Receipt.ReceiptSha256, receipt_utf8_base64 = Convert.ToBase64String(closed.Receipt.Utf8), trace = new { utf8_base64 = Convert.ToBase64String(closed.Trace.Bytes), sha256 = closed.Trace.Sha256, byte_size = closed.Trace.Bytes.Length }, replay = new { utf8_base64 = Convert.ToBase64String(closed.Replay.Bytes), sha256 = closed.Replay.Sha256, byte_size = closed.Replay.Bytes.Length }, benchmark = new { utf8_base64 = Convert.ToBase64String(closed.Benchmark.Bytes), sha256 = closed.Benchmark.Sha256, byte_size = closed.Benchmark.Bytes.Length } });
            if (evidenceSession.Expected != null) InstallEvidenceObserver(evidenceSession.Expected);
            break;
        }

        case "duel_evidence_end":
        {
            if (environment == MlContract.TacticalV3Version) throw new InvalidDataException("evidence sessions are supported only for tactical-v2");
            if (evidenceSession == null) throw new InvalidDataException("no active evidence session");
            (string sessionId, string nonce) = OracleEvidenceSession.ParseSessionRequest(root, "duel_evidence_end");
            Send(evidenceSession.End(sessionId, nonce));
            break;
        }
        case "duel_save":
        {
            if (environment == MlContract.TacticalV3Version && !tacticalV3DuelHasReset)
                throw new InvalidDataException("tactical-v3 duel_save requires a successful duel_reset");
            string path = root.TryGetProperty("path", out var pp) ? (pp.GetString() ?? "duel.replay") : "duel.replay";
            if (environment == MlContract.TacticalV3Version)
            {
                var dir = Path.GetDirectoryName(Path.GetFullPath(path));
                if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
                File.WriteAllText(path, tacticalV3Duel!.ToReplay());
                Send(new { saved = path });
            }
            else if (environment == "adaptive-v1" && adaptiveDuel != null)
            {
                var dir = Path.GetDirectoryName(Path.GetFullPath(path));
                if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
                File.WriteAllText(path, adaptiveDuel.ToReplay());
                Send(new { saved = path });
            }
            else if (environment == "tactical-v2" && tacticalV2Duel != null)
            {
                var dir = Path.GetDirectoryName(Path.GetFullPath(path));
                if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
                File.WriteAllText(path, tacticalV2Duel.ToReplay());
                Send(new { saved = path });
            }
            else if (duel != null)
            {
                var dir = Path.GetDirectoryName(Path.GetFullPath(path));
                if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
                File.WriteAllText(path, duel.ToReplay());
                Send(new { saved = path });
            }
            else Send(new { saved = "" });
            break;
        }

        case "close":
            return;
    }
}
