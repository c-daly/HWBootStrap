using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using HexWars.Engine;
using HexWars.Engine.Rl;
using HexWars.GymServer;

// Headless RL bridge: wraps a TacticalEnv and speaks one JSON object per line over stdin/stdout, so a
// Python gymnasium.Env can drive it as a subprocess. Cross-platform (.NET 8) — built for WSL2/Linux.
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
if (environment != "tactical-v1" && environment != "adaptive-v1" && environment != "tactical-v2")
{
    Console.Error.WriteLine($"unsupported environment '{environment}'");
    Environment.ExitCode = 2;
    return;
}

TrainingScenario scenario = scenarioFile == null
    ? TrainingScenario.CreateStandard(environment)
    : ScenarioJson.Load(scenarioFile);
if (!string.Equals(scenario.Environment, environment, StringComparison.Ordinal))
    throw new InvalidDataException("scenario environment does not match --environment");

EnvConfig? tacticalConfig = environment == "tactical-v1" ? scenario.BuildTactical() : null;
AdaptiveEnvConfig? adaptiveConfig = environment == "adaptive-v1" ? scenario.BuildAdaptive() : null;
TacticalV2Config? tacticalV2Config = environment == "tactical-v2" ? scenario.BuildTacticalV2() : null;

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
var tacticalV2Trace = new BufferedDuelTransitionSink();
var tacticalV2Demonstrations = new BufferedTacticalV2DemonstrationSink();
DuelEnv? duel = null; // created on first duel_* command (two external controllers)
AdaptiveDuelEnv? adaptiveDuel = null;
TacticalV2DuelEnv? tacticalV2Duel = null;
var output = Console.Out;

void Send(object payload)
{
    output.WriteLine(JsonSerializer.Serialize(payload));
    output.Flush();
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
    string cmd = root.GetProperty("cmd").GetString() ?? "";

    switch (cmd)
    {
        case "spaces":
            if (env != null)
                Send(Spaces(env.ObservationLength, env.ActionCount, env.ObsChannels, env.BoardH, env.BoardW, env.Config, MlEnvironmentKind.Tactical));
            else if (adaptiveEnv != null)
                Send(AdaptiveSpaces(adaptiveEnv.Layout, adaptiveEnv.Config, adaptiveEnv.Contract));
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
            else
            {
                var obs = tacticalV2Env!.Reset(seed);
                Send(new { obs, mask = tacticalV2Env.LegalActionMask() });
            }
            break;
        }

        case "step":
        {
            int action = root.GetProperty("action").GetInt32();
            if (env != null)
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
            string? startProfile = root.TryGetProperty("start_profile", out var sp)
                ? sp.GetString()
                : null;
            bool hasReferenceSeat = root.TryGetProperty("reference_seat", out var rs);
            int referenceSeat = hasReferenceSeat ? rs.GetInt32() : 0;
            if (learner is < 0 or > 1)
                throw new InvalidDataException("duel_reset learner must be 0 or 1");
            if (hasReferenceSeat && referenceSeat is < 0 or > 1)
                throw new InvalidDataException("duel_reset reference_seat must be 0 or 1");
            if (environment != "tactical-v2" && (startProfile != null || hasReferenceSeat))
                throw new InvalidDataException("duel_reset start_profile/reference_seat are supported only for tactical-v2");
            if (startProfile != null && !hasReferenceSeat)
                throw new InvalidDataException("duel_reset reference_seat is required when start_profile is supplied");
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
            int action = root.GetProperty("action").GetInt32();
            if (environment == "tactical-v1")
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

        case "duel_trace_enable":
        {
            if (environment != "tactical-v2")
                throw new InvalidDataException("duel trace is supported only for tactical-v2");
            bool enabled = root.GetProperty("enabled").GetBoolean();
            tacticalV2Trace.Enabled = enabled;
            if (!enabled) tacticalV2Trace.Drain();
            Send(new { enabled });
            break;
        }

        case "duel_trace_drain":
        {
            if (environment != "tactical-v2")
                throw new InvalidDataException("duel trace is supported only for tactical-v2");
            var transitions = tacticalV2Trace.Drain()
                .Select(TacticalEvaluationTrace.Project)
                .ToArray();
            Send(new { schema_version = 1, transitions });
            break;
        }

        case "duel_demo_enable":
        {
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
        {
            if (environment != "tactical-v2")
                throw new InvalidDataException(
                    "duel demonstrations are supported only for tactical-v2");
            Send(new { schema_version = 1, decisions = tacticalV2Demonstrations.Drain() });
            break;
        }

        case "duel_save":
        {
            string path = root.TryGetProperty("path", out var pp) ? (pp.GetString() ?? "duel.replay") : "duel.replay";
            if (environment == "adaptive-v1" && adaptiveDuel != null)
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
