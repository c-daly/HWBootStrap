using System;
using System.IO;
using System.Text.Json;
using HexWars.Engine;
using HexWars.Engine.Rl;

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
for (int i = 0; i < args.Length; i++)
{
    if (args[i] == "--opponent" && i + 1 < args.Length) opponent = args[++i];
    else if (args[i] == "--seat" && i + 1 < args.Length) seat = int.Parse(args[++i]);
    else if (args[i] == "--environment") environment = i + 1 < args.Length ? args[++i] : "";
}
if (environment != "tactical-v1" && environment != "adaptive-v1")
{
    Console.Error.WriteLine($"unsupported environment '{environment}'");
    Environment.ExitCode = 2;
    return;
}

Func<int, IAgent> opponentFactory = opponent == "random"
    ? (s => new RandomAgent(s))
    : (s => new GreedyAgent(s));

PlayerId learningSeat = seat == 1 ? PlayerId.Player1 : PlayerId.Player0;
TacticalEnv? env = environment == "tactical-v1" ? new TacticalEnv(opponentFactory, learningSeat) : null;
AdaptiveTacticalEnv? adaptiveEnv = environment == "adaptive-v1"
    ? new AdaptiveTacticalEnv(opponentFactory,
        opponent == "random" ? s => new RandomDeploymentPolicy(s) : s => new CombinedArmsDeploymentPolicy(s),
        learningSeat)
    : null;
DuelEnv? duel = null; // created on first duel_* command (two external controllers)
AdaptiveDuelEnv? adaptiveDuel = null;
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

object Diagnostics(AdaptiveDiagnostics value) => new
{
    design_count = value.DesignCount,
    distinct_custom_templates_deployed = value.DistinctCustomTemplatesDeployed,
    pregame_decisions = value.PregameDecisions,
    invalid_sequences = value.InvalidSequences,
    deployment_completed = value.DeploymentCompleted,
};

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
            else
                Send(AdaptiveSpaces(adaptiveEnv!.Layout, adaptiveEnv.Config, adaptiveEnv.Contract));
            break;

        case "reset":
        {
            int seed = root.TryGetProperty("seed", out var s) ? s.GetInt32() : 0;
            if (env != null)
            {
                var obs = env.Reset(seed);
                Send(new { obs, mask = env.LegalActionMask() });
            }
            else
            {
                var obs = adaptiveEnv!.Reset(seed);
                Send(new
                {
                    obs,
                    mask = adaptiveEnv.LegalActionMask(),
                    deployment_complete = adaptiveEnv.DeploymentComplete,
                    diagnostics = Diagnostics(adaptiveEnv.Diagnostics),
                });
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
            else
            {
                var r = adaptiveEnv!.Step(action);
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
            break;
        }

        case "duel_spaces":
            if (environment == "tactical-v1")
            {
                duel ??= new DuelEnv();
                Send(Spaces(duel.ObservationLength, duel.ActionCount, duel.ObsChannels, duel.BoardH, duel.BoardW, duel.Config, MlEnvironmentKind.Duel));
            }
            else
            {
                adaptiveDuel ??= new AdaptiveDuelEnv();
                Send(AdaptiveSpaces(adaptiveDuel.Layout, adaptiveDuel.Config, adaptiveDuel.Contract));
            }
            break;

        case "duel_reset":
        {
            int seed = root.TryGetProperty("seed", out var s) ? s.GetInt32() : 0;
            string? p0 = root.TryGetProperty("p0", out var a) ? a.GetString() : null; // "external"(default)/greedy/random
            string? p1 = root.TryGetProperty("p1", out var b) ? b.GetString() : null;
            int learner = root.TryGetProperty("learner", out var lr) ? lr.GetInt32() : 0; // reward perspective
            if (environment == "tactical-v1")
            {
                duel ??= new DuelEnv();
                var v = duel.Reset(seed, MakeController(p0, seed * 2 + 1), MakeController(p1, seed * 2 + 2),
                                   learner == 1 ? PlayerId.Player1 : PlayerId.Player0);
                Send(new { obs = v.Observation, mask = v.ActionMask, seat = v.Seat, reward = v.Reward, winner = v.Winner, terminated = v.Terminated, truncated = v.Truncated });
            }
            else
            {
                adaptiveDuel ??= new AdaptiveDuelEnv();
                var v = adaptiveDuel.Reset(seed,
                    MakeController(p0, seed * 2 + 1), MakeController(p1, seed * 2 + 2),
                    MakeDeployment(p0, seed * 2 + 1), MakeDeployment(p1, seed * 2 + 2),
                    learner == 1 ? PlayerId.Player1 : PlayerId.Player0);
                Send(new
                {
                    obs = v.Observation, mask = v.ActionMask, seat = v.Seat, reward = v.Reward,
                    winner = v.Winner, terminated = v.Terminated, truncated = v.Truncated,
                    deployment_complete = v.DeploymentComplete, diagnostics = Diagnostics(v.Diagnostics),
                });
            }
            break;
        }

        case "duel_step":
        {
            int action = root.GetProperty("action").GetInt32();
            if (environment == "tactical-v1")
            {
                duel ??= new DuelEnv();
                var v = duel.Step(action);
                Send(new { obs = v.Observation, mask = v.ActionMask, seat = v.Seat, reward = v.Reward, winner = v.Winner, terminated = v.Terminated, truncated = v.Truncated });
            }
            else
            {
                adaptiveDuel ??= new AdaptiveDuelEnv();
                var v = adaptiveDuel.Step(action);
                Send(new
                {
                    obs = v.Observation, mask = v.ActionMask, seat = v.Seat, reward = v.Reward,
                    winner = v.Winner, terminated = v.Terminated, truncated = v.Truncated,
                    deployment_complete = v.DeploymentComplete, diagnostics = Diagnostics(v.Diagnostics),
                });
            }
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
