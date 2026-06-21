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
// Args: --opponent greedy|random   --seat 0|1
string opponent = "greedy";
int seat = 0;
for (int i = 0; i < args.Length - 1; i++)
{
    if (args[i] == "--opponent") opponent = args[i + 1];
    else if (args[i] == "--seat") seat = int.Parse(args[i + 1]);
}

Func<int, IAgent> opponentFactory = opponent == "random"
    ? (s => new RandomAgent(s))
    : (s => new GreedyAgent(s));

var env = new TacticalEnv(opponentFactory, seat == 1 ? PlayerId.Player1 : PlayerId.Player0);
DuelEnv? duel = null; // created on first duel_* command (two external controllers)
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

// Handshake: obs/action sizes + the spatial obs shape (so Python reshapes the board part to (C,H,W))
// + the env config (recorded into each run's params for reproducibility).
object Spaces(int obsLen, int nActions, int channels, int boardH, int boardW, EnvConfig c) => new
{
    obs_len = obsLen,
    n_actions = nActions,
    channels,
    board_h = boardH,
    board_w = boardW,
    globals = TacticalCoding.Globals,
    roster = c.Roster.Count,
    biomes = c.Game.BiomesEnabled,
    round_cap = c.Game.RoundCap,
    max_steps = c.MaxSteps,
    shape_scale = c.ShapeScale,
    step_penalty = c.StepPenalty,
    closing_weight = c.ClosingWeight,
    draw_credit_weight = c.DrawCreditWeight,
    points_weight = c.PointsWeight,
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
            Send(Spaces(env.ObservationLength, env.ActionCount, env.ObsChannels, env.BoardH, env.BoardW, env.Config));
            break;

        case "reset":
        {
            int seed = root.TryGetProperty("seed", out var s) ? s.GetInt32() : 0;
            var obs = env.Reset(seed);
            Send(new { obs, mask = env.LegalActionMask() });
            break;
        }

        case "step":
        {
            int action = root.GetProperty("action").GetInt32();
            var r = env.Step(action);
            Send(new { obs = r.Observation, reward = r.Reward, terminated = r.Terminated, truncated = r.Truncated, mask = r.ActionMask });
            break;
        }

        case "duel_spaces":
            duel ??= new DuelEnv();
            Send(Spaces(duel.ObservationLength, duel.ActionCount, duel.ObsChannels, duel.BoardH, duel.BoardW, duel.Config));
            break;

        case "duel_reset":
        {
            duel ??= new DuelEnv();
            int seed = root.TryGetProperty("seed", out var s) ? s.GetInt32() : 0;
            string? p0 = root.TryGetProperty("p0", out var a) ? a.GetString() : null; // "external"(default)/greedy/random
            string? p1 = root.TryGetProperty("p1", out var b) ? b.GetString() : null;
            int learner = root.TryGetProperty("learner", out var lr) ? lr.GetInt32() : 0; // reward perspective
            var v = duel.Reset(seed, MakeController(p0, seed * 2 + 1), MakeController(p1, seed * 2 + 2),
                               learner == 1 ? PlayerId.Player1 : PlayerId.Player0);
            Send(new { obs = v.Observation, mask = v.ActionMask, seat = v.Seat, reward = v.Reward, winner = v.Winner, terminated = v.Terminated, truncated = v.Truncated });
            break;
        }

        case "duel_step":
        {
            duel ??= new DuelEnv();
            int action = root.GetProperty("action").GetInt32();
            var v = duel.Step(action);
            Send(new { obs = v.Observation, mask = v.ActionMask, seat = v.Seat, reward = v.Reward, winner = v.Winner, terminated = v.Terminated, truncated = v.Truncated });
            break;
        }

        case "duel_save":
        {
            string path = root.TryGetProperty("path", out var pp) ? (pp.GetString() ?? "duel.replay") : "duel.replay";
            if (duel != null)
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
