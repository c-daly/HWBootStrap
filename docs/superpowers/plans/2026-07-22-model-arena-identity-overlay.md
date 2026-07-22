# Model Arena Identity Overlay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep player/controller identity, resolved checkpoint, active seat, and current-session W-L-D visible while a Unity model duel is playing.

**Architecture:** A pure formatter converts `ModelDuelDriver` state and `PolicySeatInfo` into immutable row snapshots. A dedicated `ModelArenaIdentityOverlay` renders those snapshots independently of the collapsible combat log; the driver owns results and resolved bridge metadata and exposes one snapshot method.

**Tech Stack:** Unity 6000.5, C#, NUnit EditMode tests, Unity immediate-mode GUI.

## Global Constraints

- The overlay appears only while `ModelDuelDriver` is active and remains visible when the combat log is collapsed.
- P1 is cyan and P2 is red; the current seat has an explicit active marker.
- Model rows show the actually resolved algorithm, checkpoint filename, and training step; full paths remain in the ML Lab.
- Records are session-local `W-L-D`; win percentage is `wins / total completed games`, with an em dash before the first result.
- Live checkpoint identity changes only after a successful between-game reload.
- Missing metadata and reload failures must not invent a checkpoint or discard the last successfully resolved identity.

---

### Task 1: Pure arena identity snapshot and record formatting

**Files:**
- Create: `Assets/HexWars/Presentation/ModelArenaIdentity.cs`
- Create: `Assets/HexWars/Tests/Editor/ModelArenaIdentityTests.cs`
- Create: corresponding Unity `.meta` files by importing through Unity

**Interfaces:**
- Consumes: `PolicySeatInfo`, controller spec strings, current seat, `P0Wins`, `P1Wins`, and `Draws`.
- Produces: `ModelArenaSeatIdentity` and `ModelArenaIdentity.Build(...)`, `FormatRecord(...)`, and `MiddleTruncate(...)`.

- [ ] **Step 1: Write the failing formatter tests**

```csharp
using NUnit.Framework;

namespace HexWars.Presentation.Tests
{
    public sealed class ModelArenaIdentityTests
    {
        [Test]
        public void Build_LabelsScriptedSeatsAndMarksCurrentSeat()
        {
            var rows = ModelArenaIdentity.Build("greedy", "random", null, null, 1, 0, 0, 0);

            Assert.That(rows[0].Player, Is.EqualTo("P1"));
            Assert.That(rows[0].Controller, Is.EqualTo("Greedy"));
            Assert.That(rows[0].IsActive, Is.False);
            Assert.That(rows[1].Controller, Is.EqualTo("Random"));
            Assert.That(rows[1].IsActive, Is.True);
            Assert.That(rows[0].Record, Is.EqualTo("0-0-0 (—)"));
        }

        [Test]
        public void Build_UsesResolvedCheckpointAndMirrorsRecords()
        {
            var resolved = PolicyBridge.ParseReady(
                "{\"ready\":true,\"model_seats\":[0],\"seat_models\":[{\"seat\":0,\"kind\":\"run\",\"path\":\"C:/runs/alpha/checkpoints/model_20480_steps.zip\",\"algorithm\":\"maskable_ppo\",\"step\":20480}]}"
            ).Seats[0];

            var rows = ModelArenaIdentity.Build("run:C:/runs/alpha", "greedy", resolved, null, 0, 3, 1, 1);

            Assert.That(rows[0].Controller, Is.EqualTo("alpha"));
            Assert.That(rows[0].Checkpoint, Is.EqualTo("model_20480_steps.zip"));
            Assert.That(rows[0].Algorithm, Is.EqualTo("Maskable PPO"));
            Assert.That(rows[0].Step, Is.EqualTo("step 20,480"));
            Assert.That(rows[0].Record, Is.EqualTo("3-1-1 (60%)"));
            Assert.That(rows[1].Record, Is.EqualTo("1-3-1 (20%)"));
        }

        [Test]
        public void MiddleTruncate_PreservesBothEnds()
        {
            Assert.That(ModelArenaIdentity.MiddleTruncate("abcdefghijklmnop", 11), Is.EqualTo("abcd…klmnop"));
        }
    }
}
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```powershell
& 'C:\Program Files\Unity\Hub\Editor\6000.5.0f1\Editor\Unity.exe' -batchmode -nographics -projectPath 'C:\Users\cddal\HexWars\.worktrees\ml-full-game-actions' -accept-apiupdate -runTests -testPlatform EditMode -testFilter 'HexWars.Presentation.Tests.ModelArenaIdentityTests' -testResults 'Logs\model-arena-identity-red.xml' -logFile 'Logs\model-arena-identity-red.log'
```

Expected: FAIL because `ModelArenaIdentity` does not exist.

- [ ] **Step 3: Implement the immutable rows and pure formatter**

```csharp
using System;
using System.IO;

namespace HexWars.Presentation
{
    public sealed class ModelArenaSeatIdentity
    {
        public string Player { get; internal set; }
        public string Controller { get; internal set; }
        public string Algorithm { get; internal set; }
        public string Checkpoint { get; internal set; }
        public string Step { get; internal set; }
        public string Record { get; internal set; }
        public bool IsActive { get; internal set; }
    }

    public static class ModelArenaIdentity
    {
        public static ModelArenaSeatIdentity[] Build(
            string p0Spec, string p1Spec, PolicySeatInfo p0, PolicySeatInfo p1,
            int currentSeat, int p0Wins, int p1Wins, int draws) => new[]
        {
            BuildSeat(0, p0Spec, p0, currentSeat == 0, p0Wins, p1Wins, draws),
            BuildSeat(1, p1Spec, p1, currentSeat == 1, p1Wins, p0Wins, draws),
        };

        static ModelArenaSeatIdentity BuildSeat(
            int seat, string spec, PolicySeatInfo resolved, bool active,
            int wins, int losses, int draws)
        {
            bool scripted = string.Equals(spec, "greedy", StringComparison.OrdinalIgnoreCase)
                           || string.Equals(spec, "random", StringComparison.OrdinalIgnoreCase);
            string controller = scripted ? Capitalize(spec) : RunName(spec, resolved);
            return new ModelArenaSeatIdentity
            {
                Player = seat == 0 ? "P1" : "P2",
                Controller = controller,
                Algorithm = resolved == null ? string.Empty : FriendlyAlgorithm(resolved.Algorithm),
                Checkpoint = resolved == null ? (scripted ? string.Empty : "loading checkpoint") : Path.GetFileName(resolved.Path),
                Step = resolved == null || resolved.Step <= 0 ? string.Empty : $"step {resolved.Step:N0}",
                Record = FormatRecord(wins, losses, draws),
                IsActive = active,
            };
        }

        public static string FormatRecord(int wins, int losses, int draws)
        {
            int total = wins + losses + draws;
            return total == 0 ? $"{wins}-{losses}-{draws} (—)"
                              : $"{wins}-{losses}-{draws} ({100.0 * wins / total:0.#}%)";
        }

        public static string MiddleTruncate(string value, int max)
        {
            if (string.IsNullOrEmpty(value) || value.Length <= max) return value ?? string.Empty;
            int left = (max - 1) / 2;
            return value.Substring(0, left) + "…" + value.Substring(value.Length - (max - left - 1));
        }

        static string FriendlyAlgorithm(string value) => value switch
        {
            "maskable_ppo" => "Maskable PPO",
            "masked_dqn" => "Masked DQN",
            _ => string.IsNullOrWhiteSpace(value) ? "unknown algorithm" : value,
        };

        static string RunName(string spec, PolicySeatInfo resolved)
        {
            string path = resolved?.Kind == "run" ? Directory.GetParent(resolved.Path)?.Parent?.FullName : null;
            if (string.IsNullOrWhiteSpace(path))
            {
                int colon = (spec ?? string.Empty).IndexOf(':');
                path = colon >= 0 ? spec.Substring(colon + 1) : spec;
            }
            return string.IsNullOrWhiteSpace(path) ? "model" : new DirectoryInfo(path).Name;
        }

        static string Capitalize(string value) => string.IsNullOrEmpty(value)
            ? string.Empty : char.ToUpperInvariant(value[0]) + value.Substring(1).ToLowerInvariant();
    }
}
```

- [ ] **Step 4: Run the focused test and verify GREEN**

Run the Step 2 command with result/log names changed from `red` to `green`.

Expected: 3 tests pass and Unity reports no compiler errors.

- [ ] **Step 5: Commit the formatter slice**

```powershell
git add Assets/HexWars/Presentation/ModelArenaIdentity.cs Assets/HexWars/Presentation/ModelArenaIdentity.cs.meta Assets/HexWars/Tests/Editor/ModelArenaIdentityTests.cs Assets/HexWars/Tests/Editor/ModelArenaIdentityTests.cs.meta
git commit -m "feat(ml): format model arena identities"
```

### Task 2: Always-visible arena overlay and driver snapshot

**Files:**
- Create: `Assets/HexWars/Presentation/ModelArenaIdentityOverlay.cs`
- Modify: `Assets/HexWars/Presentation/ModelDuelDriver.cs`
- Modify: `Assets/HexWars/Tests/Editor/ModelArenaIdentityTests.cs`
- Create: corresponding Unity `.meta` for the new component

**Interfaces:**
- Consumes: `ModelArenaIdentity.Build(...)` from Task 1 and live fields already owned by `ModelDuelDriver`.
- Produces: `ModelDuelDriver.IdentitySnapshot()` and a self-contained overlay component automatically required by the driver.

- [ ] **Step 1: Add a failing driver snapshot test**

```csharp
using UnityEngine;

[Test]
public void Driver_AlwaysCarriesIndependentIdentityOverlay()
{
    var go = new GameObject("arena", typeof(BoardRenderer), typeof(ModelDuelDriver));
    try
    {
        Assert.That(go.GetComponent<ModelArenaIdentityOverlay>(), Is.Not.Null);
        var driver = go.GetComponent<ModelDuelDriver>();
        driver.P0Spec = "greedy";
        driver.P1Spec = "random";
        Assert.That(driver.IdentitySnapshot()[0].Controller, Is.EqualTo("Greedy"));
    }
    finally { Object.DestroyImmediate(go); }
}
```

- [ ] **Step 2: Run the focused test and verify RED**

Run the Task 1 focused Unity command.

Expected: FAIL because the overlay type and `IdentitySnapshot()` do not exist.

- [ ] **Step 3: Wire the driver to the snapshot formatter**

Add the component requirement above `ModelDuelDriver`:

```csharp
[RequireComponent(typeof(BoardRenderer))]
[RequireComponent(typeof(ModelArenaIdentityOverlay))]
public sealed class ModelDuelDriver : MonoBehaviour
```

Add this public method:

```csharp
public ModelArenaSeatIdentity[] IdentitySnapshot() => ModelArenaIdentity.Build(
    P0Spec, P1Spec, P0Resolved, P1Resolved, CurrentSeat,
    P0Wins, P1Wins, Draws);
```

- [ ] **Step 4: Implement the independent OnGUI overlay**

Create `ModelArenaIdentityOverlay.cs` with a cached `ModelDuelDriver`, two cached GUI styles created inside `OnGUI`, logical-height scaling matching `EventConsole`, and two rows at the upper-left. Each row must render:

```csharp
string marker = row.IsActive ? "▶ " : "  ";
string model = string.Join(" · ", new[] { row.Controller, row.Algorithm, row.Checkpoint, row.Step }
    .Where(value => !string.IsNullOrWhiteSpace(value)));
string text = $"{marker}{row.Player}  {ModelArenaIdentity.MiddleTruncate(model, narrow ? 42 : 72)}  ·  {row.Record}";
```

Use cyan `new Color(0.44f, 0.69f, 1f)` for P1, red `new Color(1f, 0.48f, 0.42f)` for P2, a translucent black background, and stack rows when `Screen.width < Screen.height`. Do not read or modify `EventConsole` state.

- [ ] **Step 5: Run focused and presentation tests**

Run:

```powershell
& 'C:\Program Files\Unity\Hub\Editor\6000.5.0f1\Editor\Unity.exe' -batchmode -nographics -projectPath 'C:\Users\cddal\HexWars\.worktrees\ml-full-game-actions' -accept-apiupdate -runTests -testPlatform EditMode -testFilter 'HexWars.Presentation.Tests.ModelArenaIdentityTests' -testResults 'Logs\model-arena-identity-results.xml' -logFile 'Logs\model-arena-identity.log'
```

Expected: all identity tests pass with zero compiler errors.

- [ ] **Step 6: Manually verify arena behavior**

Launch Greedy versus a fixed checkpoint and confirm both rows, active marker, checkpoint filename, step, and mirrored records. Collapse the combat log and resize the Game view to landscape and portrait; the identity overlay must remain visible. Launch a live run and confirm its row changes only after a game ends and a newer checkpoint successfully reloads.

- [ ] **Step 7: Commit the overlay slice**

```powershell
git add Assets/HexWars/Presentation/ModelDuelDriver.cs Assets/HexWars/Presentation/ModelArenaIdentityOverlay.cs Assets/HexWars/Presentation/ModelArenaIdentityOverlay.cs.meta Assets/HexWars/Tests/Editor/ModelArenaIdentityTests.cs
git commit -m "feat(ml): show arena model identity and record"
```

