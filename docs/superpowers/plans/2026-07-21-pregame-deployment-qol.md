# Pregame Deployment QoL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an authoritative pregame phase where both armies start in mixed automatic formations, may rearrange freely inside their own hidden deployment zones, and lock in before round one.

**Architecture:** Deployment is explicit immutable engine state, not a presentation-only mode. Two new commands bypass normal active-player gating only during deployment and are validated by the engine. Network/replay serialization carries phase/readiness, while Unity adds a focused controller and handoff/ready UI.

**Tech Stack:** HexWars immutable C# engine, WebSocket protocol, ReplayFile, Unity uGUI/input, NUnit.

## Global Constraints

- Automatic placement remains a valid no-input default.
- Automatic formations interleave roles and use independent deterministic seat seeds.
- Deployment relocation is free and limited to passable, unoccupied home-zone hexes.
- Opposing placements are hidden until both players are ready, regardless of fog.
- Online deployment is simultaneous; hotseat is sequential behind a handoff screen.
- Ready is final. Round one begins only after both are ready, with Player 1 active and empty turn bookkeeping.
- Existing tactical RL environments bypass deployment until their model contract is deliberately upgraded.
- After every C# edit, run Unity `check_compile_errors`.

---

### Task 1: Deployment state model and mixed automatic formations

**Files:**
- Modify: `engine/HexWars.Engine/GameState.cs:20`
- Modify: `engine/HexWars.Engine/Net/GameSetup.cs:69`
- Modify: `engine/HexWars.Engine/Rl/TacticalLayout.cs:45`
- Modify: `Assets/HexWars/Presentation/GameBootstrap.cs:115`
- Create: `engine/HexWars.Engine.Tests/DeploymentFormationTests.cs`
- Modify: `engine/HexWars.Engine.Tests/ArmyCompositionTests.cs`

**Interfaces:**
- Produces: `GamePhase { Deployment, Playing }`, `GameState.Phase`, `GameState.IsDeploymentReady(PlayerId)`.
- Produces: `GameFactory.Build(..., bool beginInDeployment = true)`.
- Preserves: `TacticalLayout.NewGame(seed)` creates `GamePhase.Playing` for old model compatibility.

- [ ] **Step 1: Write failing tests for phase defaults, per-seat readiness, same-seed reproducibility, non-mirrored seat formations, and role interleaving across sorted frontage slots.**
- [ ] **Step 2: Run focused tests and confirm failure before implementation.**
- [ ] **Step 3: Add immutable phase/readiness fields to `GameState`, constructor, `Clone`, and every engine state-copy site. Default constructor values must be `Playing/ready` so hand-built tests and older callers preserve behavior.**

```csharp
public enum GamePhase { Deployment, Playing }
public GamePhase Phase { get; }
public bool Player0Ready { get; }
public bool Player1Ready { get; }
public bool IsDeploymentReady(PlayerId player) => player == PlayerId.Player0 ? Player0Ready : Player1Ready;
```

- [ ] **Step 4: Replace grouped `BuildArmy` placement with a deterministic interleave plus seat-specific shuffle of valid zone slots.** Do not change army composition or unit stats.
- [ ] **Step 5: Make every real new game start in Deployment.** Normal `GameFactory.Build` uses Deployment; the legacy inspector-driven `GameBootstrap.NewGame()` explicitly constructs the same phase/readiness. `TacticalLayout.NewGame` and `GameBootstrap.StartDemo` explicitly request Playing so training and the title-screen demo cannot stall at Ready.
- [ ] **Step 6: Run the complete engine suite, `check_compile_errors`, and a title-demo smoke check. Confirm all prior tactical tests remain unchanged and the demo begins playing without deployment UI.**
- [ ] **Step 7: Commit.**

```powershell
git add engine/HexWars.Engine/GameState.cs engine/HexWars.Engine/Net/GameSetup.cs engine/HexWars.Engine/Rl/TacticalLayout.cs engine/HexWars.Engine.Tests/DeploymentFormationTests.cs engine/HexWars.Engine.Tests/ArmyCompositionTests.cs Assets/HexWars/Presentation/GameBootstrap.cs
git commit -m "feat(engine): add deployment phase and mixed formations"
```

### Task 2: Authoritative deployment commands

**Files:**
- Modify: `engine/HexWars.Engine/Command.cs`
- Modify: `engine/HexWars.Engine/GameEngine.cs:12`
- Modify: `engine/HexWars.Engine/RejectionReason.cs`
- Modify: `engine/HexWars.Engine/LegalMoves.cs`
- Create: `engine/HexWars.Engine.Tests/DeploymentCommandTests.cs`

**Interfaces:**
- Produces: `RepositionStartingUnit(PlayerId Issuer, int UnitId, HexCoord Cell)` and `ReadyDeployment(PlayerId Issuer)`.
- Produces rejection reasons: `DeploymentInProgress`, `NotInDeployment`, `DeploymentAlreadyReady`, `OutsideDeploymentZone`, plus existing occupancy/passability/unit errors.

- [ ] **Step 1: Write failing tests for legal relocation; opponent unit; dead/missing unit; outside zone; impassable/occupied cell; either seat acting regardless of `ActivePlayer`; duplicate Ready; gameplay command during deployment; deployment command after Playing; and the second Ready transition.**
- [ ] **Step 2: Run focused tests and confirm failure.**
- [ ] **Step 3: Route deployment commands before the normal issuer-versus-`ActivePlayer` check. Reject every non-deployment command while `Phase == Deployment`.**

```csharp
if (state.Phase == GamePhase.Deployment)
    return command switch
    {
        RepositionStartingUnit c => ApplyReposition(state, c),
        ReadyDeployment c => ApplyReadyDeployment(state, c),
        _ => Result.Reject(state, RejectionReason.DeploymentInProgress)
    };
```

- [ ] **Step 4: Implement relocation as an immutable owner-unit replacement. Do not modify points, entity IDs, movement/attack sets, round, or active player.**
- [ ] **Step 5: Implement Ready as an immutable flag update. When both flags become true, set Playing, Player0 active, round 1, and empty moved/attacked/movement-spent collections. Skip win finalization during deployment.**
- [ ] **Step 6: Ensure `LegalMoves.For` returns no combat/economy commands during deployment; deployment UI computes relocation destinations from the zone directly.**
- [ ] **Step 7: Run the complete engine suite; expect zero failures.**
- [ ] **Step 8: Commit.**

```powershell
git add engine/HexWars.Engine/Command.cs engine/HexWars.Engine/GameEngine.cs engine/HexWars.Engine/RejectionReason.cs engine/HexWars.Engine/LegalMoves.cs engine/HexWars.Engine.Tests/DeploymentCommandTests.cs
git commit -m "feat(engine): validate pregame relocation and readiness"
```

### Task 3: Command, replay, and network compatibility

**Files:**
- Modify: `engine/HexWars.Engine/Net/CommandWire.cs`
- Modify: `engine/HexWars.Engine/ReplayFile.cs`
- Modify: `engine/HexWars.Engine.Tests/CommandWireTests.cs`
- Modify: `engine/HexWars.Engine.Tests/ReplayFileTests.cs`
- Modify: `engine/HexWars.Engine.Tests/MatchHubTests.cs`
- Modify: `engine/HexWars.Engine.Tests/TokenRejoinTests.cs`

**Interfaces:**
- Consumes: deployment commands and `GameState` phase/readiness.
- Produces: replay `PHASE <phase> <p0Ready> <p1Ready>` record; absence means Playing/ready/ready.

- [ ] **Step 1: Write failing round-trip tests for both commands and deployment phase/readiness. Add a legacy replay test with no PHASE record that loads directly into Playing.**
- [ ] **Step 2: Run focused tests and confirm failure.**
- [ ] **Step 3: Add compact wire records for relocation and Ready, then add the optional replay PHASE line and backward-compatible reader defaults.**
- [ ] **Step 4: Add MatchHub tests proving either seated connection may submit its own deployment commands before normal turn order, an unseated connection cannot, and reconnect receives current placement/readiness through START replay reconstruction.**
- [ ] **Step 5: Run the full engine suite and server self-test. Rebuild/copy the Release engine DLL and run `check_compile_errors`.**
- [ ] **Step 6: Commit.**

```powershell
git add engine/HexWars.Engine/Net/CommandWire.cs engine/HexWars.Engine/ReplayFile.cs engine/HexWars.Engine.Tests/CommandWireTests.cs engine/HexWars.Engine.Tests/ReplayFileTests.cs engine/HexWars.Engine.Tests/MatchHubTests.cs engine/HexWars.Engine.Tests/TokenRejoinTests.cs
git commit -m "feat(net): serialize deployment state and commands"
```

### Task 4: Deployment visibility policy

**Files:**
- Modify: `Assets/HexWars/Presentation/FogPresentation.cs`
- Modify: `Assets/HexWars/Presentation/TokenStore.cs`
- Modify: `Assets/HexWars/Presentation/GameBootstrap.cs:356`
- Create: `Assets/HexWars/Tests/Editor/DeploymentVisibilityTests.cs`
- Create: `Assets/HexWars/Tests/Editor/DeploymentVisibilityTests.cs.meta`

**Interfaces:**
- Produces: `FogPresentation.IsUnitVisible(GameState state, PlayerId? viewer, Unit unit)` with deployment override.
- Produces: a deployment-aware `GameBootstrap.FogViewerFor` that returns the online seat, the human seat versus AI, or the not-yet-ready hotseat player even when ordinary fog is disabled.

- [ ] **Step 1: Write EditMode tests proving own units show, enemy units hide for both fog settings during Deployment, a seatless spectator sees neither hidden army, hotseat view changes from P0 to P1 after the first Ready, and normal fog behavior resumes in Playing.**
- [ ] **Step 2: Run focused tests and confirm current fog-off rendering reveals the opponent.**
- [ ] **Step 3: Centralize unit visibility through the tested helper and have `TokenStore.Sync` use it for every unit. Make `FogViewerFor` choose a meaningful deployment viewer before its ordinary fog-disabled early return. Do not hide board terrain or deployment-zone highlights.**
- [ ] **Step 4: Run `check_compile_errors` and visibility tests.**
- [ ] **Step 5: Commit.**

```powershell
git add Assets/HexWars/Presentation/FogPresentation.cs Assets/HexWars/Presentation/TokenStore.cs Assets/HexWars/Presentation/GameBootstrap.cs Assets/HexWars/Tests/Editor/DeploymentVisibilityTests.cs Assets/HexWars/Tests/Editor/DeploymentVisibilityTests.cs.meta
git commit -m "feat(deployment): hide opposing formations before ready"
```

### Task 5: Unity deployment controller and Ready UI

**Files:**
- Create: `Assets/HexWars/Presentation/DeploymentController.cs`
- Create: `Assets/HexWars/Presentation/DeploymentController.cs.meta`
- Create: `Assets/HexWars/Presentation/DeploymentOverlay.cs`
- Create: `Assets/HexWars/Presentation/DeploymentOverlay.cs.meta`
- Modify: `Assets/HexWars/Presentation/GameBootstrap.cs`
- Modify: `Assets/HexWars/Presentation/UnitInputController.cs`
- Modify: `Assets/HexWars/Presentation/MovementHighlightController.cs`
- Modify: `Assets/HexWars/Presentation/GameHud.cs`
- Create: `Assets/HexWars/Tests/Editor/DeploymentControllerTests.cs`
- Create: `Assets/HexWars/Tests/Editor/DeploymentControllerTests.cs.meta`

**Interfaces:**
- Consumes: `RepositionStartingUnit`, `ReadyDeployment`, `Board.DeploymentZone`, local seat ownership.
- Produces: selection → valid-zone highlight → relocation command, and Ready/handoff overlay state.

- [ ] **Step 1: Write EditMode tests for local-seat selection, valid destination classification, highlight clearing, Ready disablement after submission, online waiting copy, and suppression of normal movement/attack/build/deploy input.**
- [ ] **Step 2: Run focused tests and confirm failure before implementation.**
- [ ] **Step 3: Implement `DeploymentController` as the only board-input owner during Deployment. Reuse tile highlight pooling but use a distinct placement color; clicking own unit selects it, clicking a valid cell sends relocation, and clicking invalid space clears selection.**
- [ ] **Step 4: Implement `DeploymentOverlay` with phase instructions, Ready, and waiting status. Online uses the assigned seat. Hotseat exposes only the current local deployer and shows a full-screen handoff catcher after the first Ready before revealing the second army.**
- [ ] **Step 5: Gate `UnitInputController`, `BarracksPanel`, Designer, build controls, and End Turn while deployment is active. Restore them immediately on transition to Playing.**
- [ ] **Step 6: Run `check_compile_errors`, focused tests, and Play Mode local hotseat checks. Inspect Unity logs.**
- [ ] **Step 7: Commit.**

```powershell
git add Assets/HexWars/Presentation/DeploymentController.cs Assets/HexWars/Presentation/DeploymentController.cs.meta Assets/HexWars/Presentation/DeploymentOverlay.cs Assets/HexWars/Presentation/DeploymentOverlay.cs.meta Assets/HexWars/Presentation/GameBootstrap.cs Assets/HexWars/Presentation/UnitInputController.cs Assets/HexWars/Presentation/MovementHighlightController.cs Assets/HexWars/Presentation/GameHud.cs Assets/HexWars/Tests/Editor/DeploymentControllerTests.cs Assets/HexWars/Tests/Editor/DeploymentControllerTests.cs.meta
git commit -m "feat(deployment): add placement controls and ready flow"
```

### Task 6: AI and online completion

**Files:**
- Modify: `Assets/HexWars/Presentation/AiOpponent.cs`
- Modify: `engine/HexWars.Engine/GreedyAgent.cs`
- Create: `engine/HexWars.Engine.Tests/DeploymentAgentTests.cs`
- Modify: `Assets/HexWars/Tests/Editor/DeploymentControllerTests.cs`

**Interfaces:**
- Produces: deterministic AI acceptance/adjustment and Ready submission.

- [ ] **Step 1: Write tests proving the AI produces only legal deployment commands, terminates by Ready, and is deterministic for the same seed.**
- [ ] **Step 2: Implement a bounded deployment heuristic that spreads complementary roles across lanes and then submits Ready. The automatic mixed formation is a legal fallback if no improving relocation exists.**
- [ ] **Step 3: Make `AiOpponent` drive deployment commands without waiting for normal active-player turn gating, while preserving presentation pacing.**
- [ ] **Step 4: Run engine tests, rebuild/copy the DLL, run `check_compile_errors`, and complete a vs-AI deployment in Play Mode.**
- [ ] **Step 5: Run a two-browser online game: rearrange both sides simultaneously, verify hidden opponents, Ready one side early, reconnect the other, then begin round one. Inspect client/server logs.**
- [ ] **Step 6: Commit.**

```powershell
git add Assets/HexWars/Presentation/AiOpponent.cs engine/HexWars.Engine/GreedyAgent.cs engine/HexWars.Engine.Tests/DeploymentAgentTests.cs Assets/HexWars/Tests/Editor/DeploymentControllerTests.cs
git commit -m "feat(deployment): integrate AI and simultaneous online setup"
```

### Task 7: Integrated verification

**Files:**
- Modify only if verification exposes a defect in files already listed above.

- [ ] **Step 1: Run the full engine suite and full Unity EditMode suite; expect zero failures.**
- [ ] **Step 2: Run deterministic tactical RL tests and confirm their observation/action sizes and start phase remain unchanged.**
- [ ] **Step 3: Verify local hotseat, vs-AI, online P0/P1, reconnect, replay save/load, fog on/off, portrait, and landscape deployment flows.**
- [ ] **Step 4: Confirm the first Playing state is round 1, Player 1 active, with empty moved/attacked/movement-spent collections and no deployment commands counted as actions.**
- [ ] **Step 5: Inspect Unity/server logs, run `git diff --check`, and commit only narrowly scoped corrections.**
