# Opponent-Turn Presentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every action — AI, remote, spectator, replay, local — plays through one animated presentation pipeline: persistent unit tokens tween instead of teleporting, attacks land with impact-timed popups, fog hides what it should and shows a tracer from the dark for unseen attackers.

**Architecture:** A `TokenStore` (persistent `unit id → token` registry) replaces BoardRenderer's destroy-and-rebuild of entities; an `ActionPresenter` queue animates `(prev, cmd, next)` tuples pushed from `GameBootstrap.TryApply`/`OnNetApply`; pure helpers (`HexPath`, `FogPresentation`) compute paths and visibility spans. Engine state always commits immediately — only visuals lag.

**Tech Stack:** Unity 6 (URP, WebGL target), C# presentation layer in `Assets/HexWars/Presentation/`, engine consumed as a prebuilt DLL. Unity is driven via the Coplay MCP (`check_compile_errors`, `execute_script`, `capture_scene_object`); scene is `Assets/Scenes/HexWars.unity`.

**Spec:** `docs/superpowers/specs/2026-07-05-opponent-turn-presentation-design.md`

## Global Constraints

- **Zero engine changes.** Do not touch `engine/` — no DLL resync, no wire, no Python changes.
- **WebGL-safe rendering:** no new shader keywords or transparent-material variants (WebGL strips them — see `IconMaterial`'s comment in BoardRenderer). Fog appear/disappear uses **scale-based** pop, not alpha fade.
- **Do not touch `Time.timeScale`** — `PauseToggle.cs` owns it. Hit-stop is a coroutine-local hold.
- **Procedural only:** no imported assets; all visuals/SFX generated in code (project convention).
- **Sounds must not leak fog intel:** a hidden action plays no sound and takes zero playback time.
- **Git:** commit after each task. **Never add attribution trailers** (no Co-Authored-By, no tool credits).
- Timings (from spec): move tween **0.3 s/hop**, opponent inter-action gap **0.25 s**, impact hold **0.05 s**, camera nudge only when the action is off-screen, kill shake only (no shake on ordinary hits).
- Verification is Coplay-driven: after every code change run `mcp__coplay-mcp__check_compile_errors`; behavioral checks use `mcp__coplay-mcp__execute_script` (editor C#, compiled against current assemblies — a snippet referencing a not-yet-written class failing to compile IS the red step). Play-mode traps (domain reload, screenshots) are documented in the memory `hexwars-playmode-verification`.

## File Map

| File | Change | Responsibility |
|---|---|---|
| `Assets/HexWars/Presentation/HexPath.cs` | Create | Hex-line between two coords (cube lerp + round) |
| `Assets/HexWars/Presentation/FogPresentation.cs` | Create | Visible span of a path; tracer origin for unseen attackers |
| `Assets/HexWars/Presentation/TokenStore.cs` | Create | Persistent unit/generator tokens; diff-based `Sync` |
| `Assets/HexWars/Presentation/ActionPresenter.cs` | Create | The single animation queue; per-command playback; sounds |
| `Assets/HexWars/Presentation/BoardRenderer.cs` | Modify | Loses entity builders; keeps tiles/tint/materials; `RenderEntities` becomes a facade |
| `Assets/HexWars/Presentation/GameBootstrap.cs` | Modify | Routes applies through the presenter; `FogViewerFor(state)`; queue reset on teardown; `WaitingHumanSeat()` |
| `Assets/HexWars/Presentation/UnitInputController.cs` | Modify | Deletes `MoveSeq`/`AttackSeq`/projectile; fast-forward before issuing; reason toasts for refused clicks |
| `Assets/HexWars/Presentation/CameraRig.cs` | Modify | `NudgeToward` (cancel on input) + `Shake` |
| `Assets/HexWars/Presentation/AiOpponent.cs` | Modify | Waits for presenter idle between actions |
| `Assets/HexWars/Presentation/GameHud.cs` | Modify | Game-over banner waits for the presenter to drain |

Engine APIs consumed (read-only): `TargetingService.IsVisibleToArmy(state, army, cell, elevation)` (+ `InRange`/`HasShot`, the other two predicates `CanTarget` ANDs — used by Task 8's reason decomposition), `MovementService`, `LineOfSight.IsClear(board, from, fromElev, to, toElev)`, `HexLayout.ToWorld(cell, hexSize)`, `HexCoord` (`.Q/.R/.S`, `Distance`, `Neighbors`), command records in `Command.cs` (`MoveUnit(Issuer, UnitId, Dest)`, `AttackUnit(Issuer, AttackerId, TargetId)`, `DeployUnit(Issuer, TemplateIndex, Cell)`, `CaptureHex`, `BuildGenerator`, `DeployGenerator`, `CreateUnit`, `EndTurn`), `Generator` (`Id`, `Cell`, `Elevation`, `IsAlive`), `Unit` (`Id`, `Owner`, `Cell`, `Elevation`, `CurrentHp`, `IsAlive`, `Stats`), `GameState` (`Players`, `Player(pid)`, `ActivePlayer`, `Board`, `Config`, `MovedUnitIds`, `AttackedUnitIds`, `IsGameOver`).

---

### Task 1: HexPath + FogPresentation (pure helpers)

**Files:**
- Create: `Assets/HexWars/Presentation/HexPath.cs`
- Create: `Assets/HexWars/Presentation/FogPresentation.cs`

**Interfaces:**
- Consumes: engine only.
- Produces:
  - `static List<HexCoord> HexPath.Line(HexCoord a, HexCoord b)` — inclusive of both endpoints, length `Distance(a,b)+1`.
  - `static (int First, int Last) FogPresentation.VisibleSpan(GameState state, PlayerId? viewer, IReadOnlyList<HexCoord> path)` — inclusive index span of path cells visible to `viewer`; `(-1,-1)` if none; full span when `viewer == null` or fog is off in `state.Config`.
  - `static HexCoord? FogPresentation.TracerOrigin(GameState state, PlayerId? viewer, HexCoord from, HexCoord to)` — first cell along `Line(from,to)` visible to the viewer (walking from `from` toward `to`); `null` only if no cell on the line is visible. Returns `from` itself when the attacker is visible / fog off / no viewer.

- [ ] **Step 1: Write the failing check**

Run `mcp__coplay-mcp__execute_script` with this body. Expected: **compilation error** (`HexPath` / `FogPresentation` do not exist) — that is the red step.

```csharp
using HexWars.Engine;
using HexWars.Presentation;
using System.Linq;
var log = new System.Text.StringBuilder();
void Check(string name, bool ok) => log.AppendLine((ok ? "PASS " : "FAIL ") + name);

// Line: straight q-axis run
var line = HexPath.Line(new HexCoord(0, 0), new HexCoord(3, 0));
Check("line length", line.Count == 4);
Check("line endpoints", line[0] == new HexCoord(0, 0) && line[3] == new HexCoord(3, 0));
Check("line adjacency", Enumerable.Range(0, 3).All(i => HexCoord.Distance(line[i], line[i + 1]) == 1));
Check("degenerate line", HexPath.Line(new HexCoord(2, 2), new HexCoord(2, 2)).Count == 1);

// A tiny real game state for visibility checks: default config with fog ON
var cfg = GameConfig.Default(fogOfWar: true);
var board = new RandomBoardGenerator(new BoardGenConfig(9, 7, 0, 2, 1.0f, 100, 0, 0, 0)).Generate(3); // all flat plains
// one P0 unit with vision 2 at (1,1); P1 has a unit far away at (7,5)
var watcher = new Unit(1, PlayerId.Player0, new UnitStats(health: 3, damage: 1, defense: 0, movement: 2, verticalMovement: 1, range: 1, rangeArc: 1, vision: 2, visionArc: 1), new HexCoord(1, 1), 0);
var farUnit = new Unit(2, PlayerId.Player1, new UnitStats(health: 3, damage: 1, defense: 0, movement: 2, verticalMovement: 1, range: 1, rangeArc: 1, vision: 2, visionArc: 1), new HexCoord(7, 5), 0);
var p0 = new PlayerState(PlayerId.Player0, 0, unitsOnBoard: new[] { watcher });
var p1 = new PlayerState(PlayerId.Player1, 0, unitsOnBoard: new[] { farUnit });
var st = new GameState(board, cfg, new[] { p0, p1 }, PlayerId.Player0, 1, 3);

// span: a path passing next to the watcher is visible near it, not far away
var path = HexPath.Line(new HexCoord(1, 2), new HexCoord(6, 2));
var span = FogPresentation.VisibleSpan(st, PlayerId.Player0, path);
Check("span starts visible", span.First == 0);
Check("span ends before the far end", span.Last >= span.First && span.Last < path.Count - 1);
Check("null viewer = full span", FogPresentation.VisibleSpan(st, null, path) == (0, path.Count - 1));
var farPath = HexPath.Line(new HexCoord(7, 4), new HexCoord(7, 5));
Check("hidden path = (-1,-1)", FogPresentation.VisibleSpan(st, PlayerId.Player0, farPath) == (-1, -1));

// tracer: shot from far (7,5) at the watcher (1,1) must clamp its origin to a visible cell
var origin = FogPresentation.TracerOrigin(st, PlayerId.Player0, new HexCoord(7, 5), new HexCoord(1, 1));
Check("tracer origin exists", origin.HasValue);
Check("tracer origin visible", origin.HasValue && TargetingService.IsVisibleToArmy(st, PlayerId.Player0, origin.Value, st.Board.TileAt(origin.Value).Elevation));
Check("tracer origin not the muzzle", origin.HasValue && origin.Value != new HexCoord(7, 5));
Check("visible attacker keeps muzzle", FogPresentation.TracerOrigin(st, null, new HexCoord(7, 5), new HexCoord(1, 1)) == new HexCoord(7, 5));

Debug.Log(log.ToString());
return log.ToString().Contains("FAIL") ? "FAIL" : "ALL PASS";
```

*(If `GameState`/`PlayerState`/`UnitStats` constructor arities differ from `GameBootstrap.NewGame`'s usage at `GameBootstrap.cs:97-119`, mirror that file — it is the authoritative construction example.)*

- [ ] **Step 2: Create `Assets/HexWars/Presentation/HexPath.cs`**

```csharp
using System.Collections.Generic;
using HexWars.Engine;

namespace HexWars.Presentation
{
    /// <summary>The straight hex-line between two cells (cube lerp + round), endpoints inclusive.
    /// Presentation-only: used to draw movement tweens and attack tracers. The engine's own
    /// line walk (LineOfSight.LerpRound) is private, so this mirrors it rather than changing
    /// the engine.</summary>
    public static class HexPath
    {
        public static List<HexCoord> Line(HexCoord a, HexCoord b)
        {
            int n = HexCoord.Distance(a, b);
            var cells = new List<HexCoord>(n + 1) { a };
            for (int i = 1; i <= n; i++)
            {
                var c = LerpRound(a, b, (float)i / n);
                if (c != cells[cells.Count - 1]) cells.Add(c);
            }
            if (cells[cells.Count - 1] != b) cells.Add(b);
            return cells;
        }

        static HexCoord LerpRound(HexCoord a, HexCoord b, float t)
        {
            float q = a.Q + (b.Q - a.Q) * t;
            float r = a.R + (b.R - a.R) * t;
            float s = a.S + (b.S - a.S) * t;
            int rq = (int)System.Math.Round(q);
            int rr = (int)System.Math.Round(r);
            int rs = (int)System.Math.Round(s);
            double dq = System.Math.Abs(rq - q), dr = System.Math.Abs(rr - r), ds = System.Math.Abs(rs - s);
            if (dq > dr && dq > ds) rq = -rr - rs;
            else if (dr > ds) rr = -rq - rs;
            else rs = -rq - rr;
            return new HexCoord(rq, rr);
        }
    }
}
```

- [ ] **Step 3: Create `Assets/HexWars/Presentation/FogPresentation.cs`**

```csharp
using System.Collections.Generic;
using HexWars.Engine;

namespace HexWars.Presentation
{
    /// <summary>
    /// What the viewer is entitled to see of an action under fog. Pure functions over engine
    /// state — the presenter asks these before animating so the presentation never leaks more
    /// intel than the game state does. Cells are judged at their tile's ground elevation
    /// (ground-first M1: a unit's elevation is the tile it stands on).
    /// </summary>
    public static class FogPresentation
    {
        static bool Fogged(GameState state, PlayerId? viewer) => viewer.HasValue && state.Config.FogOfWar;

        static bool CellVisible(GameState state, PlayerId viewer, HexCoord cell) =>
            state.Board.Contains(cell)
            && TargetingService.IsVisibleToArmy(state, viewer, cell, state.Board.TileAt(cell).Elevation);

        /// <summary>Inclusive index span of <paramref name="path"/> cells the viewer can see;
        /// (-1,-1) when none. Full span when there is no viewer or fog is off.</summary>
        public static (int First, int Last) VisibleSpan(GameState state, PlayerId? viewer, IReadOnlyList<HexCoord> path)
        {
            if (!Fogged(state, viewer)) return (0, path.Count - 1);
            int first = -1, last = -1;
            for (int i = 0; i < path.Count; i++)
                if (CellVisible(state, viewer.Value, path[i]))
                {
                    if (first < 0) first = i;
                    last = i;
                }
            return (first, last);
        }

        /// <summary>Where a shot from <paramref name="from"/> toward <paramref name="to"/> may
        /// visually originate: the true muzzle when visible, else the first visible cell along
        /// the line (the fog boundary — real bearing, clamped origin). Null if the whole line is
        /// dark.</summary>
        public static HexCoord? TracerOrigin(GameState state, PlayerId? viewer, HexCoord from, HexCoord to)
        {
            if (!Fogged(state, viewer)) return from;
            foreach (var cell in HexPath.Line(from, to))
                if (CellVisible(state, viewer.Value, cell))
                    return cell;
            return null;
        }
    }
}
```

- [ ] **Step 4: Compile + run the check**

Run `mcp__coplay-mcp__check_compile_errors` → expect none. Re-run the Step 1 script → expected: `ALL PASS` (every `Check` line prints `PASS`).

- [ ] **Step 5: Commit**

```powershell
git add Assets/HexWars/Presentation/HexPath.cs Assets/HexWars/Presentation/FogPresentation.cs
git commit -m "feat(fx): hex-line + fog visibility helpers - pure functions the presenter uses to decide what an action may show under fog"
```

---

### Task 2: TokenStore + BoardRenderer facade (visual parity, still no animation)

**Files:**
- Create: `Assets/HexWars/Presentation/TokenStore.cs`
- Modify: `Assets/HexWars/Presentation/BoardRenderer.cs` (entity builders at lines 69-320 move/shrink; `Render` at :59; `RenderEntities` at :74)

**Interfaces:**
- Consumes: `BoardRenderer` internals exposed below.
- Produces (used by Tasks 3-6):
  - `TokenStore` (MonoBehaviour, same GameObject as BoardRenderer):
    - `public GameObject UnitToken(int unitId)` — null if absent.
    - `public void Sync(GameState state, PlayerId? viewer)` — diff to truth: spawn/despawn/snap-position/refresh materials, HP bars, `UnitView.Unit`.
    - `public void Clear()` — destroy everything (new game).
    - `public Vector3 CellTop(HexCoord cell, int elevation)` — world position of a column top.
  - `BoardRenderer`:
    - `internal void EnsureMaterials()` (was private)
    - `internal Material UnitMat(PlayerId owner, bool dim)`
    - `internal Material IconMatFor(UnitRole role)` (renamed wrapper of `IconMaterial`)
    - `internal Material BlackMat { get; }`
    - `internal Material UnlitColorMat(Color c)` (wrapper of `UnlitColor`)
    - `public void UpdateControlTint(GameState state)` (extracted from `RenderEntities` lines 106-118)
    - `public void RenderEntities(GameState state, PlayerId? viewer = null)` — **kept as a facade**: `EnsureMaterials(); tokenStore.Sync(state, viewer); UpdateControlTint(state);` so ReplayPlayer (`ReplayPlayer.cs:75`), ModelDuelDriver (`:64,:103`), and all GameBootstrap call sites keep working unchanged.

- [ ] **Step 1: Create `Assets/HexWars/Presentation/TokenStore.cs`**

The token-building code is BoardRenderer's `BuildToken`/`BuildHpBar`/`MakeBarQuad`/`BuildPylon`/`AddHull` (lines 129-320) reparented: HP bar becomes a **child of the token** so it travels with tweens; everything else is the same visuals.

```csharp
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
using HexWars.Engine;

namespace HexWars.Presentation
{
    /// <summary>
    /// Persistent tokens for units and generators, keyed by engine id. Sync() diffs the given
    /// state against what exists: spawns the missing, destroys the gone, snaps survivors to
    /// truth (position, HP bar, spent-dim material, UnitView payload). Replaces BoardRenderer's
    /// destroy-and-rebuild so the ActionPresenter can tween tokens between states.
    /// </summary>
    [RequireComponent(typeof(BoardRenderer))]
    public sealed class TokenStore : MonoBehaviour
    {
        BoardRenderer _board;
        Transform _root;
        readonly Dictionary<int, GameObject> _units = new Dictionary<int, GameObject>();
        readonly Dictionary<int, GameObject> _generators = new Dictionary<int, GameObject>();

        void Awake() => _board = GetComponent<BoardRenderer>();

        Transform Root()
        {
            if (_root == null)
            {
                var go = new GameObject("Tokens");
                go.transform.SetParent(transform, false);
                _root = go.transform;
            }
            return _root;
        }

        public GameObject UnitToken(int unitId) => _units.TryGetValue(unitId, out var go) && go != null ? go : null;

        public Vector3 CellTop(HexCoord cell, int elevation)
        {
            var w = HexLayout.ToWorld(cell, _board.HexSize);
            return new Vector3((float)w.x, (elevation + 1) * _board.LevelHeight, (float)w.z);
        }

        public void Clear()
        {
            foreach (var go in _units.Values) if (go != null) Destroy(go);
            foreach (var go in _generators.Values) if (go != null) Destroy(go);
            _units.Clear();
            _generators.Clear();
        }

        /// <summary>Snap the world to <paramref name="state"/>. With fog and a viewer, the other
        /// army exists only where the viewer's army has vision (same rule as targeting).</summary>
        public void Sync(GameState state, PlayerId? viewer)
        {
            _board.EnsureMaterials();
            bool fog = viewer.HasValue && state.Config.FogOfWar;
            var liveUnits = new HashSet<int>();
            var liveGens = new HashSet<int>();

            foreach (var player in state.Players)
            {
                bool isActive = player.Id == state.ActivePlayer;
                bool hideUnseen = fog && player.Id != viewer.Value;

                foreach (var u in player.UnitsOnBoard)
                {
                    if (!u.IsAlive) continue;
                    if (hideUnseen && !TargetingService.IsVisibleToArmy(state, viewer.Value, u.Cell, u.Elevation))
                        continue;
                    liveUnits.Add(u.Id);
                    bool spent = isActive && Contains(state.MovedUnitIds, u.Id) && Contains(state.AttackedUnitIds, u.Id);
                    var mat = _board.UnitMat(player.Id, !isActive || spent);
                    if (!_units.TryGetValue(u.Id, out var token) || token == null)
                        _units[u.Id] = token = BuildToken(u);
                    RefreshToken(token, u, mat);
                }

                foreach (var g in player.Generators)
                {
                    if (!g.IsAlive) continue;
                    if (hideUnseen && !TargetingService.IsVisibleToArmy(state, viewer.Value, g.Cell, g.Elevation))
                        continue;
                    liveGens.Add(g.Id);
                    if (!_generators.TryGetValue(g.Id, out var pylon) || pylon == null)
                        _generators[g.Id] = pylon = BuildPylon(g);
                    pylon.GetComponent<MeshRenderer>().sharedMaterial = _board.UnitMat(player.Id, !isActive);
                }
            }

            Prune(_units, liveUnits);
            Prune(_generators, liveGens);
        }

        static void Prune(Dictionary<int, GameObject> map, HashSet<int> live)
        {
            List<int> dead = null;
            foreach (var kv in map)
                if (!live.Contains(kv.Key))
                {
                    (dead = dead ?? new List<int>()).Add(kv.Key);
                    if (kv.Value != null) Destroy(kv.Value);
                }
            if (dead != null) foreach (var id in dead) map.Remove(id);
        }

        static bool Contains(IReadOnlyCollection<int> ids, int id)
        {
            foreach (var i in ids) if (i == id) return true;
            return false;
        }

        // ---- construction (visuals identical to the old BoardRenderer builders) ----

        GameObject BuildToken(Unit unit)
        {
            float sizeFactor = Mathf.Clamp(0.6f + unit.Stats.PointCost * 0.04f, 0.6f, 1.4f);
            float radius = _board.HexSize * 0.7f * sizeFactor;

            var token = new GameObject("Unit_" + unit.Id);
            token.transform.SetParent(Root(), false);
            token.AddComponent<UnitView>();
            var box = token.AddComponent<BoxCollider>();
            box.center = new Vector3(0f, 0.35f, 0f);
            box.size = new Vector3(_board.HexSize * 1.3f, 0.9f, _board.HexSize * 1.3f);

            var disc = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            disc.name = "Disc";
            DestroyImmediate(disc.GetComponent<Collider>());
            disc.transform.SetParent(token.transform, false);
            disc.transform.localPosition = new Vector3(0f, 0.18f, 0f);
            disc.transform.localScale = new Vector3(radius, 0.16f, radius);
            AddHull(disc, 1.16f, 1.05f);

            var icon = GameObject.CreatePrimitive(PrimitiveType.Quad);
            icon.name = "RoleIcon";
            DestroyImmediate(icon.GetComponent<Collider>());
            icon.transform.SetParent(token.transform, false);
            icon.transform.localPosition = new Vector3(0f, 0.345f, 0f);
            icon.transform.localRotation = Quaternion.Euler(90f, 0f, 0f);
            icon.transform.localScale = Vector3.one * (radius * 0.9f);
            var mr = icon.GetComponent<MeshRenderer>();
            mr.sharedMaterial = _board.IconMatFor(Roles.Dominant(unit.Stats));
            mr.shadowCastingMode = ShadowCastingMode.Off;

            var bar = new GameObject("HpBar");
            bar.transform.SetParent(token.transform, false);
            bar.transform.localPosition = new Vector3(0f, 0.62f, 0f);
            bar.AddComponent<Billboard>();
            return token;
        }

        void RefreshToken(GameObject token, Unit unit, Material discMat)
        {
            token.transform.localPosition = CellTop(unit.Cell, unit.Elevation);
            token.transform.localScale = Vector3.one; // a fast-forward can kill a squash/pop tween mid-scale
            token.GetComponent<UnitView>().Unit = unit; // engine states are immutable: re-point every sync
            token.transform.Find("Disc").GetComponent<MeshRenderer>().sharedMaterial = discMat;
            RefreshHpBar(token.transform.Find("HpBar"), unit.CurrentHp, unit.Stats.Health);
        }

        void RefreshHpBar(Transform bar, int cur, int max)
        {
            for (int i = bar.childCount - 1; i >= 0; i--) Destroy(bar.GetChild(i).gameObject);
            float frac = max <= 0 ? 0f : Mathf.Clamp01((float)cur / max);
            float barW = _board.HexSize * 0.85f;
            MakeBarQuad(bar, 0f, 0f, barW, 0.16f, new Color(0.18f, 0.03f, 0.03f));
            float fw = Mathf.Max(0.001f, barW * frac);
            float fx = -barW * 0.5f + fw * 0.5f;
            var fill = Color.Lerp(new Color(0.85f, 0.2f, 0.12f), new Color(0.25f, 0.85f, 0.25f), frac);
            MakeBarQuad(bar, fx, -0.01f, fw, 0.11f, fill);
        }

        void MakeBarQuad(Transform parent, float x, float zTowardCam, float w, float h, Color c)
        {
            var q = GameObject.CreatePrimitive(PrimitiveType.Quad);
            q.name = "Bar";
            DestroyImmediate(q.GetComponent<Collider>());
            q.transform.SetParent(parent, false);
            q.transform.localPosition = new Vector3(x, 0f, zTowardCam);
            q.transform.localScale = new Vector3(w, h, 1f);
            var mr = q.GetComponent<MeshRenderer>();
            mr.sharedMaterial = _board.UnlitColorMat(c);
            mr.shadowCastingMode = ShadowCastingMode.Off;
        }

        GameObject BuildPylon(Generator g)
        {
            var pylon = GameObject.CreatePrimitive(PrimitiveType.Cube);
            pylon.name = "Generator_" + g.Id;
            DestroyImmediate(pylon.GetComponent<Collider>());
            pylon.transform.SetParent(Root(), false);
            var top = CellTop(g.Cell, g.Elevation);
            pylon.transform.localPosition = new Vector3(top.x, top.y + 0.45f, top.z);
            pylon.transform.localScale = new Vector3(_board.HexSize * 0.45f, 0.9f, _board.HexSize * 0.45f);
            AddHull(pylon, 1.12f, 1.06f);
            return pylon;
        }

        void AddHull(GameObject host, float xz, float y)
        {
            var hull = new GameObject("Outline");
            hull.transform.SetParent(host.transform, false);
            hull.transform.localScale = new Vector3(xz, y, xz);
            hull.AddComponent<MeshFilter>().sharedMesh = host.GetComponent<MeshFilter>().sharedMesh;
            var mr = hull.AddComponent<MeshRenderer>();
            var m = new Material(_board.BlackMat);
            if (m.HasProperty("_Cull")) m.SetFloat("_Cull", 1f);
            mr.sharedMaterial = m;
            mr.shadowCastingMode = ShadowCastingMode.Off;
        }
    }
}
```

*(Note the deliberate change: HP bar hangs under the token root and travels with it; the old code parented bars to the entities root at absolute positions.)*

- [ ] **Step 2: Refactor `BoardRenderer.cs`**

1. Make `EnsureMaterials()` `internal` (was private, line 394).
2. Add accessors next to `EnsureMaterials`:

```csharp
internal Material UnitMat(PlayerId owner, bool dim) =>
    owner == PlayerId.Player0 ? (dim ? _p0Dim : _p0) : (dim ? _p1Dim : _p1);
internal Material IconMatFor(UnitRole role) => IconMaterial(role);
internal Material BlackMat => _black;
internal Material UnlitColorMat(Color c) => UnlitColor(c);
```

3. Replace the body of `RenderEntities` (lines 74-119) with the facade + extracted tint:

```csharp
/// <summary>Snap all entities to <paramref name="state"/> (facade over TokenStore; kept so
/// replay/duel/bootstrap call sites are unchanged). With fog and a viewer, the other army's
/// entities exist only where the viewer's army has vision.</summary>
public void RenderEntities(GameState state, PlayerId? viewer = null)
{
    EnsureMaterials();
    var store = GetComponent<TokenStore>();
    if (store == null) store = gameObject.AddComponent<TokenStore>();
    store.Sync(state, viewer);
    UpdateControlTint(state);
}

/// <summary>Tint controlled hexes toward their owner's colour (in-place material swap on the
/// tile fill; rides every entity sync and each presenter action commit).</summary>
public void UpdateControlTint(GameState state)
{
    var cols = transform.Find("Columns");
    if (cols == null) return;
    foreach (Transform col in cols)
    {
        var tv = col.GetComponent<TileView>();
        var fillT = col.Find("Fill");
        if (tv == null || fillT == null) continue;
        var terrain = state.Board.TileAt(tv.Coord).Terrain;
        var owner = state.Board.Controller(tv.Coord);
        fillT.GetComponent<MeshRenderer>().sharedMaterial =
            owner == null ? MaterialFor(terrain) : ControlTintMaterial(terrain, owner.Value);
    }
}
```

4. In `Render(Board board)` (line 59), clear stale tokens when a new board is built — add before `ClearChild("Columns")`:

```csharp
var store = GetComponent<TokenStore>();
if (store != null) store.Clear();
```

5. Delete from BoardRenderer: `BuildToken`, `BuildHpBar`, `MakeBarQuad`, `BuildPylon`, `AddHull`, `Contains`, `TopY` (only token code used it), and the dead `BuildControlCap` + `TransparentColor` (unreferenced once caps are gone — verify with a grep before deleting). Keep `UnlitColor`, `IconMaterial`, `ControlTintMaterial`, `MaterialFor`, all `EnsureMaterials` content, and the `ClearChild("Entities")`-related code is gone with the old `RenderEntities` body (tokens now live under "Tokens", managed by the store).

- [ ] **Step 3: Compile**

`mcp__coplay-mcp__check_compile_errors` → none. One expected knock-on: `CombatFx.Top` (CombatFx.cs:78-82) uses `board.HexSize/LevelHeight` directly — untouched, still compiles.

- [ ] **Step 4: Play-mode parity check**

Enter play mode (vs-AI off, hotseat, DemoPieces on) via `mcp__coplay-mcp__play_game` or the flow in memory `hexwars-playmode-verification`. Then `mcp__coplay-mcp__execute_script`:

```csharp
using HexWars.Engine;
using HexWars.Presentation;
var game = Object.FindAnyObjectByType<GameBootstrap>();
var store = Object.FindAnyObjectByType<TokenStore>();
var log = new System.Text.StringBuilder();
void Check(string n, bool ok) => log.AppendLine((ok ? "PASS " : "FAIL ") + n);

int units = 0;
foreach (var p in game.State.Players) foreach (var u in p.UnitsOnBoard) if (u.IsAlive) units++;
Check("token per unit", Object.FindObjectsByType<UnitView>(FindObjectsSortMode.None).Length == units);

// persistence: the same GameObject must survive a state change
var p0u = game.State.Player(PlayerId.Player0).UnitsOnBoard[0];
var before = store.UnitToken(p0u.Id);
Check("token exists", before != null);
HexCoord? dest = null;
foreach (var c in MovementService.ReachableTiles(game.State, p0u)) { dest = c; break; }
Check("has a legal move", dest.HasValue);
game.TryApply(new MoveUnit(PlayerId.Player0, p0u.Id, dest.Value));
var after = store.UnitToken(p0u.Id);
Check("token persisted (same instance)", ReferenceEquals(before, after));
Check("UnitView repointed", after.GetComponent<UnitView>().Unit.Cell == dest.Value);
Check("token at destination", (after.transform.localPosition - store.CellTop(dest.Value, game.State.Board.TileAt(dest.Value).Elevation)).magnitude < 0.01f);
Debug.Log(log.ToString());
return log.ToString().Contains("FAIL") ? "FAIL" : "ALL PASS";
```

Expected: `ALL PASS`. Also `mcp__coplay-mcp__capture_scene_object` of the board — visually identical to before (discs, icons, HP bars, pylons, dimming, territory tint if territory mode).

- [ ] **Step 5: Manual regression sweep (same play session)**

Click a unit → selection marker floats above it; hover → docked stats show; move via click (still animates via the old `MoveSeq` — that's expected until Task 4); attack an enemy → projectile + popup. End turn → dim/bright flips.

- [ ] **Step 6: Commit**

```powershell
git add Assets/HexWars/Presentation/TokenStore.cs Assets/HexWars/Presentation/BoardRenderer.cs
git commit -m "refactor(render): persistent unit/generator tokens (TokenStore) behind the old RenderEntities facade - tokens now survive state changes so they can be tweened; HP bars ride their token"
```

---

### Task 3: ActionPresenter core + GameBootstrap rewiring (moves animate for everyone)

**Files:**
- Create: `Assets/HexWars/Presentation/ActionPresenter.cs`
- Modify: `Assets/HexWars/Presentation/GameBootstrap.cs` (`TryApply` :134-159, `OnNetApply` :246-257, `NewGame` :91, `StartLocalGame` :171, `OnNetStart` :233, `ReturnToMenu` :199, `FogViewer` :217, delete `PlaySounds`/`LiveUnits` :288-314)
- Modify: `Assets/HexWars/Presentation/AiOpponent.cs` (`Update` :37-52)

**Interfaces:**
- Consumes: `TokenStore.UnitToken/Sync/CellTop/Clear`, `BoardRenderer.UpdateControlTint`, `HexPath.Line`, `CombatFx.Report`, `SoundManager`, `ExplosionFx.Spawn(Vector3, Color, float, bool)`.
- Produces (used by Tasks 4-7):
  - `ActionPresenter` (MonoBehaviour, added to GameBootstrap's GameObject):
    - `public bool IsBusy { get; }`
    - `public void Enqueue(GameState prev, Command cmd, GameState next, bool isLocal)`
    - `public void FastForward()` — commit everything queued instantly (used by input in Task 4)
    - `public void ResetQueue()` — teardown for new game / menu
    - `public const float SecondsPerHop = 0.3f; public const float OpponentGap = 0.25f;`
  - `GameBootstrap`:
    - `public ActionPresenter Presenter { get; private set; }`
    - `public PlayerId? FogViewerFor(GameState s)` — `FogViewer()` becomes `FogViewerFor(State)`.

- [ ] **Step 1: Create `Assets/HexWars/Presentation/ActionPresenter.cs`**

Move animation only in this task; attack/deploy/etc. arrive in Tasks 4-5 through the same switch (until then they play as "commit + sound").

```csharp
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using HexWars.Engine;

namespace HexWars.Presentation
{
    /// <summary>
    /// The single animation pipeline. Every applied command — local click, AI, server echo,
    /// spectator/replay driver — is enqueued as (prev, cmd, next) and played in order; the
    /// engine state has ALWAYS already committed, only visuals lag. After each action the
    /// TokenStore syncs to `next`, so an interrupted or fast-forwarded animation still lands
    /// the board on the truth.
    /// </summary>
    public sealed class ActionPresenter : MonoBehaviour
    {
        public const float SecondsPerHop = 0.3f;   // spec: move tween per hop
        public const float OpponentGap = 0.25f;    // spec: pacing between opponent actions

        struct Item
        {
            public GameState Prev, Next;
            public Command Cmd;
            public bool IsLocal;
        }

        readonly Queue<Item> _queue = new Queue<Item>();
        bool _playing;
        GameBootstrap _game;
        BoardRenderer _board;
        TokenStore _tokens;

        public bool IsBusy => _playing || _queue.Count > 0;

        void Awake()
        {
            _game = GetComponent<GameBootstrap>();
            _board = GetComponent<BoardRenderer>();
        }

        TokenStore Tokens() => _tokens != null ? _tokens : (_tokens = _board.GetComponent<TokenStore>());

        public void Enqueue(GameState prev, Command cmd, GameState next, bool isLocal)
        {
            _queue.Enqueue(new Item { Prev = prev, Cmd = cmd, Next = next, IsLocal = isLocal });
            if (!_playing) StartCoroutine(DrainQueue());
        }

        Item? _current;          // the item whose animation is mid-flight
        GameObject _projectile;  // live transient (attack tracer), destroyed on fast-forward
        bool _reported;          // did the mid-flight item already fire its CombatFx popups?

        /// <summary>Snap-commit the mid-flight item and everything still queued — synchronously,
        /// this frame. Called before local input issues a command so truth and visuals can't
        /// cross. Deliberately NOT a flag the coroutine polls: a flag that clears when the queue
        /// empties would also snap-commit the local action enqueued right after this call.</summary>
        public void FastForward()
        {
            if (!IsBusy) return;
            StopAllCoroutines();
            if (_projectile != null) { Destroy(_projectile); _projectile = null; }
            if (_current.HasValue) { Commit(_current.Value, skipCombatFx: _reported); _current = null; }
            while (_queue.Count > 0) Commit(_queue.Dequeue());
            _playing = false;
        }

        public void ResetQueue()
        {
            StopAllCoroutines();
            if (_projectile != null) { Destroy(_projectile); _projectile = null; }
            _queue.Clear();
            _current = null;
            _playing = false;
        }

        IEnumerator DrainQueue()
        {
            _playing = true;
            while (_queue.Count > 0)
            {
                _current = _queue.Dequeue();
                _reported = false;
                yield return Play(_current.Value);
                Commit(_current.Value, skipCombatFx: _reported);
                bool wasLocal = _current.Value.IsLocal;
                _current = null;
                if (!wasLocal && _queue.Count > 0)
                    yield return new WaitForSeconds(OpponentGap);
            }
            _playing = false;
        }

        IEnumerator Play(Item item)
        {
            var viewer = _game.FogViewerFor(item.Next);
            switch (item.Cmd)
            {
                case MoveUnit mv: yield return PlayMove(item, mv, viewer); break;
                // AttackUnit lands in Task 4; Deploy/Capture/Build/EndTurn in Task 5.
                default: PlayInstantSound(item); break;
            }
        }

        IEnumerator PlayMove(Item item, MoveUnit mv, PlayerId? viewer)
        {
            Unit before = FindUnit(item.Prev, mv.Issuer, mv.UnitId);
            var token = Tokens().UnitToken(mv.UnitId);
            if (before == null || token == null) yield break; // hidden under fog (Task 6 refines) or gone

            SoundManager.Play(SoundKind.Move);
            var path = HexPath.Line(before.Cell, mv.Dest);
            for (int i = 1; i < path.Count; i++)
            {
                Vector3 from = token.transform.localPosition;
                Vector3 to = Tokens().CellTop(path[i], item.Next.Board.TileAt(path[i]).Elevation);
                for (float t = 0f; t < SecondsPerHop; t += Time.deltaTime)
                {
                    token.transform.localPosition = Vector3.Lerp(from, to, Mathf.SmoothStep(0f, 1f, t / SecondsPerHop));
                    yield return null;
                }
            }
        }
        // (No cancellation flags inside the loops: FastForward stops the coroutines outright and
        // Commit's Sync re-snaps position and scale, so an interrupted tween can't strand a token.)

        void PlayInstantSound(Item item)
        {
            switch (item.Cmd)
            {
                case AttackUnit _: SoundManager.Play(SoundKind.Attack); break;
                case CaptureHex _: SoundManager.Play(SoundKind.Claim); break;
                case BuildGenerator _:
                case DeployGenerator _:
                case DeployUnit _:
                case CreateUnit _: SoundManager.Play(SoundKind.Build); break;
                case EndTurn _: SoundManager.Play(SoundKind.EndTurn); break;
            }
        }

        void Commit(Item item, bool skipCombatFx = false)
        {
            Tokens().Sync(item.Next, _game.FogViewerFor(item.Next));
            _board.UpdateControlTint(item.Next);
            if (!skipCombatFx && !(item.Cmd is MoveUnit))
                CombatFx.Report(item.Prev, item.Next, _board, item.Cmd); // popups (attack timing refined in Task 4)
            if (!(item.Cmd is EndTurn) && item.Next.ActivePlayer != item.Prev.ActivePlayer)
                SoundManager.Play(SoundKind.EndTurn); // paced turns auto-pass without an EndTurn command
            if (LiveUnits(item.Next) < LiveUnits(item.Prev)) SoundManager.Play(SoundKind.Death);
            if (item.Next.IsGameOver && !item.Prev.IsGameOver) SoundManager.Play(SoundKind.Win);
        }

        internal static Unit FindUnit(GameState s, PlayerId owner, int id)
        {
            foreach (var u in s.Player(owner).UnitsOnBoard)
                if (u.Id == id && u.IsAlive) return u;
            return null;
        }

        static int LiveUnits(GameState s)
        {
            int n = 0;
            foreach (var p in s.Players)
                foreach (var u in p.UnitsOnBoard)
                    if (u.IsAlive) n++;
            return n;
        }
    }
}
```

- [ ] **Step 2: Rewire `GameBootstrap.cs`**

1. Add the property and wire-up. In the fields near `public event System.Action StateChanged;` (line 68):

```csharp
public ActionPresenter Presenter { get; private set; }
```

In `Start()` (line 70), first statement:

```csharp
Presenter = GetComponent<ActionPresenter>() ?? gameObject.AddComponent<ActionPresenter>();
```

2. `FogViewer` → parameterized (replace lines 217-224):

```csharp
public PlayerId? FogViewer() => FogViewerFor(State);

/// <summary>Whose vision the fog renders for a given state (the presenter passes per-action
/// states while animations lag behind the live State).</summary>
public PlayerId? FogViewerFor(GameState s)
{
    if (s == null || !s.Config.FogOfWar) return null;
    if (Seat.HasValue) return Seat.Value;
    var ai = FindAnyObjectByType<AiOpponent>();
    if (ai != null) return ai.AiSeat == PlayerId.Player0 ? PlayerId.Player1 : PlayerId.Player0;
    return s.ActivePlayer;
}
```

3. `TryApply` local branch — replace lines 151-158:

```csharp
var prev = State;
State = result.NewState;
EventConsole.Report(State, CombatLog.Diff(prev, State, FogViewer()));
Presenter.Enqueue(prev, cmd, State, IsLocalCommand(cmd));
StateChanged?.Invoke();
return true;
```

4. `OnNetApply` — replace lines 250-256 with the same five lines (identical block; the server echo of your own command is "local" via `IsLocalCommand`).

5. Add the helper next to `FogViewerFor`:

```csharp
/// <summary>Local = issued by a seat this human controls: your seat online; any non-AI seat
/// offline (hotseat = both). Local actions play immediately with no pacing gap.</summary>
bool IsLocalCommand(Command cmd)
{
    if (Networked) return Seat.HasValue && cmd.Issuer == Seat.Value;
    var ai = GetComponent<AiOpponent>();
    return ai == null || cmd.Issuer != ai.AiSeat;
}
```

6. Reset the queue on every teardown/build: add `Presenter?.ResetQueue();` as the first line of `NewGame()`, `StartLocalGame(...)`, `OnNetStart(...)`, and `ReturnToMenu()`.

7. Delete `PlaySounds` and `LiveUnits` (lines 286-314) — the presenter owns sounds now. Remove the `PlaySounds(cmd, prev, State);` calls (they were inside the blocks replaced above).

- [ ] **Step 3: Pace the AI off the presenter**

In `AiOpponent.Update` (AiOpponent.cs:37-52), after the `if (!aiTurn) return;` line, add:

```csharp
if (_game.Presenter != null && _game.Presenter.IsBusy) { _timer = 0f; return; } // let the last action finish playing
```

- [ ] **Step 4: Compile + behavior check**

`check_compile_errors` → none. Play mode, hotseat: apply a multi-hex `MoveUnit` via script (as in Task 2 Step 4) and verify the token now **tweens** (sample its position mid-flight):

```csharp
using HexWars.Engine;
using HexWars.Presentation;
var game = Object.FindAnyObjectByType<GameBootstrap>();
var store = Object.FindAnyObjectByType<TokenStore>();
var u = game.State.Player(game.State.ActivePlayer).UnitsOnBoard[0];
HexCoord? far = null; int best = 0;
foreach (var c in MovementService.ReachableTiles(game.State, u))
{ int d = HexCoord.Distance(u.Cell, c); if (d > best) { best = d; far = c; } }
var token = store.UnitToken(u.Id);
var start = token.transform.localPosition;
game.TryApply(new MoveUnit(u.Owner, u.Id, far.Value));
// state must be committed instantly even though the token hasn't arrived:
bool presenterBusy = game.Presenter.IsBusy;
bool tokenNotYetThere = (token.transform.localPosition - store.CellTop(far.Value, game.State.Board.TileAt(far.Value).Elevation)).magnitude > 0.05f;
Debug.Log($"busy={presenterBusy} tokenLagging={tokenNotYetThere}");
return (presenterBusy && tokenNotYetThere) ? "PASS: visuals lag, truth instant" : "FAIL";
```

Expected: `PASS`. Then watch a **vs-AI game** (HexWars > Play vs AI or `VsAI` flag): AI units must slide, never teleport, with a beat between AI actions; HUD/console updates stay instant.

- [ ] **Step 5: Commit**

```powershell
git add Assets/HexWars/Presentation/ActionPresenter.cs Assets/HexWars/Presentation/GameBootstrap.cs Assets/HexWars/Presentation/AiOpponent.cs
git commit -m "feat(fx): ActionPresenter - every applied command (local, AI, server echo) animates through one queue; engine truth commits instantly, visuals lag; AI paces off the queue"
```

---

### Task 4: Attack presentation + one input path

**Files:**
- Modify: `Assets/HexWars/Presentation/ActionPresenter.cs` (add `PlayAttack`, projectile builder, impact hold)
- Modify: `Assets/HexWars/Presentation/UnitInputController.cs` (delete `MoveSeq` :195-210, `AttackSeq` :212-253, `MakeProjectile` :274-298, `HexTopWorld` :255-262, `_animating` :28; rewrite the two `StartCoroutine` call sites :121,:132)
- Modify: `Assets/HexWars/Presentation/CombatFx.cs` (no signature change — only its call timing, already handled; confirm nothing else calls it)

**Interfaces:**
- Consumes: `FogPresentation.TracerOrigin` (wired fully in Task 6 — this task passes `viewer: null` semantics via visible attackers), `LineOfSight.IsClear`, `ExplosionFx.Spawn`, `DamagePopup` (via `CombatFx.Report`).
- Produces: attack playback inside the presenter; input that never overlaps playback (`FastForward()` before issuing).

- [ ] **Step 1: Add attack playback to `ActionPresenter`**

Add the case to `Play`'s switch:

```csharp
case AttackUnit atk: yield return PlayAttack(item, atk, viewer); break;
```

Add the methods (the projectile visuals are `UnitInputController.AttackSeq`/`MakeProjectile` verbatim, retimed to fire the popup at impact):

```csharp
const float ImpactHold = 0.05f; // spec: brief hold at impact; PauseToggle owns timeScale, so hold locally

IEnumerator PlayAttack(Item item, AttackUnit atk, PlayerId? viewer)
{
    var attacker = FindUnit(item.Prev, atk.Issuer, atk.AttackerId);
    if (attacker == null) yield break;
    var targetPos = TargetTop(item.Prev, atk.TargetId);
    if (targetPos == null) yield break;

    int dmg = attacker.Stats.Damage;
    float power = Mathf.Clamp01(dmg / 8f);
    float projScale = Mathf.Lerp(0.14f, 0.5f, power);
    Color projColor = dmg >= 6 ? new Color(1f, 0.3f, 0.1f)
                    : dmg >= 3 ? new Color(1f, 0.65f, 0.2f)
                               : new Color(1f, 0.95f, 0.5f);

    Vector3 from = Tokens().CellTop(attacker.Cell, attacker.Elevation) + Vector3.up * 0.4f;
    Vector3 to = targetPos.Value + Vector3.up * 0.4f;

    bool directLos = LineOfSight.IsClear(item.Prev.Board, attacker.Cell, attacker.Elevation,
                                         TargetCell(item.Prev, atk.TargetId).Value.cell,
                                         TargetCell(item.Prev, atk.TargetId).Value.elev);
    float arc = directLos ? 0f : Mathf.Max(2.5f, Vector3.Distance(from, to) * 0.35f);
    float flightDur = Mathf.Lerp(0.45f, 0.85f, power) + Vector3.Distance(from, to) * 0.035f;

    SoundManager.Play(SoundKind.Attack);
    _projectile = MakeProjectile(from, projScale, projColor);
    for (float t = 0f; t < flightDur; t += Time.deltaTime)
    {
        float f = t / flightDur;
        var pos = Vector3.Lerp(from, to, f);
        pos.y += Mathf.Sin(f * Mathf.PI) * arc;
        _projectile.transform.position = pos;
        yield return null;
    }
    Destroy(_projectile);
    _projectile = null;

    // impact: explosion, popup, and a short hold — the popup lands WITH the hit, not before.
    // _reported tells FastForward/Commit the popups already fired (no doubles, no drops).
    ExplosionFx.Spawn(to, projColor, Mathf.Lerp(0.4f, 0.9f, power), false);
    CombatFx.Report(item.Prev, item.Next, _board, item.Cmd);
    _reported = true;
    yield return new WaitForSeconds(ImpactHold);
}

/// <summary>World top of the attacked entity (unit or generator) in a state; null if gone.</summary>
Vector3? TargetTop(GameState s, int targetId)
{
    var t = TargetCell(s, targetId);
    return t == null ? (Vector3?)null : Tokens().CellTop(t.Value.cell, t.Value.elev);
}

(HexCoord cell, int elev)? TargetCell(GameState s, int targetId)
{
    foreach (var p in s.Players)
    {
        foreach (var u in p.UnitsOnBoard) if (u.IsAlive && u.Id == targetId) return (u.Cell, u.Elevation);
        foreach (var g in p.Generators) if (g.IsAlive && g.Id == targetId) return (g.Cell, g.Elevation);
    }
    return null;
}

GameObject MakeProjectile(Vector3 pos, float scale, Color color)
{
    var p = GameObject.CreatePrimitive(PrimitiveType.Sphere);
    DestroyImmediate(p.GetComponent<Collider>());
    p.transform.position = pos;
    p.transform.localScale = Vector3.one * scale;
    var unlit = Shader.Find("Universal Render Pipeline/Unlit");
    if (unlit == null) unlit = Shader.Find("Unlit/Color");
    var m = new Material(unlit);
    if (m.HasProperty("_BaseColor")) m.SetColor("_BaseColor", color);
    m.color = color;
    var mr = p.GetComponent<MeshRenderer>();
    mr.sharedMaterial = m;
    mr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
    var trail = p.AddComponent<TrailRenderer>();
    trail.time = 0.18f;
    trail.startWidth = scale * 0.9f;
    trail.endWidth = 0f;
    trail.material = m;
    trail.startColor = color;
    trail.endColor = new Color(color.r, color.g, color.b, 0f);
    trail.numCapVertices = 2;
    return p;
}
```

No `Commit` change needed: `_reported`/`skipCombatFx` (Task 3) prevents double popups when the impact already reported, and still fires them from `Commit` when playback bailed early (e.g. the whole tracer was hidden under fog but your unit still took the hit — the popup must appear).

Also remove `case AttackUnit _: SoundManager.Play(SoundKind.Attack); break;` from `PlayInstantSound` (the attack case never reaches it now, but dead branches mislead).

- [ ] **Step 2: One input path in `UnitInputController.cs`**

1. Delete `MoveSeq`, `AttackSeq`, `MakeProjectile`, `HexTopWorld`, and the `_animating` field.
2. `blocked` (line 78) becomes:

```csharp
bool blocked = _barracks != null && _barracks.IsDeploying;
```

3. Attack call site (line 121) — `StartCoroutine(AttackSeq(_selected, unit));` becomes:

```csharp
Issue(new AttackUnit(active, _selected.Unit.Id, unit.Unit.Id));
```

4. Move call site (line 132) — `StartCoroutine(MoveSeq(_selected, tile.Coord));` becomes:

```csharp
Issue(new MoveUnit(active, _selected.Unit.Id, tile.Coord));
```

5. Add the shared helper:

```csharp
/// <summary>Issue a command through the one presentation pipeline: finish any queued playback
/// first (visuals catch up to truth), then apply. The presenter animates the result.</summary>
void Issue(Command cmd)
{
    _game.Presenter?.FastForward();
    _game.TryApply(cmd);
    ReacquireSelection();
    AutoAdvance();
}
```

*(The old snap-back `RenderEntities` call on a failed move disappears with `MoveSeq` — nothing moved optimistically, so there is nothing to snap back.)*

- [ ] **Step 3: Compile + play-mode check**

`check_compile_errors` → none. Play mode, hotseat: click-move (token slides — now via presenter), click-attack (projectile flies, **popup appears at impact**, kill → explosion + death sound). Rapid-fire two commands (click move, then immediately attack with another unit): the first animation snap-completes, the second plays — no overlap, no drift. Online sanity if convenient: your own echoed action animates without the pacing gap.

- [ ] **Step 4: Commit**

```powershell
git add Assets/HexWars/Presentation/ActionPresenter.cs Assets/HexWars/Presentation/UnitInputController.cs
git commit -m "feat(fx): attacks play through the presenter - projectile, impact-timed damage popup, brief impact hold; input fast-forwards the queue so local play never waits"
```

---

### Task 5: Deploy / claim / build / turn-handover beats + game-over gating

**Files:**
- Modify: `Assets/HexWars/Presentation/ActionPresenter.cs` (add `PlayDeploy`, claim/build pulse, EndTurn beat)
- Modify: `Assets/HexWars/Presentation/GameHud.cs` (`ShowGameOver` :192-210)

**Interfaces:**
- Consumes: `TokenStore.UnitToken` (deploy spawns via `Sync` then animates), `BoardRenderer.UpdateControlTint`.
- Produces: complete per-command playback coverage; game-over banner that waits for the queue.

- [ ] **Step 1: Deploy drop-in + claim pulse + handover beat in `ActionPresenter`**

Extend `Play`'s switch:

```csharp
case DeployUnit dep: yield return PlayDeploy(item, dep, viewer); break;
case CaptureHex cap: yield return PlayClaim(item, cap, viewer); break;
case BuildGenerator bld: yield return PlayClaim(item, null, viewer, bld.Cell, SoundKind.Build); break;
case EndTurn _: SoundManager.Play(SoundKind.EndTurn); yield return new WaitForSeconds(item.IsLocal ? 0f : OpponentGap); break;
```

Add:

```csharp
IEnumerator PlayDeploy(Item item, DeployUnit dep, PlayerId? viewer)
{
    // find the newly deployed unit: present in Next, absent in Prev
    Unit fresh = null;
    foreach (var u in item.Next.Player(dep.Issuer).UnitsOnBoard)
        if (u.IsAlive && u.Cell == dep.Cell && FindUnit(item.Prev, dep.Issuer, u.Id) == null) { fresh = u; break; }
    SoundManager.Play(SoundKind.Build);
    if (fresh == null) yield break;

    Tokens().Sync(item.Next, viewer); // spawns the token at its cell
    var token = Tokens().UnitToken(fresh.Id);
    if (token == null) yield break;   // deployed out of the viewer's sight

    // drop-in: fall from above + landing squash
    var rest = token.transform.localPosition;
    const float dur = 0.25f;
    for (float t = 0f; t < dur; t += Time.deltaTime)
    {
        float f = Mathf.SmoothStep(0f, 1f, t / dur);
        token.transform.localPosition = rest + Vector3.up * (1.6f * (1f - f));
        yield return null;
    }
    token.transform.localPosition = rest;
    yield return Squash(token.transform);
}

IEnumerator Squash(Transform tr) // landing: brief vertical squash, then restore
{
    var s0 = tr.localScale;
    tr.localScale = new Vector3(s0.x * 1.15f, s0.y * 0.7f, s0.z * 1.15f);
    float t = 0f;
    while (t < 0.12f)
    {
        t += Time.deltaTime;
        tr.localScale = Vector3.Lerp(tr.localScale, s0, t / 0.12f);
        yield return null;
    }
    tr.localScale = s0;
}

IEnumerator PlayClaim(Item item, CaptureHex cap, PlayerId? viewer, HexCoord? buildCell = null, SoundKind sound = SoundKind.Claim)
{
    var cell = cap != null ? cap.Cell : buildCell.Value;
    // hidden claims/builds are silent and instant (fog: zero time-cost, no sound leak)
    var span = FogPresentation.VisibleSpan(item.Next, viewer, new[] { cell });
    if (span.First < 0) yield break;

    SoundManager.Play(sound);
    _board.UpdateControlTint(item.Next);
    // tint pulse: flash the tile fill toward white briefly by scaling the column's fill emission —
    // cheapest WebGL-safe pulse is a quick quad flash on the hex top
    var flashPos = Tokens().CellTop(cell, item.Next.Board.TileAt(cell).Elevation) + Vector3.up * 0.02f;
    ExplosionFx.Spawn(flashPos, cap != null
        ? (cap.Issuer == PlayerId.Player0 ? new Color(0.27f, 0.68f, 1f) : new Color(0.92f, 0.28f, 0.28f))
        : new Color(0.9f, 0.9f, 0.6f), 0.5f, false);
    yield return new WaitForSeconds(0.15f);
}
```

Add the move squash while here — at the end of `PlayMove`, after the hop loop:

```csharp
yield return Squash(token.transform);
```

Also update `PlayInstantSound`: remove the now-covered `CaptureHex`, `BuildGenerator`, `DeployUnit`, `EndTurn` cases (leaving `DeployGenerator`/`CreateUnit`, which have no board presence to animate — `CreateUnit` is a barracks-panel design action).

- [ ] **Step 2: Gate the game-over banner on presenter idle (`GameHud.cs:202-209`)**

Replace the `if (!_wasOver) { ... }` block body:

```csharp
if (!_wasOver)
{
    _wasOver = true;
    var accent = s.Winner == null ? new Color(0.25f, 0.27f, 0.33f, 0.96f)
               : p0Won ? P0ToastBlue : P1ToastRed;
    StartCoroutine(ShowBannerWhenQuiet(result.ToUpperInvariant(), HowText(s), accent));
}
```

And add to GameHud:

```csharp
/// <summary>Hold the big banner until the final action's animation lands — the killing blow
/// is the climax; the banner must not cover it mid-flight.</summary>
System.Collections.IEnumerator ShowBannerWhenQuiet(string title, string how, Color accent)
{
    var presenter = _game != null ? _game.Presenter : null;
    while (presenter != null && presenter.IsBusy) yield return null;
    GameOverBanner.Show(title, how, accent, onMainMenu: () => _game.ReturnToMenu());
}
```

- [ ] **Step 3: Compile + play-mode check**

`check_compile_errors` → none. Play mode, territory mode on: deploy a unit (drops in + squash), claim a hex (flash + chime + tint updates), build a generator (pylon appears + flash), end turn (beat + handover toast still works). Kill the last enemy unit: death explosion plays **fully**, then the banner appears.

- [ ] **Step 4: Commit**

```powershell
git add Assets/HexWars/Presentation/ActionPresenter.cs Assets/HexWars/Presentation/GameHud.cs
git commit -m "feat(fx): deploy drop-in, claim/build pulse, turn-handover beat; game-over banner waits for the final animation to land"
```

---

### Task 6: Fog rules — visible spans, tracer from the dark, silent hidden actions

**Files:**
- Modify: `Assets/HexWars/Presentation/ActionPresenter.cs` (`PlayMove`, `PlayAttack`)

**Interfaces:**
- Consumes: `FogPresentation.VisibleSpan`, `FogPresentation.TracerOrigin` (Task 1).
- Produces: fog-correct playback. Rules (spec §4): enemy moves animate only their visible hops; a shot from an unseen attacker spawns at the fog boundary along the true line; fully hidden actions are skipped with zero time cost and **no sound**.

- [ ] **Step 1: Fog-aware moves — replace the guard at the top of `PlayMove`**

```csharp
IEnumerator PlayMove(Item item, MoveUnit mv, PlayerId? viewer)
{
    Unit before = FindUnit(item.Prev, mv.Issuer, mv.UnitId);
    if (before == null) yield break;

    var path = HexPath.Line(before.Cell, mv.Dest);
    // own units are always fully visible to their viewer; enemy paths are clipped to vision
    bool ownAction = !viewer.HasValue || mv.Issuer == viewer.Value;
    var span = ownAction ? (First: 0, Last: path.Count - 1)
                         : FogPresentation.VisibleSpan(item.Next, viewer, path);
    if (span.First < 0) yield break; // fully in the dark: silent, zero time

    // a token may not exist yet (unit was out of sight): sync to Prev-visibility is stale, so
    // sync Next first — the destination decides existence — then walk only the visible hops
    Tokens().Sync(item.Next, viewer);
    var token = Tokens().UnitToken(mv.UnitId);
    if (token == null) yield break;

    SoundManager.Play(SoundKind.Move);
    token.transform.localPosition = Tokens().CellTop(path[span.First], item.Next.Board.TileAt(path[span.First]).Elevation);
    if (span.First > 0) yield return PopIn(token.transform);            // enters vision mid-path
    for (int i = span.First + 1; i <= span.Last; i++)
    {
        Vector3 from = token.transform.localPosition;
        Vector3 to = Tokens().CellTop(path[i], item.Next.Board.TileAt(path[i]).Elevation);
        for (float t = 0f; t < SecondsPerHop; t += Time.deltaTime)
        {
            token.transform.localPosition = Vector3.Lerp(from, to, Mathf.SmoothStep(0f, 1f, t / SecondsPerHop));
            yield return null;
        }
    }
    if (span.Last < path.Count - 1) yield return PopOut(token.transform); // leaves vision mid-path
    else yield return Squash(token.transform);
}

// WebGL-safe appear/disappear (no transparent variants): quick scale pop at the vision boundary
IEnumerator PopIn(Transform tr)
{
    var s0 = tr.localScale;
    for (float t = 0f; t < 0.15f; t += Time.deltaTime)
    {
        tr.localScale = s0 * Mathf.SmoothStep(0f, 1f, t / 0.15f);
        yield return null;
    }
    tr.localScale = s0;
}

IEnumerator PopOut(Transform tr)
{
    var s0 = tr.localScale;
    for (float t = 0f; t < 0.15f; t += Time.deltaTime)
    {
        tr.localScale = s0 * (1f - Mathf.SmoothStep(0f, 1f, t / 0.15f));
        yield return null;
    }
    tr.localScale = s0; // Commit's Sync decides whether it still exists; scale restored either way
}
```

*(Note the ordering change: `Sync(Next)` moved to the head of `PlayMove` — the final `Commit` sync is a harmless no-op re-snap.)*

- [ ] **Step 2: Tracer from the dark — in `PlayAttack`, replace the `from` computation**

```csharp
var atkCell = (cell: attacker.Cell, elev: attacker.Elevation);
var tgt = TargetCell(item.Prev, atk.TargetId);
if (tgt == null) yield break;

var origin = FogPresentation.TracerOrigin(item.Prev, viewer, atkCell.cell, tgt.Value.cell);
if (origin == null) yield break; // both ends dark: nothing to show (state popups don't apply — your units are always visible to you)

Vector3 from = Tokens().CellTop(origin.Value, item.Prev.Board.Contains(origin.Value) ? item.Prev.Board.TileAt(origin.Value).Elevation : atkCell.elev) + Vector3.up * 0.4f;
Vector3 to = Tokens().CellTop(tgt.Value.cell, tgt.Value.elev) + Vector3.up * 0.4f;
```

(and delete the old `from`/`to`/`TargetTop` lines that this supersedes — `TargetTop` becomes unused; remove it). The rest of the flight/impact code is unchanged: the tracer flies the clamped segment with the **true bearing**, because `origin` lies on the attacker→target hex line.

- [ ] **Step 3: Compile + fog scenario check**

`check_compile_errors` → none. Play mode with **FogOfWar on, vs AI**: watch several AI turns.
- An AI unit crossing your vision edge pops in at the boundary hop, slides the visible hops, pops out — never renders in the dark.
- An AI attack from an unseen hex: the tracer streaks in from the fog edge along the true bearing; impact + popup normal; the attacker's token never appears.
- AI moves fully in the dark: nothing plays, no sound, and the AI's turn does not audibly "tick" through hidden actions (no timing leak).
- Toggle fog off in the lobby: everything animates fully (span = full path).

For a deterministic check of the "no sound leak" rule, `execute_script` while playing a fogged AI game: confirm `ActionPresenter.IsBusy` returns to false in < 0.1 s after enqueueing a fully-hidden enemy move (drive one directly: apply a `MoveUnit` for an unseen AI unit through `GameEngine.Apply` + `Presenter.Enqueue(prev, cmd, next, false)` and time the drain with `Time.realtimeSinceStartup`).

- [ ] **Step 4: Commit**

```powershell
git add Assets/HexWars/Presentation/ActionPresenter.cs
git commit -m "feat(fog): presentation honours vision - enemy moves animate only visible hops (pop at the boundary), unseen attackers fire a tracer from the fog edge with true bearing, fully hidden actions are silent and instant"
```

---

### Task 7: Camera — nudge to off-screen action, micro-shake on kills

**Files:**
- Modify: `Assets/HexWars/Presentation/CameraRig.cs`
- Modify: `Assets/HexWars/Presentation/ActionPresenter.cs` (call the nudge before opponent actions; shake on kill)

**Interfaces:**
- Produces:
  - `CameraRig.NudgeToward(Vector3 world)` — smooth focus glide (~0.4 s); **any** user camera input (keys, scroll, touch) cancels it instantly.
  - `CameraRig.Shake(float amplitude = 0.18f, float duration = 0.3f)` — additive positional shake, decaying.

- [ ] **Step 1: Extend `CameraRig.cs`**

Add fields and methods:

```csharp
Coroutine _nudge;
Vector3 _shakeOffset;

/// <summary>Glide the focus toward a world point (opponent action off-screen). User input wins:
/// the first pan/orbit/zoom keypress or touch cancels the glide.</summary>
public void NudgeToward(Vector3 world)
{
    if (_nudge != null) StopCoroutine(_nudge);
    _nudge = StartCoroutine(NudgeSeq(world));
}

IEnumerator NudgeSeq(Vector3 world)
{
    Vector3 from = _focus;
    for (float t = 0f; t < 0.4f; t += Time.deltaTime)
    {
        _focus = Vector3.Lerp(from, world, Mathf.SmoothStep(0f, 1f, t / 0.4f));
        yield return null;
    }
    _focus = world;
    _nudge = null;
}

public void Shake(float amplitude = 0.18f, float duration = 0.3f)
{
    StartCoroutine(ShakeSeq(amplitude, duration));
}

IEnumerator ShakeSeq(float amplitude, float duration)
{
    for (float t = 0f; t < duration; t += Time.deltaTime)
    {
        float damp = 1f - t / duration;
        _shakeOffset = new Vector3(
            (Mathf.PerlinNoise(t * 40f, 0.3f) - 0.5f),
            (Mathf.PerlinNoise(0.7f, t * 40f) - 0.5f), 0f) * (2f * amplitude * damp);
        yield return null;
    }
    _shakeOffset = Vector3.zero;
}
```

In `Update()`, cancel the nudge on any user input — inside the keyboard block add after each movement key group (simplest: set a flag):

```csharp
bool userMoved = false;
```
…and set `userMoved = true;` alongside every `_focus`/`Yaw`/`Distance` mutation (WASD/arrows at :47-50, Q/E at :51-52, scroll at :59-60, one-finger drag at :77, pinch at :83). Then before `Apply()`:

```csharp
if (userMoved && _nudge != null) { StopCoroutine(_nudge); _nudge = null; }
```

In `Apply()` (line 90-95), add the shake:

```csharp
transform.position = _focus - rot * Vector3.forward * Distance + _shakeOffset;
```

- [ ] **Step 2: Drive it from `ActionPresenter`**

Add to the class:

```csharp
CameraRig _rig;
CameraRig Rig() => _rig != null ? _rig : (_rig = FindAnyObjectByType<CameraRig>());

static bool OffScreen(Vector3 world)
{
    var cam = Camera.main;
    if (cam == null) return false;
    var v = cam.WorldToViewportPoint(world);
    return v.z < 0f || v.x < 0.06f || v.x > 0.94f || v.y < 0.06f || v.y > 0.94f;
}
```

At the top of `Play(Item item)`, before the switch — opponent actions only, and only when the action site is off-screen (spec: never yank a visible camera):

```csharp
if (!item.IsLocal)
{
    var site = ActionSite(item);
    if (site.HasValue && OffScreen(site.Value))
    {
        Rig()?.NudgeToward(site.Value);
        yield return new WaitForSeconds(0.25f); // let the glide lead the action
    }
}
```

With the site helper (world position of the action's focal cell, fog-clamped — reuse what playback already computes):

```csharp
Vector3? ActionSite(Item item)
{
    switch (item.Cmd)
    {
        case MoveUnit mv:
        {
            var u = FindUnit(item.Prev, mv.Issuer, mv.UnitId);
            if (u == null) return null;
            var span = FogPresentation.VisibleSpan(item.Next, _game.FogViewerFor(item.Next), HexPath.Line(u.Cell, mv.Dest));
            if (span.First < 0) return null; // hidden: no nudge (and no playback)
            return Tokens().CellTop(mv.Dest, item.Next.Board.TileAt(mv.Dest).Elevation);
        }
        case AttackUnit atk:
        {
            var t = TargetCell(item.Prev, atk.TargetId);
            return t == null ? (Vector3?)null : Tokens().CellTop(t.Value.cell, t.Value.elev);
        }
        case DeployUnit dep: return Tokens().CellTop(dep.Cell, item.Next.Board.TileAt(dep.Cell).Elevation);
        case CaptureHex cap: return Tokens().CellTop(cap.Cell, item.Next.Board.TileAt(cap.Cell).Elevation);
        case BuildGenerator bld: return Tokens().CellTop(bld.Cell, item.Next.Board.TileAt(bld.Cell).Elevation);
        default: return null;
    }
}
```

Kill shake — in `Commit`, next to the death sound:

```csharp
if (LiveUnits(item.Next) < LiveUnits(item.Prev)) { SoundManager.Play(SoundKind.Death); Rig()?.Shake(); }
```

- [ ] **Step 3: Compile + play-mode check**

`check_compile_errors` → none. Vs-AI game, camera panned away from the AI's side: on the AI's turn the camera glides to the action, plays it, glides to the next. Pan manually during a glide → the glide stops dead, your input wins. Kill a unit → short shake with the explosion; ordinary hits → no shake.

- [ ] **Step 4: Commit**

```powershell
git add Assets/HexWars/Presentation/CameraRig.cs Assets/HexWars/Presentation/ActionPresenter.cs
git commit -m "feat(camera): glide to off-screen opponent actions (user input cancels), micro-shake on kills only"
```

---

### Task 8: Action-denied feedback — every refused click says why

*(Added 2026-07-06 after user playtest: presentation-side guards eat clicks silently, so "the
system isn't responding" has no visible cause. Engine-side rejections already toast via
`GameBootstrap.TryApply` → `Toast.Show(Friendly(...))`; this task covers the guards that return
BEFORE a command is ever issued. Depends on Task 4's rewritten `HandleClick` — the code below is
its post-Task-4 shape.)*

**Files:**
- Modify: `Assets/HexWars/Presentation/GameBootstrap.cs` (add `WaitingHumanSeat()` next to `IsLocalCommand`)
- Modify: `Assets/HexWars/Presentation/UnitInputController.cs` (guard branches in `HandleClick` toast their reason; `WhyCannotTarget` / `IsOccupied` / `NotifyIfWaiting` helpers)

**Interfaces:**
- Consumes: `Toast.Show(string)` (existing bottom-centre rejection toast — built for exactly this, see its doc comment), `TargetingService.InRange` / `IsVisibleToArmy` / `HasShot` (the three predicates `CanTarget` ANDs together, `TargetingService.cs:14-24` — asking them one at a time tells us WHICH refused), `MovementService.ReachableTiles` (already used by `IsReachable`).
- Produces: a short reason toast for every refused click. No new UI surfaces, no engine changes.

- [ ] **Step 1: `GameBootstrap.WaitingHumanSeat()`**

Next to `IsLocalCommand` (same seat logic, opposite question):

```csharp
/// <summary>The seated human currently waiting out someone else's turn: your seat online when
/// it isn't your turn; the human's seat while the AI plays. Null in hotseat (the active player
/// IS the human at the screen), for unseated spectators, when the game is over, and when it is
/// your turn — i.e. null means "no one to apologise to".</summary>
public PlayerId? WaitingHumanSeat()
{
    if (State == null || State.IsGameOver) return null;
    if (Networked) return Seat.HasValue && State.ActivePlayer != Seat.Value ? Seat : null;
    var ai = GetComponent<AiOpponent>();
    if (ai != null && State.ActivePlayer == ai.AiSeat)
        return ai.AiSeat == PlayerId.Player0 ? PlayerId.Player1 : PlayerId.Player0;
    return null;
}
```

- [ ] **Step 2: Reason toasts at every silent guard in `HandleClick`**

Build-mode branch — split the compound condition so each refusal names itself:

```csharp
// build mode (territory): a tap places a generator on any empty hex you control
if (_buildMode && _game.State.Config.TerritoryMode)
{
    var bst = _game.State;
    HexCoord? target = tile != null ? (HexCoord?)tile.Coord : (unit != null ? (HexCoord?)unit.Unit.Cell : null);
    if (!target.HasValue) return; // tapped past the board — no message
    // guard first: during the opponent's turn "active" is them, so the controller check below
    // would misreport YOUR OWN hex as not yours
    if (_game.WaitingHumanSeat() != null) { Toast.Show("Opponent's turn — waiting for it to finish"); return; }
    if (bst.Board.Controller(target.Value) != active) { Toast.Show("You don't control that hex"); return; }
    if (HasGeneratorOn(bst, target.Value)) { Toast.Show("That hex already has a generator"); return; }
    _game.TryApply(new BuildGenerator(active, target.Value));
    return; // while building, taps only place generators
}
```

Attack branch (post-Task-4 shape):

```csharp
if (ownSelected && unit != null && unit.Unit.Owner != active)
{
    if (HasActed(_game.State.AttackedUnitIds, _selected.Unit.Id)) { Toast.Show("Already attacked this turn"); return; }
    if (!TargetingService.CanTarget(_game.State, _selected.Unit, unit.Unit.Cell, unit.Unit.Elevation))
    { Toast.Show(WhyCannotTarget(_game.State, _selected.Unit, unit.Unit)); return; }
    Issue(new AttackUnit(active, _selected.Unit.Id, unit.Unit.Id));
    return;
}
```

Move branch, and the fall-through gets the not-your-turn notice before selection proceeds:

```csharp
if (ownSelected && unit == null && tile != null)
{
    if (HasActed(_game.State.MovedUnitIds, _selected.Unit.Id)) { Toast.Show("Already moved this turn"); return; }
    if (!IsReachable(_selected.Unit, tile.Coord))
    { Toast.Show(IsOccupied(_game.State, tile.Coord) ? "That hex is occupied" : "Out of movement reach"); return; }
    Issue(new MoveUnit(active, _selected.Unit.Id, tile.Coord));
    return;
}
NotifyIfWaiting(unit, tile);
Select(unit);
```

Helpers:

```csharp
/// <summary>TargetingService.CanTarget's three ANDed predicates, asked one at a time so the
/// toast can say WHICH one refused. Ends on a generic fallback so a future rules change can
/// never make this method lie.</summary>
static string WhyCannotTarget(GameState s, Unit attacker, Unit target)
{
    if (!TargetingService.InRange(attacker, target.Cell, target.Elevation, s.Config)) return "Out of range";
    if (!TargetingService.IsVisibleToArmy(s, attacker.Owner, target.Cell, target.Elevation)) return "No friendly unit can see the target";
    if (!TargetingService.HasShot(s, attacker, target.Cell, target.Elevation)) return "No line of sight";
    return "Can't target that unit";
}

static bool IsOccupied(GameState s, HexCoord cell)
{
    foreach (var p in s.Players)
        foreach (var u in p.UnitsOnBoard)
            if (u.IsAlive && u.Cell == cell) return true;
    return false;
}

/// <summary>A click that reads as an order (a live unit of the waiting human's is selected and
/// they tapped a hex or an enemy) while the opponent's turn plays out: say why nothing will
/// happen. Never fires in hotseat. Selection/inspection still proceeds after the toast.</summary>
void NotifyIfWaiting(UnitView unit, TileView tile)
{
    var waiting = _game != null ? _game.WaitingHumanSeat() : null;
    if (waiting == null || _selected == null || !_selected.Unit.IsAlive || _selected.Unit.Owner != waiting.Value) return;
    bool looksLikeOrder = tile != null || (unit != null && unit.Unit.Owner != waiting.Value);
    if (looksLikeOrder) Toast.Show("Opponent's turn — waiting for it to finish");
}
```

Message wording is final as written (short, no punctuation soup, matches the `Friendly(...)` tone).

- [ ] **Step 3: Verify (editor scripts, then compile)**

RED first: `execute_script` reflecting `UnitInputController.WhyCannotTarget` fails before the method exists. Then implement, `check_compile_errors` → none, and GREEN via `execute_script`:
- craft a `GameState` with an attacker (Range 1, RangeArc 0) and a target 3 hexes away → `WhyCannotTarget` returns `"Out of range"`;
- target adjacent but behind a +2 elevation wall (no arc) with a spotter seeing it → `"No line of sight"`;
- target in range/LOS but no friendly spotter sees it → `"No friendly unit can see the target"`;
- `IsOccupied` true on a unit's cell, false on an empty one;
- `WaitingHumanSeat()`: hotseat state → null; with an `AiOpponent` whose seat is active → the other seat; game over → null.

The full click-through sweep (each toast appearing on a real refused click) rides Task 9's play-mode regression.

- [ ] **Step 4: Commit**

```powershell
git add Assets/HexWars/Presentation/UnitInputController.cs Assets/HexWars/Presentation/GameBootstrap.cs
git commit -m "feat(ui): every refused click says why - reason toasts for spent units, out-of-range/no-LOS/unseen targets, unreachable or occupied hexes, invalid build taps, and orders issued during the opponent's turn"
```

---

### Task 9: Full regression pass + WebGL build & stage

**Files:**
- No new source. Build output under `wwwroot` via `engine/stage-webgl-deploy.ps1`.

- [ ] **Step 1: The success-criterion playtest (spec §8)**

Fog ON, vs AI (Hard), territory mode on, watch 2+ full games (a spectator AI-vs-AI run via the HexWars editor menu is fine for one of them). Verify by narrating each opponent turn from the screen alone:
- no token ever teleports or pops except at the vision boundary;
- every damage popup coincides with an impact;
- hidden activity is invisible AND inaudible;
- tracers from the dark read clearly and point along a true bearing;
- the game-over banner never cuts off the final kill.

Regression checklist (spec §7 quiet details), all in the same session: spent-unit dimming flips per turn · unit click colliders/selection marker · hover + docked unit stats · territory tint updates on claim/build · HP bars track damage and ride moving tokens · barracks deploy flow · turn-handover toast · pace (K-actions) auto-pass beat · replay viewer still renders (`ReplayViewerMenu`) · lobby → game → Main menu → new game teardown leaves no stray tokens (`ReturnToMenu` resets the queue).

Denial-toast sweep (Task 8): deliberately click each refused action and read its reason — spent unit's second move/attack · out-of-range target · adjacent target behind a wall (no LOS) · unreachable and occupied hexes · build-tap on enemy territory and on an existing generator · an order during the AI's turn ("Opponent's turn") · confirm hotseat shows NO not-your-turn toast.

Fix anything found; commit fixes individually with plain `fix(fx): ...` messages.

- [ ] **Step 2: WebGL build + stage**

Build via the editor menu / `WebGLBuild.cs` as usual, then:

```powershell
powershell -File engine/stage-webgl-deploy.ps1
```

Load the staged build locally (or after deploy) in a **mobile-width** browser window: frame pacing stays smooth during animation bursts; sounds fire once per action.

- [ ] **Step 3: Commit the build + hand off for deploy**

```powershell
git add wwwroot
git commit -m "deploy: client build with opponent-turn presentation (animated actions, fog tracers, camera glide)"
```

Then tell the user: push from WSL (Windows shell has no key — memory `hexwars-deploy-and-git`), and smoke the Render URL on desktop + phone.

---

## Self-review notes (resolved during writing)

- **Scope correction vs the spec's §3.3 "free win":** only `SpectatorDriver` and `AiOpponent` route through `TryApply` and therefore animate for free. `ReplayPlayer` scrubs arbitrary frames (including backwards) and `ModelDuelDriver` renders `DuelEnv` states directly — both stay on the instant `RenderEntities` facade, which is the correct behavior for a scrubber. Animating sequential replay playback is a follow-up (the `Replay` object does keep its command list). The spec has been amended to match.
- **FastForward is synchronous** (StopAllCoroutines + drain), not a polled flag: a flag cleared on queue-empty would also snap-commit the local action enqueued immediately after the call — the exact action the player just clicked.
- `_reported` prevents both dropped and doubled damage popups when an attack is fast-forwarded before/after its impact frame.

- `CombatFx.Top` duplicates position math with `TokenStore.CellTop` — left as-is; CombatFx is untouched by design (only its call timing moved). Consolidating is post-milestone cleanup.
- `PlayInstantSound` shrinks task by task until only `DeployGenerator`/`CreateUnit` remain — intended, they have no board animation.
- `OnNetReject` (`GameBootstrap.cs:259-264`) still calls the `RenderEntities` facade — correct: it snaps tokens to truth, which is exactly what a reject needs.
- Spectators (null viewer) get full-span animation everywhere by construction (`VisibleSpan` returns full span when `viewer == null`).
