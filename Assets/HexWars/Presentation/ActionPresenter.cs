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

        CameraRig _rig;
        CameraRig Rig() => _rig != null ? _rig : (_rig = FindAnyObjectByType<CameraRig>());

        static bool OffScreen(Vector3 world)
        {
            var cam = Camera.main;
            if (cam == null) return false;
            var v = cam.WorldToViewportPoint(world);
            return v.z < 0f || v.x < 0.06f || v.x > 0.94f || v.y < 0.06f || v.y > 0.94f;
        }

        public void Enqueue(GameState prev, Command cmd, GameState next, bool isLocal)
        {
            _queue.Enqueue(new Item { Prev = prev, Cmd = cmd, Next = next, IsLocal = isLocal });
            if (!_playing) StartCoroutine(DrainQueue());
        }

        Item? _current;          // the item whose animation is mid-flight
        GameObject _projectile;  // live transient (attack tracer), destroyed on fast-forward
        bool _reported;          // did the mid-flight item already fire its CombatFx popups?
        bool _presented;         // did the mid-flight item show/sound anything, or was it fully hidden by fog?

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
                _presented = false;
                yield return Play(_current.Value);
                Commit(_current.Value, skipCombatFx: _reported);
                bool wasLocal = _current.Value.IsLocal;
                _current = null;
                // no pacing beat after a fully-hidden (silent, zero-time) action: spec §4 forbids a
                // timing side-channel that would let the gap itself leak "something happened in the fog"
                if (!wasLocal && _presented && _queue.Count > 0)
                    yield return new WaitForSeconds(OpponentGap);
            }
            _playing = false;
        }

        IEnumerator Play(Item item)
        {
            if (!item.IsLocal)
            {
                var site = ActionSite(item);
                if (site.HasValue && OffScreen(site.Value))
                {
                    Rig()?.NudgeToward(site.Value);
                    yield return new WaitForSeconds(0.25f); // let the glide lead the action
                }
            }

            var viewer = _game.FogViewerFor(item.Next);
            switch (item.Cmd)
            {
                case MoveUnit mv: yield return PlayMove(item, mv, viewer); break;
                case AttackUnit atk: yield return PlayAttack(item, atk, viewer); break;
                case DeployUnit dep: yield return PlayDeploy(item, dep, viewer); break;
                case CaptureHex cap: yield return PlayClaim(item, cap, viewer); break;
                case BuildGenerator bld: yield return PlayClaim(item, null, viewer, bld.Cell, SoundKind.Build); break;
                case EndTurn _: SoundManager.Play(SoundKind.EndTurn); _presented = true; yield return new WaitForSeconds(item.IsLocal ? 0f : OpponentGap); break;
                default: PlayInstantSound(item); _presented = true; break;
            }
        }

        IEnumerator PlayMove(Item item, MoveUnit mv, PlayerId? viewer)
        {
            Unit? before = FindUnit(item.Prev, mv.Issuer, mv.UnitId);
            if (before == null) yield break;

            var path = HexPath.Line(before.Value.Cell, mv.Dest);
            // own units are always fully visible to their viewer; enemy paths are clipped to vision
            bool ownAction = !viewer.HasValue || mv.Issuer == viewer.Value;
            var span = ownAction ? (First: 0, Last: path.Count - 1)
                                 : FogPresentation.VisibleSpan(item.Next, viewer, path);
            if (span.First < 0) yield break; // fully in the dark: silent, zero time

            // a token may not exist yet (unit was out of sight): sync to Prev-visibility is stale, so
            // sync Next first — the destination decides existence — then walk only the visible hops
            Tokens().Sync(item.Next, viewer);
            var token = Tokens().UnitToken(mv.UnitId);
            if (token == null && span.Last < path.Count - 1)
            {
                // popped back out of sight before its true (hidden) destination: Sync(Next) correctly
                // prunes a token that isn't visible at rest, but the pop-out animation below still
                // needs one for the visible middle stretch — re-sync as if the unit rested at the
                // last visible cell instead. Commit() re-syncs to the true Next right after this
                // coroutine returns, so this is purely a transient presentation fiction.
                Tokens().Sync(WithUnitCell(item.Next, mv.Issuer, mv.UnitId, path[span.Last]), viewer);
                token = Tokens().UnitToken(mv.UnitId);
            }
            if (token == null) yield break;

            SoundManager.Play(SoundKind.Move);
            _presented = true;
            token.transform.localPosition = Tokens().CellTop(path[span.First], item.Next.Board.TileAt(path[span.First]).Elevation);
            if (span.First > 0) yield return PopIn(token.transform);            // enters vision mid-path
            int lastElev = item.Next.Board.TileAt(path[span.First]).Elevation;
            for (int i = span.First + 1; i <= span.Last; i++)
            {
                Vector3 from = token.transform.localPosition;
                // the board rectangle is not convex in cube space, so a hex-line between two on-board
                // cells can cross cells that are off the board — same guard as LineOfSight's walk
                int elev = item.Next.Board.Contains(path[i]) ? item.Next.Board.TileAt(path[i]).Elevation : lastElev;
                lastElev = elev;
                Vector3 to = Tokens().CellTop(path[i], elev);
                for (float t = 0f; t < SecondsPerHop; t += Time.deltaTime)
                {
                    token.transform.localPosition = Vector3.Lerp(from, to, Mathf.SmoothStep(0f, 1f, t / SecondsPerHop));
                    yield return null;
                }
            }
            if (span.Last < path.Count - 1) yield return PopOut(token.transform); // leaves vision mid-path
            else yield return Squash(token.transform);
        }
        // (No cancellation flags inside the loops: FastForward stops the coroutines outright and
        // Commit's Sync re-snaps position and scale, so an interrupted tween can't strand a token.)

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

        const float ImpactHold = 0.05f; // spec: brief hold at impact; PauseToggle owns timeScale, so hold locally

        IEnumerator PlayAttack(Item item, AttackUnit atk, PlayerId? viewer)
        {
            var attacker = FindUnit(item.Prev, atk.Issuer, atk.AttackerId);
            if (attacker == null) yield break;
            var tgt = TargetCell(item.Prev, atk.TargetId);
            if (tgt == null) yield break;

            var origin = FogPresentation.TracerOrigin(item.Prev, viewer, attacker.Value.Cell, tgt.Value.cell);
            if (origin == null) yield break; // both ends dark: nothing to show (state popups don't apply — your units are always visible to you)

            int dmg = attacker.Value.Stats.Damage;
            float power = Mathf.Clamp01(dmg / 8f);
            float projScale = Mathf.Lerp(0.14f, 0.5f, power);
            int projTier = dmg >= 6 ? 2 : dmg >= 3 ? 1 : 0;
            Color projColor = ProjectileTierColors[projTier];

            Vector3 from = Tokens().CellTop(origin.Value, item.Prev.Board.Contains(origin.Value) ? item.Prev.Board.TileAt(origin.Value).Elevation : attacker.Value.Elevation) + Vector3.up * 0.4f;
            Vector3 to = Tokens().CellTop(tgt.Value.cell, tgt.Value.elev) + Vector3.up * 0.4f;

            bool directLos = LineOfSight.IsClear(item.Prev.Board, attacker.Value.Cell, attacker.Value.Elevation,
                                                 tgt.Value.cell, tgt.Value.elev);
            float arc = directLos ? 0f : Mathf.Max(2.5f, Vector3.Distance(from, to) * 0.35f);
            float flightDur = Mathf.Lerp(0.45f, 0.85f, power) + Vector3.Distance(from, to) * 0.035f;

            SoundManager.Play(SoundKind.Attack);
            _presented = true;
            _projectile = MakeProjectile(from, projScale, projTier);
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

        IEnumerator PlayDeploy(Item item, DeployUnit dep, PlayerId? viewer)
        {
            // find the newly deployed unit: present in Next, absent in Prev
            Unit? fresh = null;
            foreach (var u in item.Next.Player(dep.Issuer).UnitsOnBoard)
                if (u.IsAlive && u.Cell == dep.Cell && FindUnit(item.Prev, dep.Issuer, u.Id) == null) { fresh = u; break; }
            if (fresh == null) yield break;

            Tokens().Sync(item.Next, viewer); // spawns the token at its cell
            var token = Tokens().UnitToken(fresh.Value.Id);
            if (token == null) yield break;   // deployed out of the viewer's sight — silent, zero time
            SoundManager.Play(SoundKind.Build); // visibility gate first, then sound (mirrors PlayClaim)
            _presented = true;

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
            _presented = true;
            _board.UpdateControlTint(item.Next);
            // tint pulse: flash the tile fill toward white briefly by scaling the column's fill emission —
            // cheapest WebGL-safe pulse is a quick quad flash on the hex top
            var flashPos = Tokens().CellTop(cell, item.Next.Board.TileAt(cell).Elevation) + Vector3.up * 0.02f;
            ExplosionFx.Spawn(flashPos, cap != null
                ? (cap.Issuer == PlayerId.Player0 ? new Color(0.27f, 0.68f, 1f) : new Color(0.92f, 0.28f, 0.28f))
                : new Color(0.9f, 0.9f, 0.6f), 0.5f, false);
            yield return new WaitForSeconds(0.15f);
        }

        /// <summary>World position of the action's focal cell, fog-clamped (reuses what playback already
        /// computes). Null when there's nothing to nudge toward — including a fully hidden action, which
        /// must not move the camera either (fog discipline: zero time, zero sound, zero camera motion).</summary>
        Vector3? ActionSite(Item item)
        {
            var viewer = _game.FogViewerFor(item.Next);
            switch (item.Cmd)
            {
                case MoveUnit mv:
                {
                    var u = FindUnit(item.Prev, mv.Issuer, mv.UnitId);
                    if (u == null) return null;
                    var span = FogPresentation.VisibleSpan(item.Next, viewer, HexPath.Line(u.Value.Cell, mv.Dest));
                    if (span.First < 0) return null; // hidden: no nudge (and no playback)
                    return Tokens().CellTop(mv.Dest, item.Next.Board.TileAt(mv.Dest).Elevation);
                }
                case AttackUnit atk:
                {
                    // mirror PlayAttack's own gate exactly: a null tracer origin means both ends are
                    // dark, so PlayAttack shows nothing — the nudge must not leak that a shot happened
                    var attacker = FindUnit(item.Prev, atk.Issuer, atk.AttackerId);
                    var t = TargetCell(item.Prev, atk.TargetId);
                    if (attacker == null || t == null) return null;
                    var origin = FogPresentation.TracerOrigin(item.Prev, viewer, attacker.Value.Cell, t.Value.cell);
                    return origin == null ? (Vector3?)null : Tokens().CellTop(t.Value.cell, t.Value.elev);
                }
                case DeployUnit dep:
                {
                    if (FogPresentation.VisibleSpan(item.Next, viewer, new[] { dep.Cell }).First < 0) return null;
                    return Tokens().CellTop(dep.Cell, item.Next.Board.TileAt(dep.Cell).Elevation);
                }
                case CaptureHex cap:
                {
                    if (FogPresentation.VisibleSpan(item.Next, viewer, new[] { cap.Cell }).First < 0) return null;
                    return Tokens().CellTop(cap.Cell, item.Next.Board.TileAt(cap.Cell).Elevation);
                }
                case BuildGenerator bld:
                {
                    if (FogPresentation.VisibleSpan(item.Next, viewer, new[] { bld.Cell }).First < 0) return null;
                    return Tokens().CellTop(bld.Cell, item.Next.Board.TileAt(bld.Cell).Elevation);
                }
                default: return null;
            }
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

        static readonly Color[] ProjectileTierColors =
            { new Color(1f, 0.95f, 0.5f), new Color(1f, 0.65f, 0.2f), new Color(1f, 0.3f, 0.1f) };
        static readonly Dictionary<int, Material> ProjectileMats = new Dictionary<int, Material>();

        static Material ProjectileMaterial(int tier)
        {
            if (ProjectileMats.TryGetValue(tier, out var m) && m != null) return m;
            var unlit = Shader.Find("Universal Render Pipeline/Unlit");
            if (unlit == null) unlit = Shader.Find("Unlit/Color");
            m = new Material(unlit);
            var color = ProjectileTierColors[tier];
            if (m.HasProperty("_BaseColor")) m.SetColor("_BaseColor", color);
            m.color = color;
            ProjectileMats[tier] = m;
            return m;
        }

        GameObject MakeProjectile(Vector3 pos, float scale, int tier)
        {
            var p = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            DestroyImmediate(p.GetComponent<Collider>());
            p.transform.position = pos;
            p.transform.localScale = Vector3.one * scale;
            var mat = ProjectileMaterial(tier);
            var color = ProjectileTierColors[tier];
            var mr = p.GetComponent<MeshRenderer>();
            mr.sharedMaterial = mat;
            mr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            var trail = p.AddComponent<TrailRenderer>();
            trail.time = 0.18f;
            trail.startWidth = scale * 0.9f;
            trail.endWidth = 0f;
            trail.sharedMaterial = mat;   // shared tier material — TrailRenderer.material (the non-"shared"
                                           // property) auto-instantiates a clone on assignment, which would
                                           // silently defeat this cache one Material per projectile; tint
                                           // goes through the TrailRenderer's own startColor/endColor instead,
                                           // never through the material
            trail.startColor = color;
            trail.endColor = new Color(color.r, color.g, color.b, 0f);
            trail.numCapVertices = 2;
            return p;
        }

        void PlayInstantSound(Item item)
        {
            switch (item.Cmd)
            {
                case DeployGenerator _:
                case CreateUnit _: SoundManager.Play(SoundKind.Build); break;
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
            if (LiveUnits(item.Next) < LiveUnits(item.Prev)) { SoundManager.Play(SoundKind.Death); Rig()?.Shake(); }
            if (item.Next.IsGameOver && !item.Prev.IsGameOver) SoundManager.Play(SoundKind.Win);
        }

        internal static Unit? FindUnit(GameState s, PlayerId owner, int id)
        {
            foreach (var u in s.Player(owner).UnitsOnBoard)
                if (u.Id == id && u.IsAlive) return u;
            return null;
        }

        /// <summary>A copy of <paramref name="s"/> with one unit relocated — used to feed TokenStore a
        /// presentation-only snapshot (see PlayMove's pop-out fix-up) without touching engine truth.</summary>
        static GameState WithUnitCell(GameState s, PlayerId owner, int unitId, HexCoord cell)
        {
            var players = new List<PlayerState>(s.Players.Count);
            foreach (var p in s.Players)
            {
                if (p.Id != owner) { players.Add(p); continue; }
                var units = new List<Unit>(p.UnitsOnBoard);
                for (int i = 0; i < units.Count; i++)
                    if (units[i].Id == unitId) { units[i] = units[i].WithCell(cell, units[i].Elevation); break; }
                players.Add(new PlayerState(p.Id, p.Points, p.Barracks, units, p.Generators, p.DestroyedValue));
            }
            return new GameState(s.Board, s.Config, players, s.ActivePlayer, s.Round, s.NextEntityId,
                                  s.IsGameOver, s.Winner, s.MovedUnitIds, s.AttackedUnitIds, s.MovementSpent);
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
