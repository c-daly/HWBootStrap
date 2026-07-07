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
                case AttackUnit atk: yield return PlayAttack(item, atk, viewer); break;
                // Deploy/Capture/Build/EndTurn in Task 5.
                default: PlayInstantSound(item); break;
            }
        }

        IEnumerator PlayMove(Item item, MoveUnit mv, PlayerId? viewer)
        {
            Unit? before = FindUnit(item.Prev, mv.Issuer, mv.UnitId);
            var token = Tokens().UnitToken(mv.UnitId);
            if (before == null || token == null) yield break; // hidden under fog (Task 6 refines) or gone

            SoundManager.Play(SoundKind.Move);
            var path = HexPath.Line(before.Value.Cell, mv.Dest);
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

        const float ImpactHold = 0.05f; // spec: brief hold at impact; PauseToggle owns timeScale, so hold locally

        IEnumerator PlayAttack(Item item, AttackUnit atk, PlayerId? viewer)
        {
            var attacker = FindUnit(item.Prev, atk.Issuer, atk.AttackerId);
            if (attacker == null) yield break;
            var targetPos = TargetTop(item.Prev, atk.TargetId);
            if (targetPos == null) yield break;

            int dmg = attacker.Value.Stats.Damage;
            float power = Mathf.Clamp01(dmg / 8f);
            float projScale = Mathf.Lerp(0.14f, 0.5f, power);
            Color projColor = dmg >= 6 ? new Color(1f, 0.3f, 0.1f)
                            : dmg >= 3 ? new Color(1f, 0.65f, 0.2f)
                                       : new Color(1f, 0.95f, 0.5f);

            Vector3 from = Tokens().CellTop(attacker.Value.Cell, attacker.Value.Elevation) + Vector3.up * 0.4f;
            Vector3 to = targetPos.Value + Vector3.up * 0.4f;

            bool directLos = LineOfSight.IsClear(item.Prev.Board, attacker.Value.Cell, attacker.Value.Elevation,
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

        void PlayInstantSound(Item item)
        {
            switch (item.Cmd)
            {
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

        internal static Unit? FindUnit(GameState s, PlayerId owner, int id)
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
