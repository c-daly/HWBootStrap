using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.InputSystem;
using UnityEngine.EventSystems;
using HexWars.Engine;

namespace HexWars.Presentation
{
    /// <summary>
    /// Right-side barracks: lists the active player's reusable templates. Select one, then click a
    /// deployment-zone hex to deploy a paid clone (the template is not consumed). Stays in deploy
    /// mode so you can place several; re-click the selected template to stop.
    /// </summary>
    public sealed class BarracksPanel : MonoBehaviour
    {
        GameBootstrap _game;
        GameObject _canvasGo;
        RectTransform _list;
        Text _hint;
        int _deployIndex = -1;
        readonly List<Button> _rows = new List<Button>();

        public bool IsDeploying => _deployIndex >= 0;

        /// <summary>Spectator mode: still shows the active player's barracks, but the human can't deploy
        /// (the AI is playing). Set by <see cref="SpectatorDriver"/>.</summary>
        public bool ReadOnly;

        void Start()
        {
            _game = FindAnyObjectByType<GameBootstrap>();
            Build();
            if (_game != null) { _game.StateChanged += Rebuild; Rebuild(); }
        }

        void OnDestroy()
        {
            if (_game != null) _game.StateChanged -= Rebuild;
        }

        void Update()
        {
            if (ReadOnly || _deployIndex < 0 || _game == null) return; // spectating: no human deploys
            var pointer = Pointer.current; // mouse or touch
            var cam = Camera.main;
            if (pointer == null || cam == null || !pointer.press.wasReleasedThisFrame || IsOverUi()) return;

            // Deploy mode is easy to leave: only a successful deploy keeps it. Clicking a unit,
            // an illegal hex, or empty space hands control back to the board (it used to stay
            // sticky until the template was re-clicked).
            var mp = pointer.position.ReadValue();
            if (Physics.Raycast(cam.ScreenPointToRay(mp), out var hit, 1000f))
            {
                if (hit.collider.GetComponentInParent<UnitView>() != null) { StopDeploying(); return; }

                var tv = hit.collider.GetComponentInParent<TileView>();
                if (tv == null
                    || !_game.TryApply(new DeployUnit(_game.State.ActivePlayer, _deployIndex, tv.Coord)))
                    StopDeploying();
            }
            else StopDeploying();
        }

        void StopDeploying()
        {
            _deployIndex = -1;
            Rebuild();
        }

        void Build()
        {
            var canvasGo = UiKit.Canvas("BarracksCanvas", UiKit.OrderPanels, transform);
            _canvasGo = canvasGo;

            const float w = 230f;
            var panel = UiKit.Panel(canvasGo.transform, "BarracksPanel", UiKit.Surface);
            var prt = panel.GetComponent<RectTransform>();
            prt.anchorMin = prt.anchorMax = new Vector2(1f, 1f);
            prt.pivot = new Vector2(1f, 1f);
            prt.sizeDelta = new Vector2(w, 420f);
            prt.anchoredPosition = new Vector2(-8f, -58f);

            UiKit.Label(panel.transform, "BARRACKS", 0f, -8f, w - 24f, 24f, 18, TextAnchor.MiddleLeft);
            _hint = UiKit.Label(panel.transform, "Design a unit, then deploy it here.", 0f, -34f, w - 24f, 22f, 13, TextAnchor.MiddleLeft);

            var listGo = new GameObject("List");
            listGo.transform.SetParent(panel.transform, false);
            _list = listGo.AddComponent<RectTransform>();
            UiKit.SetRect(_list, 0f, -60f, w, 360f);
        }

        void Rebuild()
        {
            foreach (var r in _rows) Destroy(r.gameObject);
            _rows.Clear();
            if (_game == null) return;

            // hidden during the title demo and the connecting window (no state yet)
            if (_game.DemoMode || _game.State == null)
            {
                if (_canvasGo != null) _canvasGo.SetActive(false);
                return;
            }
            if (_canvasGo != null && !_canvasGo.activeSelf) _canvasGo.SetActive(true);

            var s = _game.State;
            var p = s.Player(s.ActivePlayer);
            if (_deployIndex >= p.Barracks.Count) _deployIndex = -1;

            for (int i = 0; i < p.Barracks.Count; i++)
            {
                var stats = p.Barracks[i].Stats;   // Task 10 rebuilds this row properly (name + delete)
                int cost = Economy.DeployCost(stats, s.Config);
                bool selected = i == _deployIndex;
                int idx = i;
                var row = UiKit.Button(_list, $"{Roles.Dominant(stats)}   deploy {cost}", 0f, -(4f + i * 34f), 214f, 30f,
                                       () => Select(idx), UiKit.ButtonStyle.Secondary);
                UiKit.SetToggled(row, selected);
                _rows.Add(row);
            }

            _hint.text = _deployIndex >= 0
                ? "Click a zone hex to deploy - anywhere else to stop."
                : (p.Barracks.Count == 0 ? "Design a unit, then deploy it here." : "Select a template to deploy.");
        }

        void Select(int i)
        {
            _deployIndex = (_deployIndex == i) ? -1 : i; // toggle
            Rebuild();
        }

        static bool IsOverUi()
        {
            var es = EventSystem.current;
            return es != null && es.IsPointerOverGameObject();
        }
    }
}
