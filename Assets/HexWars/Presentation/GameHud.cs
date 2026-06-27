using UnityEngine;
using UnityEngine.UI;
using UnityEngine.EventSystems;
using UnityEngine.InputSystem.UI;
using HexWars.Engine;

namespace HexWars.Presentation
{
    /// <summary>
    /// Top banner showing the active player, points, round, and barracks count, with an End Turn
    /// button. Builds its own canvas; ensures an EventSystem exists so UI buttons get clicks.
    /// </summary>
    public sealed class GameHud : MonoBehaviour
    {
        GameBootstrap _game;
        Text _banner;

        void Start()
        {
            EnsureEventSystem();
            Build();
            _game = FindAnyObjectByType<GameBootstrap>();
            if (_game != null)
            {
                _game.StateChanged += Refresh;
                Refresh();
            }
        }

        void OnDestroy()
        {
            if (_game != null) _game.StateChanged -= Refresh;
        }

        static void EnsureEventSystem()
        {
            if (FindAnyObjectByType<EventSystem>() != null) return;
            var es = new GameObject("EventSystem");
            es.AddComponent<EventSystem>();
            var module = es.AddComponent<InputSystemUIInputModule>();
            module.AssignDefaultActions(); // without actions the module silently ignores UI input
        }

        void Build()
        {
            var canvasGo = new GameObject("HudCanvas");
            canvasGo.transform.SetParent(transform, false);
            var canvas = canvasGo.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            canvas.sortingOrder = 500;
            var scaler = canvasGo.AddComponent<CanvasScaler>();
            scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(1600f, 900f);
            scaler.matchWidthOrHeight = 0.5f;
            canvasGo.AddComponent<GraphicRaycaster>();

            var bar = new GameObject("Banner");
            bar.transform.SetParent(canvasGo.transform, false);
            bar.AddComponent<Image>().color = new Color(0.05f, 0.06f, 0.10f, 0.88f);
            var brt = bar.GetComponent<RectTransform>();
            brt.anchorMin = new Vector2(0f, 1f);
            brt.anchorMax = new Vector2(1f, 1f);
            brt.pivot = new Vector2(0.5f, 1f);
            brt.sizeDelta = new Vector2(0f, 46f);
            brt.anchoredPosition = Vector2.zero;

            var textGo = new GameObject("BannerText");
            textGo.transform.SetParent(bar.transform, false);
            _banner = textGo.AddComponent<Text>();
            _banner.font = BuiltinFont();
            _banner.fontSize = 20;
            _banner.color = Color.white;
            _banner.alignment = TextAnchor.MiddleLeft;
            var trt = _banner.GetComponent<RectTransform>();
            trt.anchorMin = Vector2.zero;
            trt.anchorMax = Vector2.one;
            trt.offsetMin = new Vector2(16f, 0f);
            trt.offsetMax = new Vector2(-160f, 0f);

            var btn = new GameObject("EndTurnButton");
            btn.transform.SetParent(bar.transform, false);
            btn.AddComponent<Image>().color = new Color(0.20f, 0.34f, 0.55f, 1f);
            btn.AddComponent<Button>().onClick.AddListener(OnEndTurn);
            var rt = btn.GetComponent<RectTransform>();
            rt.anchorMin = new Vector2(1f, 0.5f);
            rt.anchorMax = new Vector2(1f, 0.5f);
            rt.pivot = new Vector2(1f, 0.5f);
            rt.sizeDelta = new Vector2(140f, 34f);
            rt.anchoredPosition = new Vector2(-8f, 0f);

            var btnTextGo = new GameObject("Text");
            btnTextGo.transform.SetParent(btn.transform, false);
            var btnText = btnTextGo.AddComponent<Text>();
            btnText.font = BuiltinFont();
            btnText.text = "End Turn";
            btnText.fontSize = 18;
            btnText.color = Color.white;
            btnText.alignment = TextAnchor.MiddleCenter;
            var btrt = btnText.GetComponent<RectTransform>();
            btrt.anchorMin = Vector2.zero;
            btrt.anchorMax = Vector2.one;
            btrt.offsetMin = Vector2.zero;
            btrt.offsetMax = Vector2.zero;
        }

        void OnEndTurn()
        {
            if (_game != null) _game.TryApply(new EndTurn(_game.State.ActivePlayer));
        }

        void Refresh()
        {
            if (_game == null || _game.State == null) return;
            var s = _game.State;
            var p = s.Player(s.ActivePlayer);
            bool p0 = s.ActivePlayer == PlayerId.Player0;
            int who = p0 ? 1 : 2;
            _banner.color = p0 ? new Color(0.4f, 0.8f, 1f) : new Color(1f, 0.45f, 0.45f);

            if (!s.Config.TerritoryMode)
            {
                _banner.text = $"Player {who}'s turn  (move {(p0 ? "cyan" : "red")})     {p.Points} pts     Round {s.Round}     Barracks {p.Barracks.Count}";
                return;
            }

            _banner.text =
                $"P{who}'s turn  Round {s.Round}     " +
                $"P1 {Stat(s, PlayerId.Player0)}   |   P2 {Stat(s, PlayerId.Player1)}";
        }

        static string Stat(GameState s, PlayerId id)
        {
            int net = Economy.Income(s, id) - Economy.Upkeep(s, id);
            int pts = s.Player(id).Points;
            int hexes = s.Board.ControlledCount(id);
            int score = WinCheck.Score(s, id);
            string sign = net >= 0 ? "+" : "";
            return $"{pts}p ({sign}{net}/t)  {hexes} hex  score {score}";
        }

        static Font BuiltinFont()
        {
            var f = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            if (f == null) f = Resources.GetBuiltinResource<Font>("Arial.ttf");
            return f;
        }
    }
}
