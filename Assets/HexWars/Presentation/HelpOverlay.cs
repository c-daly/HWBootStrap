using UnityEngine;
using UnityEngine.UI;

namespace HexWars.Presentation
{
    /// <summary>
    /// A "?" button (top-right) that opens the full <see cref="GameRules"/> reference. Auto-created in
    /// any scene that has a <see cref="GameBootstrap"/>, so players always have instructions one tap away.
    /// </summary>
    public sealed class HelpOverlay : MonoBehaviour
    {
        GameObject _canvasGo;
        GameBootstrap _game;
        Font _font;

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void AutoCreate()
        {
            if (FindAnyObjectByType<GameBootstrap>() == null) return;
            new GameObject("HelpOverlay").AddComponent<HelpOverlay>();
        }

        // Hidden while the title demo runs (the title menu has its own How to Play) and during the
        // connecting window (no state yet). Poll with the same SetActive-when-different guard as
        // GameHud, so the canvas flips once per change.
        void Update()
        {
            if (_game == null || _canvasGo == null) return;
            bool hidden = _game.DemoMode || _game.State == null;
            if (_canvasGo.activeSelf == hidden)
                _canvasGo.SetActive(!hidden);
        }

        void Start()
        {
            _font = UiKit.Font();
            _game = FindAnyObjectByType<GameBootstrap>();

            var canvasGo = new GameObject("HelpCanvas");
            _canvasGo = canvasGo;
            canvasGo.transform.SetParent(transform, false);
            var canvas = canvasGo.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            canvas.sortingOrder = 900; // above the HUD, below the lobby (1000)
            var scaler = canvasGo.AddComponent<CanvasScaler>();
            scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(1200f, 800f);
            scaler.matchWidthOrHeight = 0.5f;
            canvasGo.AddComponent<GraphicRaycaster>();

            // UiKit.Button anchors top-centre (SetRect); re-anchor to top-right afterward so the "?"
            // lands in the same corner (-12, -12) it always has.
            var q = UiKit.Button(canvasGo.transform, "?", 0f, 0f, 54f, 54f,
                                 () => GameRules.Show(canvasGo.transform, _font, 950),
                                 UiKit.ButtonStyle.Secondary, 26);
            var qrt = q.GetComponent<RectTransform>();
            qrt.anchorMin = qrt.anchorMax = new Vector2(1f, 1f);
            qrt.pivot = new Vector2(1f, 1f);
            qrt.anchoredPosition = new Vector2(-12f, -12f);

            var tipsBtn = TipsService.BuildToggle(canvasGo.transform, 0f, 0f);
            var trt = tipsBtn.GetComponent<RectTransform>();
            trt.anchorMin = trt.anchorMax = new Vector2(1f, 1f);
            trt.pivot = new Vector2(1f, 1f);
            trt.anchoredPosition = new Vector2(-12f - 54f - 8f, -12f); // left of the "?" (54 wide, 8px gap)

            // the escape menu's touch trigger — mobile has no Esc key; EscapeMenu owns the modal itself
            var menuBtn = UiKit.Button(canvasGo.transform, "Menu", 0f, 0f, 80f, 34f,
                                       () => FindAnyObjectByType<EscapeMenu>()?.Toggle(),
                                       UiKit.ButtonStyle.Secondary, UiKit.SizeCaption);
            var mrt = menuBtn.GetComponent<RectTransform>();
            mrt.anchorMin = mrt.anchorMax = new Vector2(1f, 1f);
            mrt.pivot = new Vector2(1f, 1f);
            mrt.anchoredPosition = new Vector2(-12f - 54f - 8f - 110f - 8f, -12f); // left of the Tips toggle (110 wide)
        }
    }
}
