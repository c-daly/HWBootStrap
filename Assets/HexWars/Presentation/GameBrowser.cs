using System;
using System.Collections;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.UI;

namespace HexWars.Presentation
{
    /// <summary>
    /// The lobby browser: polls the server's <c>GET /games</c> every 3 s while open and lists every
    /// public game waiting for an opponent — code + config summary per row, tap a row to expand the
    /// full configuration with a Join button. You never end up in a game you didn't want. Joining is
    /// the ordinary by-code path; if the seat filled in the race window the server's SEAT FULL toast
    /// fires and the list refreshes on the next poll.
    /// </summary>
    public sealed class GameBrowser : MonoBehaviour
    {
        const float PollSeconds = 3f;

        GameBootstrap _game;
        GameObject _canvasGo;
        Transform _listRoot;
        Text _status;
        string _expandedCode;         // row currently expanded to the full-config card
        GameDto[] _lastGames = Array.Empty<GameDto>();

        [Serializable] class GamesDto { public GameDto[] games; }
        [Serializable] public class GameDto
        {
            public string code; public string mode;
            public int width, height, pace, army, ageSeconds;
            public bool fog;
        }

        public static GameBrowser Open(GameBootstrap game)
        {
            var existing = game.GetComponent<GameBrowser>();
            if (existing != null) existing.Close();
            var b = game.gameObject.AddComponent<GameBrowser>();
            b._game = game;
            return b;
        }

        void Start()
        {
            if (_game == null) _game = FindAnyObjectByType<GameBootstrap>();
            Build();
            StartCoroutine(PollLoop());
        }

        void Update()
        {
            if (_game != null && _game.State != null && !_game.DemoMode) Close(); // a match started
        }

        public void Close()
        {
            StopAllCoroutines();
            if (_canvasGo != null) Destroy(_canvasGo);
            Destroy(this);
        }

        float _panelW;

        void Build()
        {
            UiKit.EnsureEventSystem();
            _canvasGo = UiKit.Canvas("BrowserCanvas", UiKit.OrderMenu, transform);
            _panelW = Mathf.Min(760f, AvailWidth() - 40f); // GameRules' clamp pattern — unchanged (700/720
                                                            // math below) whenever the canvas is wide enough,
                                                            // same as it always was at desktop/landscape widths

            var panel = UiKit.Panel(_canvasGo.transform, "Panel", UiKit.Surface).gameObject;
            var prt = panel.GetComponent<RectTransform>();
            prt.anchorMin = prt.anchorMax = new Vector2(0.5f, 0.5f);
            prt.pivot = new Vector2(0.5f, 0.5f);
            prt.sizeDelta = new Vector2(_panelW, 640f);
            prt.anchoredPosition = Vector2.zero;

            UiKit.Label(panel.transform, "Open Games", 0f, -24f, _panelW, 36f, UiKit.SizeTitle, TextAnchor.MiddleCenter);
            UiKit.Button(panel.transform, "Back", -_panelW * 0.5f + 50f, -26f, 90f, 34f,
                         () => { Close(); TitleScreen.Reopen(_game); }, UiKit.ButtonStyle.Secondary, UiKit.SizeBody);
            UiKit.Button(panel.transform, "Refresh", _panelW * 0.5f - 60f, -26f, 110f, 34f,
                         () => { StopAllCoroutines(); StartCoroutine(PollLoop()); }, // restart: fetch now, resume cadence — never two in-flight fetches
                         UiKit.ButtonStyle.Secondary, UiKit.SizeBody);

            _status = UiKit.Label(panel.transform, "Loading…", 0f, -70f, _panelW - 60f, 26f,
                                  UiKit.SizeCaption, TextAnchor.MiddleCenter, UiKit.TextFaint);

            var listGo = new GameObject("List");
            listGo.transform.SetParent(panel.transform, false);
            var lrt = listGo.AddComponent<RectTransform>();
            UiKit.SetRect(lrt, 0f, -100f, _panelW - 40f, 520f);
            _listRoot = listGo.transform;
        }

        float AvailWidth()
        {
            var rt = _canvasGo.GetComponent<RectTransform>();
            return rt != null && rt.rect.width > 0f ? rt.rect.width : 1200f;
        }

        IEnumerator PollLoop()
        {
            while (true)
            {
                yield return FetchOnce();
                yield return new WaitForSeconds(PollSeconds);
            }
        }

        IEnumerator FetchOnce()
        {
            using (var req = UnityWebRequest.Get(GamesUrl()))
            {
                req.timeout = 4;
                yield return req.SendWebRequest();
                if (req.result != UnityWebRequest.Result.Success)
                {
                    if (_status != null) _status.text = "Can't reach the server — retrying…";
                    yield break;
                }
                GamesDto dto = null;
                try { dto = JsonUtility.FromJson<GamesDto>(req.downloadHandler.text); }
                catch (Exception) { /* malformed = treat as fetch failure */ }
                if (dto == null || dto.games == null)
                {
                    if (_status != null) _status.text = "Can't reach the server — retrying…";
                    yield break;
                }
                _lastGames = dto.games;
                Rebuild();
            }
        }

        void Rebuild()
        {
            if (_listRoot == null) return;
            for (int i = _listRoot.childCount - 1; i >= 0; i--) Destroy(_listRoot.GetChild(i).gameObject);

            if (_lastGames.Length == 0)
            {
                _status.text = "No open games right now — host one, or play the AI while you wait.";
                UiKit.Button(_listRoot, "Play vs AI", -140f, -40f, 260f, 48f, () =>
                {
                    Close();
                    SetupForm.Open(_game, SetupForm.SetupMode.VsAi);
                }, UiKit.ButtonStyle.Primary);
                UiKit.Button(_listRoot, "Host Game", 140f, -40f, 260f, 48f, () =>
                {
                    Close();
                    SetupForm.Open(_game, SetupForm.SetupMode.Host);
                }, UiKit.ButtonStyle.Cta);
                return;
            }

            _status.text = $"{_lastGames.Length} open game{(_lastGames.Length == 1 ? "" : "s")} — tap one for details";
            float y = -4f;
            foreach (var g in _lastGames)
            {
                bool expanded = g.code == _expandedCode;
                y = BuildRow(g, y, expanded);
            }
        }

        static string PaceText(int k) => k <= 0 ? "whole army" : k + (k == 1 ? " action/turn" : " actions/turn");

        float BuildRow(GameDto g, float y, bool expanded)
        {
            float rowW = _panelW - 60f; // == 700f at desktop widths, same as the literal it replaces
            string age = g.ageSeconds < 60 ? $"{g.ageSeconds}s" : $"{g.ageSeconds / 60}m";
            string summary = $"{g.code}   ·   {g.mode} · {g.width}×{g.height}{(g.fog ? " · Fog" : "")}" +
                             $" · {PaceText(g.pace)} · {g.army} units · {age} ago";
            var code = g.code;
            var row = UiKit.Button(_listRoot, summary, 0f, y, rowW, 42f, () =>
            {
                _expandedCode = _expandedCode == code ? null : code;
                Rebuild();
            }, UiKit.ButtonStyle.Secondary, UiKit.SizeBody);
            var rowText = row.GetComponentInChildren<Text>();
            rowText.alignment = TextAnchor.MiddleLeft;
            var trt = rowText.GetComponent<RectTransform>();
            trt.anchoredPosition = new Vector2(14f, trt.anchoredPosition.y);
            y -= 46f;

            if (expanded)
            {
                var card = UiKit.Panel(_listRoot, "Detail", new Color(0.09f, 0.11f, 0.18f, 1f)).gameObject;
                UiKit.SetRect(card.GetComponent<RectTransform>(), 0f, y, rowW, 96f);
                UiKit.Label(card.transform,
                            $"Mode {g.mode}    Map {g.width}×{g.height}    Fog {(g.fog ? "on" : "off")}\n" +
                            $"Pace {PaceText(g.pace)}    Army {g.army} units    Waiting {age}",
                            -80f, -12f, 520f, 72f, UiKit.SizeBody, TextAnchor.UpperLeft, UiKit.TextDim);
                UiKit.Button(card.transform, "Join", rowW * 0.5f - 90f, -26f, 140f, 44f, () =>
                {
                    _status.text = $"Joining {g.code}…";
                    _game.StartNetGame(g.code, null);   // seat+start arrive via the normal net path
                }, UiKit.ButtonStyle.Cta, UiKit.SizeBody + 2);
                y -= 102f;
            }
            return y;
        }

        static string GamesUrl()
        {
            string page = Application.absoluteURL;
            if (!string.IsNullOrEmpty(page))
            {
                try { var uri = new Uri(page); return uri.Scheme + "://" + uri.Authority + "/games"; }
                catch { }
            }
            return "http://127.0.0.1:5234/games"; // editor dev default — matches NetClient's fallback
        }
    }
}
