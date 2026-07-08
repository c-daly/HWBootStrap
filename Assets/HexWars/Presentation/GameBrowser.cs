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
        bool _fetchFailed;

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

        void Build()
        {
            UiKit.EnsureEventSystem();
            _canvasGo = UiKit.Canvas("BrowserCanvas", UiKit.OrderMenu, transform);

            var panel = UiKit.Panel(_canvasGo.transform, "Panel", UiKit.Surface).gameObject;
            var prt = panel.GetComponent<RectTransform>();
            prt.anchorMin = prt.anchorMax = new Vector2(0.5f, 0.5f);
            prt.pivot = new Vector2(0.5f, 0.5f);
            prt.sizeDelta = new Vector2(760f, 640f);
            prt.anchoredPosition = Vector2.zero;

            UiKit.Label(panel.transform, "Open Games", 0f, -24f, 760f, 36f, UiKit.SizeTitle, TextAnchor.MiddleCenter);
            UiKit.Button(panel.transform, "Back", -330f, -26f, 90f, 34f,
                         () => { Close(); TitleScreen.Reopen(_game); }, UiKit.ButtonStyle.Secondary, UiKit.SizeBody);
            UiKit.Button(panel.transform, "Refresh", 320f, -26f, 110f, 34f,
                         () => StartCoroutine(FetchOnce()), UiKit.ButtonStyle.Secondary, UiKit.SizeBody);

            _status = UiKit.Label(panel.transform, "Loading…", 0f, -70f, 700f, 26f,
                                  UiKit.SizeCaption, TextAnchor.MiddleCenter, UiKit.TextFaint);

            var listGo = new GameObject("List");
            listGo.transform.SetParent(panel.transform, false);
            var lrt = listGo.AddComponent<RectTransform>();
            UiKit.SetRect(lrt, 0f, -100f, 720f, 520f);
            _listRoot = listGo.transform;
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
                    _fetchFailed = true;
                    if (_status != null) _status.text = "Can't reach the server — retrying…";
                    yield break;
                }
                GamesDto dto = null;
                try { dto = JsonUtility.FromJson<GamesDto>(req.downloadHandler.text); }
                catch (Exception) { /* malformed = treat as fetch failure */ }
                if (dto == null || dto.games == null)
                {
                    _fetchFailed = true;
                    if (_status != null) _status.text = "Can't reach the server — retrying…";
                    yield break;
                }
                _fetchFailed = false;
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
                _status.text = "No open games right now — host one!";
                UiKit.Button(_listRoot, "Host Game", 0f, -40f, 260f, 48f, () =>
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

        float BuildRow(GameDto g, float y, bool expanded)
        {
            string age = g.ageSeconds < 60 ? $"{g.ageSeconds}s" : $"{g.ageSeconds / 60}m";
            string summary = $"{g.code}   ·   {g.mode} · {g.width}×{g.height}{(g.fog ? " · Fog" : "")}" +
                             $" · {(g.pace <= 0 ? "whole army" : g.pace + " acts/turn")} · {g.army} units · {age} ago";
            var code = g.code;
            var row = UiKit.Button(_listRoot, summary, 0f, y, 700f, 42f, () =>
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
                UiKit.SetRect(card.GetComponent<RectTransform>(), 0f, y, 700f, 96f);
                UiKit.Label(card.transform,
                            $"Mode {g.mode}    Map {g.width}×{g.height}    Fog {(g.fog ? "on" : "off")}\n" +
                            $"Pace {(g.pace <= 0 ? "whole army" : g.pace + " actions/turn")}    Army {g.army} units    Waiting {age}",
                            -80f, -12f, 520f, 72f, UiKit.SizeBody, TextAnchor.UpperLeft, UiKit.TextDim);
                UiKit.Button(card.transform, "Join", 260f, -26f, 140f, 44f, () =>
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
