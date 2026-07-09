using UnityEngine;
using UnityEngine.UI;
using HexWars.Engine;

namespace HexWars.Presentation
{
    /// <summary>
    /// Top banner showing the active player, points, round, barracks count, and — under a paced turn
    /// structure — the actions left this turn, with an End Turn button. Announces turn handovers with a
    /// toast so an auto-pass (K actions spent) is unmistakable. Builds its own canvas; ensures an
    /// EventSystem exists so UI buttons get clicks.
    /// </summary>
    public sealed class GameHud : MonoBehaviour
    {
        static readonly Color P0ToastBlue = new Color(0.10f, 0.30f, 0.45f, 0.94f);
        static readonly Color P1ToastRed = new Color(0.42f, 0.16f, 0.16f, 0.94f);
        static readonly Color EndTurnIdle = new Color(0.20f, 0.34f, 0.55f, 1f);
        static readonly Color EndTurnUrge = new Color(0.16f, 0.52f, 0.28f, 1f); // nothing left to do

        GameBootstrap _game;
        GameObject _canvasGo;
        Text _banner;
        Image _endBtn;
        Text _endBtnText;      // "End Turn" in play; "Main menu" once the game is over
        PlayerId? _lastActive; // last seen active player, to detect handovers (incl. auto-pass)
        bool _wasOver;         // so the game-over announcement fires exactly once per game

        void Start()
        {
            UiKit.EnsureEventSystem();
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

        void Build()
        {
            var canvasGo = UiKit.Canvas("HudCanvas", UiKit.OrderHud, transform);
            _canvasGo = canvasGo;

            var bar = new GameObject("Banner");
            bar.transform.SetParent(canvasGo.transform, false);
            bar.AddComponent<Image>().color = new Color(UiKit.Bg.r, UiKit.Bg.g, UiKit.Bg.b, 0.88f);
            var brt = bar.GetComponent<RectTransform>();
            brt.anchorMin = new Vector2(0f, 1f);
            brt.anchorMax = new Vector2(1f, 1f);
            brt.pivot = new Vector2(0.5f, 1f);
            brt.sizeDelta = new Vector2(0f, 46f);
            brt.anchoredPosition = Vector2.zero;

            var textGo = new GameObject("BannerText");
            textGo.transform.SetParent(bar.transform, false);
            _banner = textGo.AddComponent<Text>();
            _banner.font = UiKit.Font();
            _banner.fontSize = 20;
            _banner.color = Color.white;
            _banner.alignment = TextAnchor.MiddleLeft;
            var trt = _banner.GetComponent<RectTransform>();
            trt.anchorMin = Vector2.zero;
            trt.anchorMax = Vector2.one;
            trt.offsetMin = new Vector2(160f, 0f); // leave room for the End Turn button on the left
            trt.offsetMax = new Vector2(-16f, 0f);

            var btn = new GameObject("EndTurnButton");
            btn.transform.SetParent(bar.transform, false);
            _endBtn = btn.AddComponent<Image>();
            _endBtn.sprite = UiKit.Rounded();
            _endBtn.type = Image.Type.Sliced;
            _endBtn.color = EndTurnIdle;
            btn.AddComponent<Button>().onClick.AddListener(OnEndTurn);
            var rt = btn.GetComponent<RectTransform>();
            rt.anchorMin = new Vector2(0f, 0.5f);
            rt.anchorMax = new Vector2(0f, 0.5f);
            rt.pivot = new Vector2(0f, 0.5f);
            rt.sizeDelta = new Vector2(140f, 34f);
            rt.anchoredPosition = new Vector2(8f, 0f); // left side — clear of the right-side log/speed controls

            var btnTextGo = new GameObject("Text");
            btnTextGo.transform.SetParent(btn.transform, false);
            var btnText = btnTextGo.AddComponent<Text>();
            _endBtnText = btnText;
            btnText.font = UiKit.Font();
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
            if (_game == null || _game.State == null) return;
            // once the game is over this button IS the way back to the title — the centre banner is
            // dismissible ("tap outside to look at the board"), so it can't be the only exit
            if (_game.State.IsGameOver) { _game.ReturnToMenu(); return; }
            _game.TryApply(new EndTurn(_game.State.ActivePlayer));
        }

        void Refresh()
        {
            if (_game == null || _game.State == null) return;
            if (_canvasGo != null && _canvasGo.activeSelf == _game.DemoMode)
                _canvasGo.SetActive(!_game.DemoMode);
            if (_game.DemoMode) return;
            var s = _game.State;

            if (s.IsGameOver)
            {
                ShowGameOver(s);
                return;
            }
            if (_wasOver) // a new game started while the old result was still up
            {
                _wasOver = false;
                GameOverBanner.Dismiss();
                if (_endBtn != null) _endBtn.gameObject.SetActive(true);
                if (_endBtnText != null) _endBtnText.text = "End Turn";
            }

            var p = s.Player(s.ActivePlayer);
            bool p0 = s.ActivePlayer == PlayerId.Player0;
            int who = p0 ? 1 : 2;
            _banner.color = p0 ? new Color(0.4f, 0.8f, 1f) : new Color(1f, 0.45f, 0.45f);

            // What can the active player still do? Drives the "you're not stuck, you're done" signal:
            // without it, a turn with no legal move/attack left just feels unresponsive.
            bool localHuman = LocalHumanActs(s);
            bool anyAction = false, anyCombat = false;
            if (localHuman && !s.IsGameOver)
                foreach (var m in LegalMoves.For(s))
                {
                    if (m is EndTurn) continue;
                    anyAction = true;
                    if (m is MoveUnit || m is AttackUnit) { anyCombat = true; break; }
                }

            string pace = "";
            if (s.Config.TurnPolicy.ActionsPerTurn is int k)
            {
                int left = s.Config.TurnPolicy.RemainingActions(s) ?? k;
                pace = $"     Actions left {left}/{k}";
                if (localHuman && !s.IsGameOver && left > 0 && !anyCombat && anyAction)
                    pace += "  (no unit can act)";
            }
            string done = localHuman && !s.IsGameOver && !anyAction
                ? "     Nothing left to do - press End Turn"
                : "";

            // the comeback rule keeps a unitless side alive while it can afford a redeploy — say so,
            // or (especially under fog) a wiped board reads as "the game ended and nothing happened"
            string armies = "";
            foreach (var pl in s.Players)
            {
                int alive = 0;
                foreach (var u in pl.UnitsOnBoard) if (u.IsAlive) alive++;
                if (alive == 0)
                    armies += $"     P{((int)pl.Id) + 1} has no army (can still deploy)";
            }

            _banner.text = s.Config.TerritoryMode
                ? $"P{who}'s turn{pace}{done}{armies}     Round {s.Round}     " +
                  $"P1 {Stat(s, PlayerId.Player0)}   |   P2 {Stat(s, PlayerId.Player1)}"
                : $"Player {who}'s turn  (move {(p0 ? "cyan" : "red")}){pace}{done}{armies}     {p.Points} pts     Round {s.Round}     Barracks {p.Barracks.Count}";
            if (_game.Reconnecting) _banner.text = "⚠ Connection lost — reconnecting…     " + _banner.text;

            if (_endBtn != null) _endBtn.color = done.Length > 0 ? EndTurnUrge : EndTurnIdle;

            AnnounceTurnIfChanged(s);
        }

        /// <summary>The moment the game ends: result in the HUD banner, the End Turn button becomes a
        /// persistent Main-menu button, and (once) the big centre announcement. The HUD button must
        /// survive the banner's dismissal — otherwise inspecting the final board strands the player
        /// with no way back to the title.</summary>
        void ShowGameOver(GameState s)
        {
            bool p0Won = s.Winner == PlayerId.Player0;
            _banner.color = s.Winner == null ? Color.white
                          : p0Won ? new Color(0.4f, 0.8f, 1f) : new Color(1f, 0.45f, 0.45f);
            string result = ResultText(s);
            _banner.text = $"GAME OVER   {result}     Round {s.Round}     " +
                           $"P1 {Stat(s, PlayerId.Player0)}   |   P2 {Stat(s, PlayerId.Player1)}";
            if (_endBtn != null)
            {
                _endBtn.gameObject.SetActive(true);
                _endBtn.color = EndTurnIdle;
            }
            if (_endBtnText != null) _endBtnText.text = "Main menu";

            if (!_wasOver)
            {
                _wasOver = true;
                var accent = s.Winner == null ? new Color(0.25f, 0.27f, 0.33f, 0.96f)
                           : p0Won ? P0ToastBlue : P1ToastRed;
                StartCoroutine(ShowBannerWhenQuiet(result.ToUpperInvariant(), HowText(s), accent));
            }
        }

        /// <summary>Hold the big banner until the final action's animation lands — the killing blow
        /// is the climax; the banner must not cover it mid-flight.</summary>
        System.Collections.IEnumerator ShowBannerWhenQuiet(string title, string how, Color accent)
        {
            var presenter = _game != null ? _game.Presenter : null;
            while (presenter != null && presenter.IsBusy) yield return null;
            GameOverBanner.Show(title, how, accent, onMainMenu: () => _game.ReturnToMenu());
        }

        static string ResultText(GameState s)
        {
            if (s.Winner == null) return "Draw";
            return s.Winner == PlayerId.Player0 ? "Player 1 wins" : "Player 2 wins";
        }

        /// <summary>Which win condition decided it, derived from the final state.</summary>
        static string HowText(GameState s)
        {
            if (s.Winner == null) return "Round cap reached with equal scores";
            var loser = s.Winner == PlayerId.Player0 ? PlayerId.Player1 : PlayerId.Player0;
            if (WinCheck.IsEliminated(s, loser)) return "The enemy army was wiped out";
            if ((s.Config.WinConditions & WinBy.Economy) != 0
                && s.Player(s.Winner.Value).Points >= s.Config.EconomyWinThreshold)
                return $"Banked {s.Config.EconomyWinThreshold} points (economy win)";
            return "Higher score at the round cap";
        }

        /// <summary>Is the seat that's about to act driven by a human on this machine? (Hotseat: both;
        /// vs AI: only the human's seat; online: only this browser's seat.)</summary>
        bool LocalHumanActs(GameState s)
        {
            if (_game.Seat.HasValue) return s.ActivePlayer == _game.Seat.Value;
            var ai = FindAnyObjectByType<AiOpponent>();
            return ai == null || s.ActivePlayer != ai.AiSeat;
        }

        /// <summary>Toast the handover whenever the active player flips — the only reliable signal when a
        /// paced turn auto-passes mid-flow. Skips the AI's own turn (it plays out on screen anyway).</summary>
        void AnnounceTurnIfChanged(GameState s)
        {
            if (_lastActive == s.ActivePlayer) return;
            bool first = _lastActive == null;
            _lastActive = s.ActivePlayer;
            if (first || s.IsGameOver) return;

            var ai = FindAnyObjectByType<AiOpponent>();
            if (ai != null && s.ActivePlayer == ai.AiSeat) return;

            string who;
            if (_game.Seat.HasValue) who = s.ActivePlayer == _game.Seat.Value ? "Your turn" : "Opponent's turn";
            else if (ai != null) who = "Your turn";
            else who = s.ActivePlayer == PlayerId.Player0 ? "Player 1's turn" : "Player 2's turn";

            if (s.Config.TurnPolicy.ActionsPerTurn is int k)
                who += k == 1 ? "  (1 action)" : $"  ({k} actions)";

            // top slot + short life: turn flips are frequent under a paced game and must never
            // cover the battlefield (or the damage popups rising from it)
            Toast.Show(who, s.ActivePlayer == PlayerId.Player0 ? P0ToastBlue : P1ToastRed,
                       top: true, seconds: 1.6f);
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
    }
}
