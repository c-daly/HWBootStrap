using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Rendering;
using HexWars.Engine;

namespace HexWars.Presentation
{
    /// <summary>
    /// Wires the engine to the scene: builds a new game (config + seeded board + two players),
    /// sets up soft ambient + a soft-shadow light + the starfield skybox, and renders board + units.
    /// Map-generation parameters are configurable here. Holds the live <see cref="State"/> — the seam
    /// the input/HUD layer drives. (DemoPieces seeds a couple of visible units/generators until input
    /// drives creation.)
    /// </summary>
    [RequireComponent(typeof(BoardRenderer))]
    public sealed class GameBootstrap : MonoBehaviour
    {
        [Header("Map generation")]
        public int Seed = 7;
        public int Width = 13;
        public int Height = 9;
        public int MaxElevation = 4;
        public int ZoneDepth = 3;
        [Range(0f, 1f)] public float FlatChance = 0.6f;

        [Header("Terrain weights (relative)")]
        public int PlainsWeight = 70;
        public int ForestWeight = 15;
        public int RoughWeight = 10;
        public int WaterWeight = 5;

        [Tooltip("Off = biomes are mechanically inert (every tile plays as flat plains); the board still renders varied terrain.")]
        public bool BiomesEnabled = false; // off for now

        [Tooltip("On = one action per turn (chess-like). Off = act with your whole army, then End Turn. Takes effect on a new game.")]
        public bool OneActionPerTurn = false;

        [Tooltip("On = enemy units/generators render only where your army has vision. Takes effect on a new game.")]
        public bool FogOfWar = false;

        [Header("Territory mode")]
        [Tooltip("On = territory mode: control gates deploy/build, claiming a hex is a turn-exclusive action. Takes effect on a new game.")]
        public bool TerritoryMode = false;
        [Tooltip("Points each player starts with in territory mode (you need these to claim/build).")]
        public int TerritoryStartingPoints = 40;

        [Header("Opponent")]
        [Tooltip("On = Player 2 is played by the AI; you play Player 1. (How a vs-AI game starts in a build.)")]
        public bool VsAI = false;
        [Tooltip("Easy = Random, Hard = Greedy (challenging).")]
        public AiLevel Difficulty = AiLevel.Hard;

        [Header("Demo")]
        public bool DemoPieces = true;

        [Header("Online")]
        [Tooltip("On = play online vs another browser through the HexWars server: wait for the server's start state instead of building a local game, and send moves to the server.")]
        public bool Networked = false;

        NetClient _net;
        /// <summary>The seat the server assigned this browser (null until seated, or if the room was full).</summary>
        public PlayerId? Seat { get; private set; }

        public GameState State { get; private set; }

        /// <summary>True while the title-screen demo game (AI vs AI, muted, gameplay UI hidden) is
        /// running. Real-game starters clear it. HUD/panels early-out on it — see the spec's
        /// suppression contract.</summary>
        public bool DemoMode { get; private set; }

        /// <summary>True while a started game's socket dropped and NetClient is retrying with backoff.
        /// GameHud reads this to show a persistent status line; OnNetReconnecting Toasts once per drop
        /// episode (not once per attempt).</summary>
        public bool Reconnecting { get; private set; }

        /// <summary>The setup/difficulty of the most recent LOCAL vs-AI game — null whenever the most
        /// recent game start was anything else (hotseat, online). Set only by StartLocalGame's vsAi
        /// path; cleared by NewGame(), StartNetGame(), and StartLocalGame's non-vsAi path so a stale
        /// vs-AI setup can never make RematchAvailable true for a hotseat or online game that
        /// started afterward.</summary>
        public GameSetup? LastLocalSetup { get; private set; }
        public AiLevel LastLocalAi { get; private set; }

        /// <summary>Game-over banner shows Rematch only for a local vs-AI game (spec §6).</summary>
        public bool RematchAvailable => LastLocalSetup.HasValue;

        /// <summary>Raised after the state changes (new game or applied command) so HUD can refresh.</summary>
        public event System.Action StateChanged;

        public ActionPresenter Presenter { get; private set; }

        void Start()
        {
            Presenter = GetComponent<ActionPresenter>() ?? gameObject.AddComponent<ActionPresenter>();
#if UNITY_WEBGL && !UNITY_EDITOR
            Networked = true; // the deployed browser build is always the online client
#endif
            if (Networked)
            {
                SetupEnvironment();          // light/skybox now; the board renders when the server deals the start state
                string room = RoomFromPageUrl();
                if (!string.IsNullOrEmpty(room)) StartNetGame(room, null); // opened via a shared ?room= link → join it
                else { StartDemo(); gameObject.AddComponent<TitleScreen>(); } // front door: demo + title menu
                return;
            }
            NewGame();
            if (VsAI)
            {
                var ai = GetComponent<AiOpponent>() ?? gameObject.AddComponent<AiOpponent>();
                ai.Level = Difficulty;
            }
        }

        public void NewGame()
        {
            EndDemo();
            TipsService.NewGame();
            LastLocalSetup = null; // hotseat isn't a vs-AI game — a stale Rematch target must not survive into it
            Presenter?.ResetQueue();
            SetupEnvironment();

            // damageFloor 1 matches GameFactory (the lobby path): a landed hit always deals at least 1,
            // so an attacker with damage > 0 never "does nothing" against high defense
            var config = TerritoryMode
                ? GameConfig.Default(biomesEnabled: BiomesEnabled,
                                     turnPolicy: OneActionPerTurn ? new OneActionPolicy() : null,
                                     winConditions: WinBy.Economy | WinBy.Annihilation,
                                     startingPoints: TerritoryStartingPoints,
                                     territoryMode: true, damageFloor: 1, fogOfWar: FogOfWar)
                : GameConfig.Default(biomesEnabled: BiomesEnabled,
                                     turnPolicy: OneActionPerTurn ? new OneActionPolicy() : null,
                                     damageFloor: 1, fogOfWar: FogOfWar);
            var genConfig = new BoardGenConfig(Width, Height, MaxElevation, ZoneDepth, FlatChance,
                                               PlainsWeight, ForestWeight, RoughWeight, WaterWeight);
            var board = new RandomBoardGenerator(genConfig).Generate(Seed);
            if (TerritoryMode)
            {
                board = board.WithControl(board.DeploymentZone(PlayerId.Player0), PlayerId.Player0);
                board = board.WithControl(board.DeploymentZone(PlayerId.Player1), PlayerId.Player1);
            }

            int nextId = 1;
            int startPts = TerritoryMode ? config.StartingPoints : 0;
            var p0 = BuildPlayer(board, PlayerId.Player0, startPts, ref nextId);
            var p1 = BuildPlayer(board, PlayerId.Player1, startPts, ref nextId);
            State = new GameState(board, config, new[] { p0, p1 }, PlayerId.Player0, 1, nextId);

            var renderer = GetComponent<BoardRenderer>();
            renderer.Render(board);
            renderer.RenderEntities(State, FogViewer());

            var rig = FindAnyObjectByType<CameraRig>();
            if (rig != null) rig.Frame(); // fit the camera once the board exists

            EventConsole.Clear();
            EventConsole.Report(State, null); // seed the scoreboard at game start
            StateChanged?.Invoke();
        }

        /// <summary>Apply a command through the engine; on success update state, re-render, notify.</summary>
        public bool TryApply(Command cmd)
        {
            if (Networked)
            {
                if (_net == null || State == null) return false;
                if (Seat.HasValue && cmd.Issuer != Seat.Value) return false; // only act as your own seat, on your turn
                _net.Send(NetProtocol.Cmd(cmd));
                return true; // the server validates and echoes APPLY (or REJECT) — state updates there
            }

            var result = GameEngine.Apply(State, cmd);
            if (!result.Success)
            {
                Debug.Log($"[HexWars] {cmd.GetType().Name} rejected: {result.Reason}");
                if (!DemoMode) Toast.Show(Friendly(result.Reason.ToString()));
                return false;
            }
            var prev = State;
            State = result.NewState;
            if (!DemoMode) EventConsole.Report(State, CombatLog.Diff(prev, State, FogViewer()));
            Presenter.Enqueue(prev, cmd, State, IsLocalCommand(cmd));
            if (!DemoMode) CheckFirstBounty(prev, cmd);
            StateChanged?.Invoke();
            return true;
        }

        /// <summary>Spec §6's "first bounty earned" reveal: a kill (AttackUnit that increases the
        /// attacker's Points — CombatResolver awards bounty only on a kill, never a plain hit) issued by
        /// a seat this human controls fires the tip once per game, CTA drawing attention to the Designer.</summary>
        void CheckFirstBounty(GameState prev, Command cmd)
        {
            if (!(cmd is AttackUnit atk) || !IsLocalCommand(cmd)) return;
            int gained = State.Player(atk.Issuer).Points - prev.Player(atk.Issuer).Points;
            if (gained <= 0) return;
            TipsService.Show("first-bounty",
                $"You earned {gained} points. A wall? A sniper? Eyes that see everything? Design your answer.",
                cta: "Design your answer", onCta: OpenDesigner);
        }

        void OpenDesigner() => FindAnyObjectByType<DesignPanel>()?.Highlight();

        /// <summary>Connect to the server for a room. <paramref name="setupWire"/> is non-null only for
        /// the host (carries the lobby picks); a joiner passes null and gets the host's game.
        /// <paramref name="isPrivate"/> keeps the room out of the public browser list.</summary>
        public void StartNetGame(string room, string setupWire, bool isPrivate = false)
        {
            EndDemo();
            TipsService.NewGame();
            LastLocalSetup = null; // online isn't a vs-AI game — same reasoning as NewGame() above
            // the demo's state must not linger: panels dismiss on (State != null && !DemoMode), and
            // the authoritative state arrives later via START — until then there is no game here
            State = null;
            StateChanged?.Invoke();
            Networked = true;
            if (_net != null) { Destroy(_net); _net = null; }
            _net = gameObject.AddComponent<NetClient>();
            _net.Connect(this, room, setupWire, isPrivate);
        }

        /// <summary>Host changed their mind while waiting: drop the socket and seat. State stays as it
        /// was (null before START), so the title/demo behind the form is untouched.</summary>
        public void CancelHosting()
        {
            if (_net != null) { Destroy(_net); _net = null; }
            Seat = null;
            Reconnecting = false;
        }

        /// <summary>Start a single-machine game from the lobby's setup (no server). <paramref name="vsAi"/>
        /// adds an AI opponent on Player 2 at <paramref name="level"/>; otherwise it's a local hotseat.
        /// Used by the lobby's vs-AI option.</summary>
        public void StartLocalGame(GameSetup setup, bool vsAi, AiLevel level = AiLevel.Hard)
        {
            EndDemo();
            TipsService.NewGame();
            Presenter?.ResetQueue();
            Networked = false; // play locally — TryApply applies here instead of going to the server
            State = GameFactory.Build(setup);
            var renderer = GetComponent<BoardRenderer>();
            renderer.Render(State.Board);
            renderer.RenderEntities(State, FogViewer());
            FindAnyObjectByType<CameraRig>()?.Frame();
            EventConsole.Clear();
            EventConsole.Report(State, null);
            StateChanged?.Invoke();
            SoundManager.StartAmbience();
            if (vsAi)
            {
                // destroy-before-add: Rematch() re-enters here with the previous game's AiOpponent
                // still alive — an unconditional AddComponent would stack a second (then Nth) live
                // AI all driving Player1 (doubled action rate), and ReturnToMenu destroys only one,
                // leaking a phantom AI into the next game. Guarding here protects every caller.
                var oldAi = GetComponent<AiOpponent>();
                if (oldAi != null) Destroy(oldAi);
                var ai = gameObject.AddComponent<AiOpponent>();
                ai.Level = level;
                LastLocalSetup = setup;
                LastLocalAi = level;
            }
            else
            {
                // defensive: a hotseat game via this path isn't a vs-AI game either — without this,
                // a future lobby hotseat option would leave RematchAvailable wrongly true
                LastLocalSetup = null;
            }
        }

        /// <summary>Game-over banner's Rematch button: same setup, fresh seed, instant restart. A no-op
        /// if the last game wasn't local vs-AI (defensive — the banner only shows the button when
        /// RematchAvailable is already true, so this guard should never actually trigger).</summary>
        public void Rematch()
        {
            if (!LastLocalSetup.HasValue) return;
            var s = LastLocalSetup.Value;
            var reseeded = new GameSetup(s.Mode, s.Width, s.Height, s.StartingPoints,
                                         UnityEngine.Random.Range(1, 99999), s.ArmySize, s.Brutes, s.Strikers,
                                         s.Snipers, s.TurnActions, s.Fog);
            StartLocalGame(reseeded, true, LastLocalAi);
        }

        /// <summary>The title screen's living background: a muted Greedy-vs-Random match on a fresh
        /// standard map, driven by SpectatorDriver through the normal presenter path (camera glides
        /// and all), with every gameplay UI surface suppressed via <see cref="DemoMode"/>.
        /// Greedy-vs-Greedy is deliberately avoided: mirror matches draw ~93% as standoffs.</summary>
        public void StartDemo()
        {
            Presenter?.ResetQueue();
            Networked = false;
            DemoMode = true;
            SoundManager.Muted = true;
            SoundManager.StartTitleMusic();
            SoundManager.StopAmbience();
            var setup = new GameSetup(GameMode.Annihilation, 11, 8, 0,
                                      UnityEngine.Random.Range(1, 99999), 5, 2, 2, 1, 3);
            State = GameFactory.Build(setup);
            var renderer = GetComponent<BoardRenderer>();
            renderer.Render(State.Board);
            renderer.RenderEntities(State, FogViewer());
            FindAnyObjectByType<CameraRig>()?.Frame();
            EventConsole.Clear();
            if (GetComponent<SpectatorDriver>() == null) gameObject.AddComponent<SpectatorDriver>();
            StateChanged?.Invoke();
        }

        /// <summary>Leave demo mode before a real game starts: drop the spectator driver, restore
        /// sound, and give input back (the driver had set ReadOnly).</summary>
        void EndDemo()
        {
            if (!DemoMode) return;
            DemoMode = false;
            SoundManager.Muted = false;
            SoundManager.StopTitleMusic();
            var driver = GetComponent<SpectatorDriver>();
            if (driver != null) Destroy(driver);
            var input = FindAnyObjectByType<UnitInputController>();
            if (input != null) input.ReadOnly = false;
            var barracks = FindAnyObjectByType<BarracksPanel>();
            if (barracks != null) barracks.ReadOnly = false;
        }

        static string RoomFromPageUrl()
        {
            string page = Application.absoluteURL;
            if (string.IsNullOrEmpty(page)) return null;
            int q = page.IndexOf('?');
            if (q < 0) return null;
            foreach (var kv in page.Substring(q + 1).Split('&'))
            {
                var p = kv.Split('=');
                if (p.Length == 2 && p[0] == "room") return p[1];
            }
            return null;
        }

        /// <summary>Tear down the current game and return to the lobby: disconnect the socket, drop
        /// the AI and seat, clear the state (the lobby dismisses itself whenever a state exists, so
        /// a null state is what lets it come back). The next created game rebuilds everything.</summary>
        public void ReturnToMenu()
        {
            Presenter?.ResetQueue();
            GameOverBanner.Dismiss();
            if (_net != null) { Destroy(_net); _net = null; }
            Seat = null;
            Reconnecting = false;
            var ai = GetComponent<AiOpponent>();
            if (ai != null) Destroy(ai);
            State = null;
            StateChanged?.Invoke();
            GetComponent<SetupForm>()?.Close();
            GetComponent<GameBrowser>()?.Close();
            StartDemo();
            TitleScreen.Reopen(this);
        }

        /// <summary>Whose vision the fog renders: this browser's seat online, the human's seat vs AI,
        /// or the active player in hotseat (vision hands over with the turn). Null when fog is off
        /// or there is no seated human (spectators see everything).</summary>
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

        /// <summary>Local = issued by a seat this human controls: your seat online; any non-AI seat
        /// offline (hotseat = both) — but never local while spectating, since no seat is "yours" there
        /// (spectators watch every action paced and animated, per spec). Local actions play immediately
        /// with no pacing gap.</summary>
        bool IsLocalCommand(Command cmd)
        {
            if (Networked) return Seat.HasValue && cmd.Issuer == Seat.Value;
            if (GetComponent<SpectatorDriver>() != null) return false;
            var ai = GetComponent<AiOpponent>();
            return ai == null || cmd.Issuer != ai.AiSeat;
        }

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

        // ---- server callbacks (online mode), invoked by NetClient ----

        internal void OnNetSeat(PlayerId seat) { Seat = seat; StateChanged?.Invoke(); }

        internal void OnNetSeatFull()
        {
            Toast.Show("That game is unavailable — it may have filled, or the code is wrong.");
            CancelHosting();
            // don't stack the title over an open sub-screen: the browser's own poll refreshes its
            // list, and the toast above is the feedback for a form's failed host/join attempt
            if (GetComponent<GameBrowser>() == null && GetComponent<SetupForm>() == null)
                TitleScreen.Reopen(this);
        }

        /// <summary>The socket died before a match began (server down / network drop while hosting or
        /// joining). Mid-game drops are the reconnect follow-up (audit U2) — pre-game, a toast plus the
        /// waiting screen's Cancel is normally the whole story. But a shared <c>?room=</c> link auto-joins
        /// straight from <see cref="Start"/> with none of SetupForm/GameBrowser/TitleScreen ever created
        /// (see Start's Networked branch) — if THAT connection fails (bad/expired code, server down), none
        /// of those exist to catch it, and the player is stranded on a blank screen forever with no way
        /// back (audit I4). Self-heal the same way SEAT FULL already does: reopen the title screen, whose
        /// own self-heal restores the demo + music.</summary>
        internal void OnNetClosed()
        {
            if (Networked && State == null && _net != null)
                Toast.Show("Connection lost — check the link and try again.");
            GetComponent<SetupForm>()?.OnConnectionLost();
            if (GetComponent<SetupForm>() == null && GetComponent<GameBrowser>() == null && GetComponent<TitleScreen>() == null)
                TitleScreen.Reopen(this);
        }

        /// <summary>A started game's socket dropped and NetClient is retrying with backoff. Called once
        /// per attempt (so a persistent status line can show progress); the Toast only fires transitioning
        /// INTO reconnecting, not on every retry, matching spec §7 ("every attempt updates the status
        /// line" — GameHud's banner text is that status line).</summary>
        internal void OnNetReconnecting(int attempt)
        {
            if (!Reconnecting) Toast.Show("Connection lost — reconnecting…", new Color(0.42f, 0.34f, 0.12f, 0.94f));
            Reconnecting = true;
            StateChanged?.Invoke();
        }

        /// <summary>The socket reopened. The server re-deals START right behind this (OnNetStart also
        /// clears Reconnecting, redundantly, so arrival order between the two can never leave it stuck).</summary>
        internal void OnNetReconnected()
        {
            Reconnecting = false;
            StateChanged?.Invoke();
        }

        /// <summary>The server dealt the authoritative start state — load and render it. The payload is
        /// the room's start state PLUS every command accepted since (see MatchHub) — a fresh join has an
        /// empty command list, but a reconnect's re-deal does not, so it must be fast-forwarded through
        /// the same engine that produced it rather than treated as a start state on its own: ReplayFile's
        /// start-state encoding assumes full-health units and omits per-turn tracking (fine for a FRESH
        /// deal, corrupting for a mid-game one). No per-command presentation here — this is a silent
        /// resync, not a replay to watch.</summary>
        internal void OnNetStart(string startStateText)
        {
            Reconnecting = false;      // a START re-deal (fresh join OR a reconnect) always means we're live
            Presenter?.ResetQueue();
            var data = ReplayFile.Read(startStateText);
            State = data.Start;
            foreach (var cmd in data.Commands)
            {
                var result = GameEngine.Apply(State, cmd);
                if (result.Success) State = result.NewState;
                else Debug.LogError("[Net] re-deal fast-forward: a logged command failed to reapply — " + result.Reason);
            }
            var renderer = GetComponent<BoardRenderer>();
            renderer.Render(State.Board);
            renderer.RenderEntities(State, FogViewer());
            FindAnyObjectByType<CameraRig>()?.Frame();
            EventConsole.Clear();
            EventConsole.Report(State, null);
            StateChanged?.Invoke();
            SoundManager.StartAmbience();
        }

        /// <summary>A validated move from the server — apply it locally (same engine, so identical result).</summary>
        internal void OnNetApply(Command cmd)
        {
            var result = GameEngine.Apply(State, cmd);
            if (!result.Success) { Debug.LogWarning("[Net] server move rejected locally: " + result.Reason); return; }
            var prev = State;
            State = result.NewState;
            EventConsole.Report(State, CombatLog.Diff(prev, State, FogViewer()));
            Presenter.Enqueue(prev, cmd, State, IsLocalCommand(cmd));
            CheckFirstBounty(prev, cmd);
            // The server confirmed THIS client's own CreateUnit (final review N2): the success cues
            // DesignPanel.OnCreate deliberately skips online (its optimistic TryApply `true` isn't a
            // verdict) fire here instead, on the real APPLY — the same sound + name-box clear the
            // local path plays synchronously. The opponent's creates stay silent, as before.
            if (cmd is CreateUnit && IsLocalCommand(cmd))
            {
                SoundManager.Play(SoundKind.Design);
                FindAnyObjectByType<DesignPanel>()?.ConfirmCreate();
            }
            StateChanged?.Invoke();
        }

        internal void OnNetReject(string reason)
        {
            Debug.Log("[Net] move rejected: " + reason);
            Toast.Show(Friendly(reason));
            if (State != null) GetComponent<BoardRenderer>().RenderEntities(State, FogViewer()); // snap optimistic UI back to truth
        }

        /// <summary>Turn a RejectionReason name into a plain-language explanation for the player.</summary>
        static string Friendly(string reason) => reason switch
        {
            "InsufficientPoints"    => "Not enough points for that.",
            "HexNotControlled"      => "You can only build or deploy on your own territory.",
            "OutsideDeploymentZone" => "Deploy inside your own starting area.",
            "TileOccupied"          => "That hex is already taken.",
            "TileImpassable"        => "A unit can't go there.",
            "NotYourTurn"           => "It's not your turn.",
            "UnitAlreadyMoved"      => "That unit has no movement left this turn.",
            "MovementEndedByAttack" => "Attacking ends movement for this unit.",
            "UnitAlreadyAttacked"   => "That unit already attacked this turn.",
            "OutOfMovementRange"    => "Too far — not enough movement left this turn.",
            "MustClaimFirst"        => "Claiming has to be your turn's first action.",
            "AlreadyControlled"     => "You already control that hex.",
            "NoUnitOnHex"           => "You need one of your units on that hex.",
            "TemplateNotFound"      => "Pick a unit to deploy first.",
            "GameAlreadyOver"       => "The game is over.",
            _                        => "That move isn't allowed right now.",
        };

        PlayerState BuildPlayer(Board board, PlayerId id, int startingPoints, ref int nextId)
        {
            if (!DemoPieces)
                return new PlayerState(id, startingPoints);

            var flatZone = board.DeploymentZone(id)
                .Where(c => board.TileAt(c).Elevation == 0)
                .OrderBy(c => c.Q).ThenBy(c => c.R)
                .ToList();

            // a few distinct builds so the role icons + size-by-points are visible
            var demos = new[]
            {
                new UnitStats(health: 7, damage: 2, defense: 2, movement: 3, verticalMovement: 2, range: 1, rangeArc: 1, vision: 2, visionArc: 1), // Brute
                new UnitStats(health: 2, damage: 6, defense: 0, movement: 3, verticalMovement: 2, range: 2, rangeArc: 1, vision: 3, visionArc: 1), // Striker
                new UnitStats(health: 2, damage: 2, defense: 0, movement: 2, verticalMovement: 2, range: 6, rangeArc: 1, vision: 4, visionArc: 1), // Sniper
                new UnitStats(health: 2, damage: 0, defense: 0, movement: 4, verticalMovement: 3, range: 0, rangeArc: 0, vision: 7, visionArc: 2), // Spotter
            };

            // start with an army (the only resource); no generators, no starting points —
            // the only way to earn points (for reinforcements) is bounty from kills
            var units = new List<Unit>();
            int placed = 0;
            for (; placed < demos.Length && placed < flatZone.Count; placed++)
                units.Add(new Unit(nextId++, id, demos[placed], flatZone[placed], 0));

            return new PlayerState(id, startingPoints, unitsOnBoard: units);
        }

        static Cubemap _reflection;

        void SetupEnvironment()
        {
            RenderSettings.ambientMode = AmbientMode.Flat;
            RenderSettings.ambientLight = new Color(0.40f, 0.43f, 0.52f);
            RenderSettings.skybox = StarfieldSkybox();                 // dark starfield stays the visible background
            RenderSettings.defaultReflectionMode = UnityEngine.Rendering.DefaultReflectionMode.Custom;
            RenderSettings.customReflectionTexture = BrightReflection(); // ...but metal reflects an even bright env
            RenderSettings.reflectionIntensity = 1f;
            DynamicGI.UpdateEnvironment();

            // a stray fill from an earlier setup would re-add a second hotspot — remove it
            var stray = GameObject.Find("FillLight");
            if (stray != null) { if (Application.isPlaying) Destroy(stray); else DestroyImmediate(stray); }

            // one gentle light: brightness/shine comes from the even reflection, not light hotspots
            EnsureLight("KeyLight", new Color(1f, 0.98f, 0.94f), 0.9f, Quaternion.Euler(45f, -40f, 0f), LightShadows.Soft, 0.35f);
        }

        static Cubemap BrightReflection()
        {
            if (_reflection != null) return _reflection;
            const int s = 32;
            var cm = new Cubemap(s, TextureFormat.RGBA32, false);
            var tint = new Color(0.98f, 0.98f, 1f);
            for (int f = 0; f < 6; f++)
            {
                var face = (CubemapFace)f;
                // per-face base brightness gives the 6 hex facets directional variation = metallic sheen
                float b = face switch
                {
                    CubemapFace.PositiveY => 1.0f,   // up: bright
                    CubemapFace.NegativeY => 0.18f,  // down: dark
                    CubemapFace.PositiveX => 0.92f,
                    CubemapFace.NegativeX => 0.30f,
                    CubemapFace.PositiveZ => 0.70f,
                    CubemapFace.NegativeZ => 0.48f,
                    _ => 0.6f,
                };
                var cols = new Color[s * s];
                for (int y = 0; y < s; y++)
                    for (int x = 0; x < s; x++)
                    {
                        float v = Mathf.Clamp01(b * (0.8f + 0.2f * (y / (float)(s - 1))));
                        cols[y * s + x] = new Color(tint.r * v, tint.g * v, tint.b * v);
                    }
                cm.SetPixels(cols, face);
            }
            cm.Apply();
            _reflection = cm;
            return cm;
        }

        static void EnsureLight(string name, Color color, float intensity, Quaternion rot, LightShadows shadows, float shadowStrength)
        {
            var go = GameObject.Find(name);
            if (go == null) go = new GameObject(name);
            var l = go.GetComponent<Light>();
            if (l == null) l = go.AddComponent<Light>();
            l.type = LightType.Directional;
            l.color = color;
            l.intensity = intensity;
            l.shadows = shadows;
            l.shadowStrength = shadowStrength;
            go.transform.rotation = rot;
        }

        static Material StarfieldSkybox()
        {
            var tex = new Texture2D(1024, 512, TextureFormat.RGB24, false);
            var px = new Color32[1024 * 512];
            var space = new Color32(6, 7, 17, 255);
            for (int i = 0; i < px.Length; i++) px[i] = space;
            var rng = new System.Random(3);
            for (int i = 0; i < 750; i++)
            {
                int x = rng.Next(1024), y = rng.Next(512);
                byte b = (byte)rng.Next(120, 256);
                px[y * 1024 + x] = new Color32(b, b, b, 255);
            }
            tex.SetPixels32(px);
            tex.Apply();

            var sky = new Material(Shader.Find("Skybox/Panoramic"));
            if (sky.HasProperty("_MainTex")) sky.SetTexture("_MainTex", tex);
            if (sky.HasProperty("_Exposure")) sky.SetFloat("_Exposure", 1f);
            return sky;
        }
    }
}
