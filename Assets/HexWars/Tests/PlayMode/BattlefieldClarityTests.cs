using System.Collections;
using System.Linq;
using HexWars.Engine;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
using UnityEngine.UI;
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.LowLevel;

namespace HexWars.Presentation.PlayModeTests
{
    public class BattlefieldClarityTests
    {
        GameObject _host; GameBootstrap _game; UnitInputController _input; BoardRenderer _board;
        [UnitySetUp]
        public IEnumerator SetUp()
        {
            _host = new GameObject("Battlefield clarity fixture", typeof(BoardRenderer), typeof(TokenStore), typeof(GameBootstrap));
            _game = _host.GetComponent<GameBootstrap>(); _game.enabled = false;
            var presenter = _host.AddComponent<ActionPresenter>();
            typeof(GameBootstrap).GetProperty("Presenter").SetValue(_game, presenter);
            var tiles = Enumerable.Range(0, 6).SelectMany(q => Enumerable.Range(0, 3)
                .Select(r => new Tile(new HexCoord(q, r), 0, (TerrainType)(q % 4)))).ToArray();
            var stats = new UnitStats(7, 2, 0, 4, 1, 3, 0, 6, 0);
            var mine = new Unit(1, PlayerId.Player0, stats, new HexCoord(0, 1), 0, "Scout", "atlas-01");
            var near = new Unit(2, PlayerId.Player1, stats, new HexCoord(2, 1), 0, "Near", "atlas-01");
            var far = new Unit(3, PlayerId.Player1, stats, new HexCoord(5, 1), 0, "Far", "atlas-01");
            var state = new GameState(new Board(tiles), GameConfig.Default(biomesEnabled: false), new[] {
                new PlayerState(PlayerId.Player0, 30, unitsOnBoard: new[] { mine }),
                new PlayerState(PlayerId.Player1, 30, unitsOnBoard: new[] { near, far }) }, PlayerId.Player0, 1, 4);
            typeof(GameBootstrap).GetProperty("State").SetValue(_game, state);
            _board = _host.GetComponent<BoardRenderer>(); _board.Render(state.Board); _board.RenderEntities(state);
            _input = _host.AddComponent<UnitInputController>();
            yield return null; _input.SelectById(1); _input.enabled = false;
        }
        [UnityTearDown]
        public IEnumerator TearDown() { Object.Destroy(_host); yield return null; }

        [UnityTest]
        public IEnumerator RangeMarkersAppearBeforeTargetSelectionAndDistinguishMovePreviews()
        {
            var before = _game.State;
            var markers = _host.transform.Find("AttackTargetHighlights");
            Assert.That(_input.TargetId, Is.EqualTo(-1));
            Assert.That(_input.Mode, Is.EqualTo(UnitInputController.Intent.Move));
            Assert.That(markers.Find("AttackTarget_2").gameObject.activeSelf, Is.True);
            Assert.That(markers.GetComponentsInChildren<TextMesh>().Select(t => t.text), Is.EqualTo(new[] { "IN RANGE" }));
            Assert.That(_input.PreviewMove(new HexCoord(3, 1)), Is.True);
            Assert.That(markers.GetComponentsInChildren<TextMesh>().Select(t => t.text), Is.EqualTo(new[] { "IN RANGE", "AFTER MOVE" }));
            Assert.That(_game.State, Is.SameAs(before));
            _input.ClearPreview();
            Assert.That(markers.GetComponentsInChildren<TextMesh>().Select(t => t.text), Is.EqualTo(new[] { "IN RANGE" }));
            Assert.That(_input.PreviewAttack(2), Is.True); Assert.That(_input.ConfirmPreview(), Is.True);
            _game.Presenter.FastForward();
            Assert.That(markers.GetComponentsInChildren<TextMesh>(), Is.Empty, "Spent units must not advertise available attacks.");
            yield return null;
        }

        void BoardTap(int? unit, HexCoord? cell, double time)
        {
            var view = unit.HasValue ? _host.GetComponent<TokenStore>().UnitToken(unit.Value).GetComponent<UnitView>() : null;
            var tile = cell.HasValue ? _host.transform.Find($"Columns/Hex_{cell.Value.Q}_{cell.Value.R}").GetComponent<TileView>() : null;
            typeof(UnitInputController).GetMethod("HandleBoardTap", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)
                .Invoke(_input, new object[] { view, tile, new Vector2(200, 200), time });
        }

        [UnityTest]
        public IEnumerator SingleClickMovesAndUndoRestoresThePieceWithoutAnotherConfirmation()
        {
            var start = _game.State;
            BoardTap(null, new HexCoord(1, 1), 1);
            Assert.That(_game.State.Player(PlayerId.Player0).UnitsOnBoard.Single().Cell, Is.EqualTo(new HexCoord(1, 1)));
            Assert.That(_input.CanUndoMove, Is.True);
            var moved = _game.State;
            BoardTap(null, new HexCoord(1, 1), 1.15);
            Assert.That(_game.State, Is.SameAs(moved), "A habitual second click must not submit a second move.");
            Assert.That(_input.UndoLastMove(), Is.True);
            _game.Presenter.FastForward();
            Assert.That(_game.State.Player(PlayerId.Player0).UnitsOnBoard.Single().Cell, Is.EqualTo(new HexCoord(0, 1)));
            Assert.That(_game.State.MovementSpent, Is.Empty);
            Assert.That(_input.CanUndoMove, Is.False);
            Assert.That(_host.GetComponent<TokenStore>().UnitToken(1).GetComponent<UnitView>().Unit.Cell, Is.EqualTo(new HexCoord(0, 1)));
            yield return null;
        }

        [UnityTest]
        public IEnumerator AttackCanBeConfirmedAfterReadingThePreviewWithoutDoubleClickTiming()
        {
            var start = _game.State;
            BoardTap(2, null, 1);
            Assert.That(_game.State, Is.SameAs(start));
            BoardTap(2, null, 8); _game.Presenter.FastForward();
            Assert.That(_game.State.Player(PlayerId.Player1).UnitsOnBoard.First().CurrentHp, Is.EqualTo(5));
            Assert.That(_input.TargetId, Is.EqualTo(-1));
            Assert.That(_input.CanUndoMove, Is.False);
            yield return null;
        }

        [UnityTest]
        public IEnumerator DifferentTargetsAndCancelledAttackPreviewsDoNotFire()
        {
            BoardTap(null, new HexCoord(3, 1), 1);
            _game.Presenter.FastForward();
            var moved = _game.State;
            BoardTap(2, null, 2);
            BoardTap(3, null, 2.1);
            Assert.That(_game.State, Is.SameAs(moved), "Changing targets only changes the damage preview.");
            _input.ClearPreview();
            BoardTap(3, null, 2.2);
            Assert.That(_game.State, Is.SameAs(moved), "Cancelling disarms attack confirmation.");
            yield return null;
        }

        [UnityTest]
        public IEnumerator ManualPlacementUsesFreeHomeHexesAndBothReadyButtonsBeforeBattle()
        {
            var state = GameFactory.Build(new GameSetup(GameMode.Annihilation, 9, 7, 40, 7, turnActions: 1, manualPlacement: true));
            typeof(GameBootstrap).GetProperty("State").SetValue(_game, state);
            _board.Render(state.Board); _board.RenderEntities(state);
            _input.SelectById(state.Players[0].UnitsOnBoard.First().Id);
            var hud = _host.AddComponent<TacticalHud>(); yield return null; yield return null;
            var destination = _input.PlacementCells.OrderByDescending(c => state.Board.TileAt(c).Elevation).First();
            BoardTap(null, destination, 1);
            Assert.That(_input.Routes, Is.Empty, "Placement must not pretend to spend a movement route.");
            Assert.That(_input.PreviewAttack(state.Players[1].UnitsOnBoard.First().Id), Is.False);
            _game.Presenter.FastForward(); yield return null;
            Assert.That(_game.State.Players[0].UnitsOnBoard.First().Cell, Is.EqualTo(destination));
            var ready = _host.GetComponentsInChildren<Button>().Single(b => b.name == "End turn");
            Assert.That(ready.GetComponentInChildren<Text>().text, Is.EqualTo("Ready"));
            ready.onClick.Invoke(); _game.Presenter.FastForward(); yield return null;
            Assert.That(_game.State.PlacingStartingUnits, Is.True); Assert.That(_game.State.ActivePlayer, Is.EqualTo(PlayerId.Player1));
            ready.onClick.Invoke(); _game.Presenter.FastForward(); yield return null;
            Assert.That(_game.State.PlacingStartingUnits, Is.False); Assert.That(_game.State.ActivePlayer, Is.EqualTo(PlayerId.Player0));
            Assert.That(_input.SelectedUnit.Value.Owner, Is.EqualTo(PlayerId.Player0), "Battle begins with the first player's army selected.");
            Assert.That(_game.State.Round, Is.EqualTo(1)); Assert.That(_game.State.MovementSpent, Is.Empty);
            Assert.That(_game.State.Players.Select(p => p.Points), Is.EqualTo(new[] { 40, 40 }));
        }

        [UnityTest]
        public IEnumerator EscapeImmediatelyDismissesHelpThenDesignerWithoutOpeningMenu()
        {
            _host.AddComponent<DesignPanel>(); _host.AddComponent<BarracksPanel>();
            var hud = _host.AddComponent<TacticalHud>(); var menu = _host.AddComponent<EscapeMenu>();
            yield return null; yield return null;
            hud.SetWorkshop(true);
            TipBubble.Show("Unit destroyed. +4 points for your army.", new Vector2(200, 160), modal: false);
            Assert.That(TipBubble.IsOpen, Is.True);
            Assert.That(menu.HandleEscape(), Is.True);
            Assert.That(TipBubble.IsOpen, Is.False, "Escape must not wait for a grace period or deferred destruction.");
            Assert.That(hud.WorkshopOpen, Is.True, "One Escape closes only the topmost surface.");
            Assert.That(GameObject.Find("EscapeMenuCanvas"), Is.Null);
            yield return null;
            Assert.That(menu.HandleEscape(), Is.True); Assert.That(hud.WorkshopOpen, Is.False);
            Assert.That(GameObject.Find("EscapeMenuCanvas"), Is.Null);
            yield return null;
            menu.Toggle();
            var menuCanvas = GameObject.Find("EscapeMenuCanvas");
            Assert.That(menuCanvas, Is.Not.Null);
            GameRules.Show(menuCanvas.transform, UiKit.Font(), UiKit.OrderEscape + 10);
            Assert.That(menu.HandleEscape(), Is.True);
            Assert.That(GameObject.Find("RulesCanvas"), Is.Null);
            Assert.That(menuCanvas.activeInHierarchy, Is.True, "Escape closes the help above the menu first.");
            yield return null;
            Assert.That(menu.HandleEscape(), Is.True);
            Assert.That(GameObject.Find("EscapeMenuCanvas"), Is.Null);
        }

        [UnityTest]
        public IEnumerator EscapeKeyDoesNotCancelAPreviewUnderDismissedHelp()
        {
            var menu = _host.AddComponent<EscapeMenu>(); yield return null;
            var focus = DeviceInput.FocusProbe; var previousKeyboard = Keyboard.current;
            var background = InputSystem.settings.backgroundBehavior;
#if UNITY_EDITOR
            var editorInput = InputSystem.settings.editorInputBehaviorInPlayMode;
#endif
            var keyboard = InputSystem.AddDevice<Keyboard>();
            try
            {
                DeviceInput.FocusProbe = () => true;
                InputSystem.settings.backgroundBehavior = InputSettings.BackgroundBehavior.IgnoreFocus;
#if UNITY_EDITOR
                // Batch-mode tests have no focused Game View; deliver this synthetic device to the game.
                InputSystem.settings.editorInputBehaviorInPlayMode = InputSettings.EditorInputBehaviorInPlayMode.AllDeviceInputAlwaysGoesToGameView;
#endif
                InputSystem.EnableDevice(keyboard);
                var cell = new HexCoord(1, 1);
                Assert.That(_input.PreviewMove(cell), Is.True);
                TipBubble.Show("Help above the preview", new Vector2(200, 160), modal: false);
                InputSystem.QueueStateEvent(keyboard, new KeyboardState(Key.Escape));
                InputSystem.Update();
                Assert.That(Keyboard.current, Is.SameAs(keyboard));
                Assert.That(keyboard.escapeKey.wasPressedThisFrame, Is.True, "Fixture must deliver a fresh Escape press.");
                // Invoke each component in its declared execution order. SendMessage would invoke
                // every component on this shared fixture host in attachment order instead.
                const System.Reflection.BindingFlags flags = System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic;
                typeof(EscapeMenu).GetMethod("Update", flags).Invoke(menu, null);
                typeof(UnitInputController).GetMethod("Update", flags).Invoke(_input, null);
                Assert.That(TipBubble.IsOpen, Is.False);
                Assert.That(_input.Destination, Is.EqualTo(cell), "The consumed key must not reach the board.");
                Assert.That(GameObject.Find("EscapeMenuCanvas"), Is.Null);
            }
            finally
            {
                InputSystem.RemoveDevice(keyboard); previousKeyboard?.MakeCurrent();
                InputSystem.settings.backgroundBehavior = background;
#if UNITY_EDITOR
                InputSystem.settings.editorInputBehaviorInPlayMode = editorInput;
#endif
                DeviceInput.FocusProbe = focus; TipBubble.Dismiss();
            }
            yield return null;
        }

        [UnityTest]
        public IEnumerator AutomaticTipsAreSmallerAndDoNotReplaceExplicitHelp()
        {
            bool enabled = TipsService.Enabled;
            try
            {
                TipsService.Enabled = true; TipsService.NewGame();
                TipBubble.Show("Reference help", new Vector2(200, 200));
                TipsService.Show("new-bounty", "Automatic coaching");
                var root = GameObject.Find(TipBubble.RootName);
                Assert.That(root.GetComponentsInChildren<Text>().Any(t => t.text == "Reference help"), Is.True);
                TipBubble.Dismiss();
                TipsService.Show("new-bounty", "Automatic coaching");
                root = GameObject.Find(TipBubble.RootName);
                Assert.That(root.GetComponentInChildren<Text>().fontSize, Is.EqualTo(15));
                Assert.That(root.transform.Find("Backdrop"), Is.Null);
                Assert.That(root.transform.Find("Bubble").GetComponent<RectTransform>().rect.width, Is.EqualTo(320));
            }
            finally { TipBubble.Dismiss(); TipsService.Enabled = enabled; }
            yield return null;
        }

        [UnityTest]
        public IEnumerator DisabledBiomesHaveNoTerrainColorOrDetailsAndCanBeEnabledAgain()
        {
            var columns = _host.transform.Find("Columns");
            var plain = columns.Find("Hex_0_0/Fill").GetComponent<MeshRenderer>().sharedMaterial;
            foreach (Transform column in columns)
            {
                Assert.That(column.Find("Fill").GetComponent<MeshRenderer>().sharedMaterial, Is.SameAs(plain));
                var detail = column.Find("Terrain detail");
                if (detail != null) Assert.That(detail.gameObject.activeSelf, Is.False);
            }
            var s = _game.State;
            var enabled = new GameState(s.Board, GameConfig.Default(biomesEnabled: true), s.Players, s.ActivePlayer, s.Round, s.NextEntityId);
            _board.RenderEntities(enabled);
            Assert.That(columns.GetComponentsInChildren<MeshFilter>().Any(m => m.name == "Terrain detail"), Is.True);
            _board.RenderEntities(s);
            Assert.That(columns.GetComponentsInChildren<MeshFilter>().Any(m => m.name == "Terrain detail"), Is.False);
            yield return null;
        }

        [UnityTest]
        public IEnumerator OwnerHullRimAndPortraitRemainDistinctAcrossTurnsAndSelection()
        {
            var store = _host.GetComponent<TokenStore>();
            var a = store.UnitToken(1).transform.Find("Art_atlas-01");
            var b = store.UnitToken(2).transform.Find("Art_atlas-01");
            var aHull = a.Find("Graphite").GetComponent<MeshRenderer>().sharedMaterial;
            var bHull = b.Find("Graphite").GetComponent<MeshRenderer>().sharedMaterial;
            Assert.That(aHull.GetColor("_BaseColor"), Is.Not.EqualTo(bHull.GetColor("_BaseColor")));
            Assert.That(a.Find("TeamRim").GetComponent<MeshFilter>().sharedMesh,
                Is.Not.SameAs(b.Find("TeamRim").GetComponent<MeshFilter>().sharedMesh));
            var bRim = b.Find("TeamRim").GetComponent<MeshRenderer>().sharedMaterial;
            _game.TryApply(new EndTurn(PlayerId.Player0)); _game.Presenter.FastForward();
            Assert.That(b.Find("TeamRim").GetComponent<MeshRenderer>().sharedMaterial, Is.SameAs(bRim));
            Assert.That(b.Find("Graphite").GetComponent<MeshRenderer>().sharedMaterial, Is.SameAs(bHull));
            _host.AddComponent<TacticalHud>(); yield return null;
            _input.SelectById(2); yield return null;
            var hero = _host.GetComponentsInChildren<UnitPortrait>().Single(p => p.GetComponent<RawImage>().rectTransform.rect.width > 100);
            Assert.That(hero.Owner, Is.EqualTo(PlayerId.Player1));
            Texture p0 = null, p1 = null;
            for (int i = 0; i < 20 && (p0 == null || p1 == null); i++)
            {
                GraphitePieces.TryGetPortrait(1, true, out p0, PlayerId.Player0);
                GraphitePieces.TryGetPortrait(1, true, out p1, PlayerId.Player1);
                yield return null;
            }
            Assert.That(p0, Is.Not.Null); Assert.That(p1, Is.Not.Null); Assert.That(p1, Is.Not.SameAs(p0));
        }
    }
}
