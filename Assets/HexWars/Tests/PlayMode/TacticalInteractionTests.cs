using System.Collections;
using System.Linq;
using System.Reflection;
using System.Collections.Generic;
using HexWars.Engine;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
using UnityEngine.UI;
using UnityEngine.EventSystems;

namespace HexWars.Presentation.PlayModeTests
{
    public class TacticalInteractionTests
    {
        GameObject _host;GameBootstrap _game;UnitInputController _input;
        [UnitySetUp]
        public IEnumerator SetUp()
        {
            _host=new GameObject("Tactical interaction fixture",typeof(BoardRenderer),typeof(TokenStore),typeof(GameBootstrap));
            _game=_host.GetComponent<GameBootstrap>();_game.enabled=false;
            var presenter=_host.AddComponent<ActionPresenter>();typeof(GameBootstrap).GetProperty("Presenter").SetValue(_game,presenter);
            var tiles=Enumerable.Range(0,5).SelectMany(q=>Enumerable.Range(0,3).Select(r=>new Tile(new HexCoord(q,r),0,TerrainType.Plains))).ToArray();
            var a=new Unit(1,PlayerId.Player0,new UnitStats(4,3,0,3,1,5,1,6,1),new HexCoord(0,1),0,"Longshot","lance-01");
            var e=new Unit(2,PlayerId.Player1,new UnitStats(8,2,1,1,1,2,1,2,1),new HexCoord(4,1),0,"Bulwark","atlas-01");
            var state=new GameState(new Board(tiles),GameConfig.Default(),new[]{new PlayerState(PlayerId.Player0,30,unitsOnBoard:new[]{a}),new PlayerState(PlayerId.Player1,30,unitsOnBoard:new[]{e})},PlayerId.Player0,1,3);
            typeof(GameBootstrap).GetProperty("State").SetValue(_game,state);
            _host.GetComponent<BoardRenderer>().Render(state.Board);_host.GetComponent<BoardRenderer>().RenderEntities(state);
            _input=_host.AddComponent<UnitInputController>();
            yield return null;_input.SelectById(1);
        }
        [UnityTearDown]
        public IEnumerator TearDown(){Object.Destroy(_host);yield return null;}
        [UnityTest]
        public IEnumerator MoveAndAttackRequireConfirmationAndAttackEndsMovement()
        {
            var before=_game.State;
            Assert.That(_input.PreviewMove(new HexCoord(1,1)),Is.True);
            Assert.That(_game.State,Is.SameAs(before));
            Assert.That(_input.LockedRoute.HorizontalCost,Is.EqualTo(1));
            Assert.That(_input.ConfirmPreview(),Is.True);
            Assert.That(_game.State.Player(PlayerId.Player0).UnitsOnBoard.Single().Cell,Is.EqualTo(new HexCoord(1,1)));
            _game.Presenter.FastForward();yield return null;
            var moved=_game.State;
            Assert.That(_input.PreviewAttack(2),Is.True);Assert.That(_game.State,Is.SameAs(moved));
            Assert.That(_input.ConfirmPreview(),Is.True);_game.Presenter.FastForward();
            Assert.That(_game.State.Player(PlayerId.Player1).UnitsOnBoard.Single().CurrentHp,Is.EqualTo(6));
            Assert.That(_input.PreviewAttack(2),Is.False);Assert.That(_input.PreviewMove(new HexCoord(2,1)),Is.False);
            Assert.That(_input.ConfirmPreview(),Is.False);yield return null;
        }
        [UnityTest]
        public IEnumerator AuthoritativeStateChangeInvalidatesThePreview()
        {
            Assert.That(_input.PreviewAttack(2),Is.True);
            Assert.That(_game.TryApply(new EndTurn(PlayerId.Player0)),Is.True);
            Assert.That(_input.TargetId,Is.EqualTo(-1));Assert.That(_input.ConfirmPreview(),Is.False);
            Assert.That(_game.State.Player(PlayerId.Player1).UnitsOnBoard.Single().CurrentHp,Is.EqualTo(8));yield return null;
        }
        [UnityTest]
        public IEnumerator RejectionUnlocksOnlineSubmissionWithoutApplyingDamage()
        {
            Assert.That(_input.PreviewAttack(2),Is.True);
            _game.Networked=true; // no connection: submission must fail rather than claim a hit.
            var before=_game.State;Assert.That(_input.ConfirmPreview(),Is.False);
            Assert.That(_input.AwaitingServer,Is.False);Assert.That(_game.State,Is.SameAs(before));
            _game.OnNetReject("TemporaryFailure");Assert.That(_input.AwaitingServer,Is.False);yield return null;
        }

        void EnableDeferredNetworkSubmission()
        {
            // Exercise the real submission path while leaving delivery deferred; no socket is opened.
            var net=_host.AddComponent<NetClient>();net.enabled=false;
            typeof(GameBootstrap).GetField("_net",BindingFlags.Instance|BindingFlags.NonPublic).SetValue(_game,net);
            _game.Networked=true;_game.OnNetSeat(PlayerId.Player0);
        }

        [UnityTest]
        public IEnumerator ReconnectionAndSeatEventsKeepThePendingCommandLockedUntilResync()
        {
            EnableDeferredNetworkSubmission();
            var before=_game.State;
            Assert.That(_input.PreviewAttack(2),Is.True);Assert.That(_input.ConfirmPreview(),Is.True);
            Assert.That(_input.AwaitingServer,Is.True);
            _game.OnNetReconnecting(1);
            Assert.That(_input.AwaitingServer,Is.True);
            _game.OnNetReconnected();_game.OnNetSeat(PlayerId.Player0);
            Assert.That(_game.State,Is.SameAs(before));
            Assert.That(_input.AwaitingServer,Is.True);Assert.That(_input.CanCommand,Is.False);
            Assert.That(_input.PreviewAttack(2),Is.False);Assert.That(_input.ConfirmPreview(),Is.False);
            _game.OnNetStart(ReplayFile.Write(before,System.Array.Empty<Command>()));
            Assert.That(_game.State,Is.Not.SameAs(before));
            Assert.That(_input.AwaitingServer,Is.False);Assert.That(_input.CanCommand,Is.True);
            yield return null;
        }

        [UnityTest]
        public IEnumerator ServerRejectionAndAuthoritativeApplyReleaseThePendingCommand()
        {
            EnableDeferredNetworkSubmission();
            var before=_game.State;
            Assert.That(_input.PreviewAttack(2),Is.True);Assert.That(_input.ConfirmPreview(),Is.True);
            _game.OnNetReject("TemporaryFailure");
            Assert.That(_input.AwaitingServer,Is.False);Assert.That(_game.State,Is.SameAs(before));
            Assert.That(_input.PreviewAttack(2),Is.True);Assert.That(_input.ConfirmPreview(),Is.True);
            _game.OnNetApply(new AttackUnit(PlayerId.Player0,1,2));
            Assert.That(_input.AwaitingServer,Is.False);Assert.That(_game.State,Is.Not.SameAs(before));
            Assert.That(_game.State.Player(PlayerId.Player1).UnitsOnBoard.Single().CurrentHp,Is.EqualTo(6));
            yield return null;
        }

        void SetOpponentTurn(PlayerId human)
        {
            var state=_game.State;
            typeof(GameBootstrap).GetProperty("State").SetValue(_game,new GameState(state.Board,
                GameConfig.Default(fogOfWar:false),new[]{state.Players[0].WithPoints(31),state.Players[1].WithPoints(79)},
                human==PlayerId.Player0?PlayerId.Player1:PlayerId.Player0,1,state.NextEntityId));
        }

        void AssertHumanSquad(PlayerId human)
        {
            string own="Select unit "+(human==PlayerId.Player0?1:2),other="Select unit "+(human==PlayerId.Player0?2:1);
            var buttons=_host.GetComponentsInChildren<Button>();
            Assert.That(buttons.Any(b=>b.name==own),Is.True);
            Assert.That(buttons.Any(b=>b.name==other),Is.False);
            string points=$"/   {_game.State.Player(human).Points} points";
            Assert.That(_host.GetComponentsInChildren<Text>().Any(t=>t.text.Contains(points)),Is.True);
        }

        [UnityTest]
        public IEnumerator OnlineHudKeepsTheAssignedSquadAndPointsWhenFogIsOff()
        {
            SetOpponentTurn(PlayerId.Player0);_game.Networked=true;_game.OnNetSeat(PlayerId.Player0);
            _host.AddComponent<TacticalHud>();yield return null;yield return null;
            Assert.That(_game.FogViewer(),Is.Null);AssertHumanSquad(PlayerId.Player0);
            SetOpponentTurn(PlayerId.Player1);_game.OnNetSeat(PlayerId.Player1);yield return null;
            AssertHumanSquad(PlayerId.Player1);
        }

        [UnityTest]
        public IEnumerator AiHudKeepsTheHumanSquadAndPointsWhenFogIsOff()
        {
            var ai=_host.AddComponent<AiOpponent>();ai.enabled=false;ai.AiSeat=PlayerId.Player1;
            SetOpponentTurn(PlayerId.Player0);_host.AddComponent<TacticalHud>();yield return null;yield return null;
            Assert.That(_game.FogViewer(),Is.Null);AssertHumanSquad(PlayerId.Player0);
            ai.AiSeat=PlayerId.Player0;SetOpponentTurn(PlayerId.Player1);
            _game.OnNetReconnected();yield return null;AssertHumanSquad(PlayerId.Player1);
        }

        [UnityTest]
        public IEnumerator TerritoryActionHasAnUncoveredFooterSlotAndSurvivesHudRebuilds()
        {
            var state=_game.State;
            typeof(GameBootstrap).GetProperty("State").SetValue(_game,new GameState(state.Board,
                GameConfig.Default(territoryMode:true,claimEndsTurn:false,buildAnywhere:true),state.Players,
                state.ActivePlayer,state.Round,state.NextEntityId));
            var hud=_host.AddComponent<TacticalHud>();yield return null;yield return null;
            _input.SelectById(1);RefreshTerritoryAction();
            // RefreshTerritoryAction can activate this graphic for the first time in a headless
            // input fixture. Raycast only after its canvas has rendered and assigned draw depth.
            yield return null;
            var button=_host.GetComponentsInChildren<Button>().Single(b=>b.name=="ActionButton");
            AssertTerritoryFooter(button);
            _host.GetComponentsInChildren<Button>().Single(b=>b.name=="Unit details").onClick.Invoke();
            yield return null;RefreshTerritoryAction();yield return null;
            Assert.That(_host.GetComponentsInChildren<Button>().Single(b=>b.name=="ActionButton"),Is.SameAs(button));
            AssertTerritoryFooter(button);
            button.onClick.Invoke();
            Assert.That(_game.State.Board.Controller(new HexCoord(0,1)),Is.EqualTo(PlayerId.Player0),
                "The footer must retain the actual claim handler.");
            RefreshTerritoryAction();button.onClick.Invoke();RefreshTerritoryAction();
            Assert.That(button.GetComponentInChildren<Text>().text,Does.Contain("Building"),
                "The same accessible button must still enter generator-build mode.");
            hud.SetWorkshop(true);yield return null;Assert.That(button.gameObject.activeInHierarchy,Is.False);
            hud.SetWorkshop(false);yield return null;Assert.That(button.gameObject.activeInHierarchy,Is.True);
        }

        void RefreshTerritoryAction()=>typeof(UnitInputController).GetMethod("UpdateActionButton",
            BindingFlags.Instance|BindingFlags.NonPublic).Invoke(_input,null);

        void AssertTerritoryFooter(Button button)
        {
            Canvas.ForceUpdateCanvases();
            var graphic=button.GetComponent<Image>();
            Assert.That(graphic.enabled,Is.True,"The territory graphic must survive destruction of the previous HUD canvas.");
            Assert.That(graphic.raycastTarget,Is.True);
            Assert.That(graphic.canvasRenderer.cull,Is.False,"The territory control must be visible, not clipped.");
            Assert.That(graphic.depth,Is.GreaterThanOrEqualTo(0),"The territory control must have entered the canvas draw order.");
            var panel=_host.GetComponentsInChildren<RectTransform>().Single(r=>r.name=="Selected unit panel");
            Assert.That(button.transform.IsChildOf(panel),Is.True,
                "Territory controls must share the panel's canvas instead of sitting beneath it.");
            var bounds=RectTransformUtility.CalculateRelativeRectTransformBounds(panel,button.transform);
            Assert.That(bounds.min.x,Is.GreaterThanOrEqualTo(panel.rect.xMin));
            Assert.That(bounds.max.x,Is.LessThanOrEqualTo(panel.rect.xMax));
            Assert.That(bounds.min.y,Is.GreaterThanOrEqualTo(panel.rect.yMin));
            Assert.That(bounds.max.y,Is.LessThanOrEqualTo(panel.rect.yMax));
            var end=_host.GetComponentsInChildren<Button>().Single(b=>b.name=="End turn");
            var confirm=_host.GetComponentsInChildren<Button>().Single(b=>b.name=="Confirm action");
            Assert.That(bounds.min.y,Is.GreaterThan(RectTransformUtility.CalculateRelativeRectTransformBounds(panel,end.transform).max.y));
            Assert.That(bounds.max.y,Is.LessThan(RectTransformUtility.CalculateRelativeRectTransformBounds(panel,confirm.transform).min.y));
            var rect=(RectTransform)button.transform;
            var pointer=new PointerEventData(EventSystem.current)
                {position=RectTransformUtility.WorldToScreenPoint(null,rect.TransformPoint(rect.rect.center))};
            var hits=new List<RaycastResult>();
            var raycaster=button.GetComponentInParent<GraphicRaycaster>();
            raycaster.Raycast(pointer,hits);
            Assert.That(hits.Count,Is.GreaterThan(0));
            Assert.That(hits[0].gameObject,Is.SameAs(button.gameObject),
                "Top-to-bottom hits: "+string.Join(",",hits.Select(h=>$"{h.gameObject.name}:{h.depth}")));
        }
        [UnityTest]
        public IEnumerator RebuiltMatchSelectsCurrentTokenBeforeOldObjectsAreDestroyed()
        {
            var board=_host.GetComponent<BoardRenderer>();var store=_host.GetComponent<TokenStore>();
            var old=store.UnitToken(1);old.transform.position=new Vector3(100,0,100);
            board.Render(_game.State.Board);board.RenderEntities(_game.State);
            var current=store.UnitToken(1);Assert.That(current,Is.Not.SameAs(old));
            _input.SelectById(1);
            Assert.That(Vector3.Distance(_host.transform.Find("SelectionBrackets").position,
                current.transform.position+Vector3.up*.04f),Is.LessThan(.001f));
            yield return null;
            Assert.That(_input.SelectedId,Is.EqualTo(1));
        }
        [UnityTest]
        public IEnumerator ReducedMotionCommitsTheSameAttackWithoutProjectileOrCameraEffects()
        {
            bool previous=MotionSettings.Reduced;
            try
            {
                MotionSettings.Reduced=true;
                Assert.That(_input.PreviewAttack(2),Is.True);
                Assert.That(_input.ConfirmPreview(),Is.True);
                yield return null;
                Assert.That(_game.Presenter.IsBusy,Is.False);
                Assert.That(_game.State.Player(PlayerId.Player1).UnitsOnBoard.Single().CurrentHp,Is.EqualTo(6));
                Assert.That(_host.GetComponent<TokenStore>().UnitToken(2).GetComponent<UnitView>().Unit.CurrentHp,Is.EqualTo(6));
                Assert.That(GameObject.Find("Projectile"),Is.Null);
            }
            finally { MotionSettings.Reduced=previous; }
        }
        [UnityTest]
        public IEnumerator InspectorShowsActualForecastAndArmyToolsAreSecondary()
        {
            _host.AddComponent<DesignPanel>();_host.AddComponent<BarracksPanel>();
            var hud=_host.AddComponent<TacticalHud>();yield return null;yield return null;
            Assert.That(_host.GetComponent<DesignPanel>().Expanded,Is.False);
            Assert.That(_input.PreviewAttack(2),Is.True);yield return null;
            var confirm=_host.GetComponentsInChildren<Button>().Single(b=>b.name=="Confirm action");
            Assert.That(confirm.interactable,Is.True);Assert.That(confirm.GetComponentInChildren<Text>().text,Does.Contain("2 damage"));
            Assert.That(_host.GetComponentsInChildren<Text>().Any(t=>t.text.Contains("Weapon 3")),Is.False);
            _host.GetComponentsInChildren<Button>().Single(b=>b.name=="Unit details").onClick.Invoke();yield return null;
            Assert.That(_host.GetComponentsInChildren<Text>().Any(t=>t.text.Contains("Weapon 3")),Is.True);
            Assert.That(_input.TargetId,Is.EqualTo(2),"Expanding details must preserve the preview.");
            hud.SetWorkshop(true);yield return null;
            Assert.That(_host.GetComponent<DesignPanel>().Expanded,Is.True);Assert.That(_host.GetComponent<BarracksPanel>().Expanded,Is.True);
            hud.SetWorkshop(false);yield return null;Assert.That(_host.GetComponent<DesignPanel>().Expanded,Is.False);
        }
    }
}
