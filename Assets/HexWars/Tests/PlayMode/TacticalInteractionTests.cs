using System.Collections;
using System.Linq;
using HexWars.Engine;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
using UnityEngine.UI;

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
