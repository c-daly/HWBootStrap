using System.Collections;
using System.Linq;
using System.Reflection;
using HexWars.Engine;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

namespace HexWars.Presentation.PlayModeTests
{
    public class GraphiteDesignerSmokeTests
    {
        [UnityTest]
        public IEnumerator ManualChoiceSurvivesStatEditCreationAndBoardRendering()
        {
            SessionBarracksCache.ResetForTests();
            var host=new GameObject("Graphite designer smoke",typeof(BoardRenderer),typeof(TokenStore),typeof(GameBootstrap));
            var game=host.GetComponent<GameBootstrap>();game.enabled=false;
            var presenter=host.AddComponent<ActionPresenter>();
            typeof(GameBootstrap).GetProperty("Presenter").SetValue(game,presenter);
            game.StartLocalGame(new GameSetup(GameMode.Annihilation,5,5,200,7),false);
            var designer=host.AddComponent<DesignPanel>();
            yield return null;
            designer.SelectArt("halo-01");
            typeof(DesignPanel).GetMethod("Adjust",BindingFlags.NonPublic|BindingFlags.Instance).Invoke(designer,new object[]{0,4});
            Assert.That(designer.AppearanceSelection,Is.EqualTo("halo-01"));
            int slot=game.State.Player(PlayerId.Player0).Barracks.Count;
            typeof(DesignPanel).GetMethod("OnCreate",BindingFlags.NonPublic|BindingFlags.Instance).Invoke(designer,null);
            var template=game.State.Player(PlayerId.Player0).Barracks[slot];
            Assert.That(template.ArtId,Is.EqualTo("halo-01"));
            var state=game.State;
            var cell=state.Board.DeploymentZone(PlayerId.Player0).First(c=>!state.Players.Any(p=>p.UnitsOnBoard.Any(u=>u.Cell==c)));
            Assert.That(game.TryApply(new DeployUnit(PlayerId.Player0,slot,cell)),Is.True);
            yield return null;
            yield return new WaitForSeconds(.5f);
            var unit=game.State.Player(PlayerId.Player0).UnitsOnBoard.Last();
            var token=host.GetComponent<TokenStore>().UnitToken(unit.Id);
            Assert.That(token.transform.Find("Art_halo-01"),Is.Not.Null);
            Assert.That(token.GetComponent<BoxCollider>(),Is.Not.Null);
            Assert.That(SessionBarracksCache.ForLocalPlayer(0).Snapshot().Last().ArtId,Is.EqualTo("halo-01"));
            Object.Destroy(host);
            SessionBarracksCache.ResetForTests();
            yield return null;
        }
    }
}
