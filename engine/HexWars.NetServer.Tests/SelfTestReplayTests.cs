using System.Linq;
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.NetServer.Tests;

public class SelfTestReplayTests
{
    [Test]
    public void RecoveryFixtureMovesAndDamagesBeforeReconnectingWithoutEndingTheGame()
    {
        var start = GameFactory.Build(SelfTest.ReplaySetup);
        var move = GameEngine.Apply(start, new MoveUnit(PlayerId.Player0, 2, new HexCoord(3, 0)));
        Assert.That(move.Success, Is.True, move.Reason.ToString());
        var hit = GameEngine.Apply(move.NewState, new AttackUnit(PlayerId.Player0, 2, 5));
        Assert.That(hit.Success, Is.True, hit.Reason.ToString());
        var before = start.Player(PlayerId.Player1).UnitsOnBoard.Single(u => u.Id == 5).CurrentHp;
        var after = hit.NewState.Player(PlayerId.Player1).UnitsOnBoard.Where(u => u.Id == 5).Sum(u => u.CurrentHp);
        Assert.That(after, Is.LessThan(before));
        Assert.That(hit.NewState.IsGameOver, Is.False);
        Assert.That(GameEngine.Apply(hit.NewState, new EndTurn(PlayerId.Player0)).Success, Is.True);
    }
}
