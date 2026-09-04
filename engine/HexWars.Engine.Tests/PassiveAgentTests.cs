using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public sealed class PassiveAgentTests
    {
        [Test]
        public void Decide_AlwaysEndsTheActivePlayersTurn()
        {
            var agent = new PassiveAgent();
            GameState player0Turn = TestStates.Fresh();
            GameState player1Turn = GameEngine.Apply(
                player0Turn, new EndTurn(PlayerId.Player0)).NewState;

            Assert.Multiple(() =>
            {
                Assert.That(agent.Decide(player0Turn),
                    Is.EqualTo(new EndTurn(PlayerId.Player0)));
                Assert.That(agent.Decide(player1Turn),
                    Is.EqualTo(new EndTurn(PlayerId.Player1)));
            });
        }
    }
}
