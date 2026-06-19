using System.Collections.Generic;
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class GameEngineApplyTests
    {
        private static GameState Fresh(int p0Points = 12, int p1Points = 12) =>
            Make(GameConfig.Default(), p0Points, p1Points);

        private static GameState Make(GameConfig config, int p0Points, int p1Points)
        {
            var board = new Board(new[]
            {
                new Tile(new HexCoord(0, 0), 0, TerrainType.Plains),
                new Tile(new HexCoord(1, 0), 0, TerrainType.Plains),
            }, zone0: new[] { new HexCoord(0, 0) }, zone1: new[] { new HexCoord(1, 0) });
            var players = new[]
            {
                new PlayerState(PlayerId.Player0, p0Points),
                new PlayerState(PlayerId.Player1, p1Points),
            };
            return new GameState(board, config, players, PlayerId.Player0, 1, 1);
        }

        private static GameConfig WithDesignFee(int fee) =>
            new GameConfig(new Dictionary<TerrainType, TerrainDef>
            {
                { TerrainType.Plains, new TerrainDef(1, 0, 0, true) },
            }, designFee: fee);

        private static UnitStats Cost(int c) => new UnitStats(c, 0, 0, 0, 0, 0, 0, 0, 0);

        [Test]
        public void CreateUnit_IsFreeByDefault_AndAddsTemplateToBarracks_NonMutating()
        {
            var state = Fresh(p0Points: 5);
            var result = GameEngine.Apply(state, new CreateUnit(PlayerId.Player0, Cost(3)));

            Assert.That(result.Success, Is.True);
            Assert.That(result.NewState.Player(PlayerId.Player0).Points, Is.EqualTo(5)); // design is free by default
            Assert.That(result.NewState.Player(PlayerId.Player0).Barracks.Count, Is.EqualTo(1));
            Assert.That(state.Player(PlayerId.Player0).Barracks.Count, Is.EqualTo(0)); // original untouched
        }

        [Test]
        public void CreateUnit_ChargesDesignFee_WhenConfigured()
        {
            var result = GameEngine.Apply(Make(WithDesignFee(3), 5, 12),
                                          new CreateUnit(PlayerId.Player0, Cost(3)));
            Assert.That(result.Success, Is.True);
            Assert.That(result.NewState.Player(PlayerId.Player0).Points, Is.EqualTo(2));
        }

        [Test]
        public void CreateUnit_Rejects_WhenCannotAffordDesignFee()
        {
            var result = GameEngine.Apply(Make(WithDesignFee(3), 2, 12),
                                          new CreateUnit(PlayerId.Player0, Cost(3)));
            Assert.That(result.Success, Is.False);
            Assert.That(result.Reason, Is.EqualTo(RejectionReason.InsufficientPoints));
        }

        [Test]
        public void Apply_Rejects_WhenNotYourTurn()
        {
            var result = GameEngine.Apply(Fresh(), new CreateUnit(PlayerId.Player1, Cost(1)));
            Assert.That(result.Reason, Is.EqualTo(RejectionReason.NotYourTurn));
        }

        [Test]
        public void CreateUnit_Rejects_InvalidStatsBelowOneHealth()
        {
            var bad = new UnitStats(0, 5, 0, 0, 0, 0, 0, 0, 0); // 0 health
            var result = GameEngine.Apply(Fresh(), new CreateUnit(PlayerId.Player0, bad));
            Assert.That(result.Reason, Is.EqualTo(RejectionReason.InvalidStats));
        }
    }
}
