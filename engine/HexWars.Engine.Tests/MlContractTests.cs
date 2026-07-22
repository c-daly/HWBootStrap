using System.Collections.Generic;
using HexWars.Engine;
using HexWars.Engine.Rl;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class MlContractTests
    {
        [Test]
        public void Create_IsDeterministicAndMatchesTacticalLayout()
        {
            var config = new EnvConfig();
            var layout = new TacticalLayout(config);

            var first = MlContract.Create(config);
            var second = MlContract.Create(config);

            Assert.That(first.ContractHash, Is.EqualTo(second.ContractHash));
            Assert.That(first.ObservationSize, Is.EqualTo(layout.ObservationLength));
            Assert.That(first.ActionSize, Is.EqualTo(layout.ActionCount));
            Assert.That(first.Board["width"], Is.EqualTo(layout.BoardW));
            Assert.That(first.Board["height"], Is.EqualTo(layout.BoardH));
        }

        [Test]
        public void Create_ChangesHashForBoardRosterAndRewardSemantics()
        {
            var baseline = MlContract.Create(new EnvConfig());
            var changedBoard = MlContract.Create(new EnvConfig
            {
                BoardGen = new BoardGenConfig(width: 14),
            });
            var changedRoster = MlContract.Create(new EnvConfig
            {
                Roster = new List<UnitStats>
                {
                    new UnitStats(6, 3, 2, 3, 2, 1, 1, 2, 1),
                    new UnitStats(3, 5, 0, 3, 2, 2, 1, 3, 1),
                    new UnitStats(2, 2, 0, 4, 3, 1, 0, 5, 2),
                },
            });
            var changedReward = MlContract.Create(new EnvConfig { ClosingWeight = 0.03f });
            var changedHorizon = MlContract.Create(new EnvConfig { MaxSteps = 601 });

            Assert.That(changedBoard.ContractHash, Is.Not.EqualTo(baseline.ContractHash));
            Assert.That(changedRoster.ContractHash, Is.Not.EqualTo(baseline.ContractHash));
            Assert.That(changedReward.ContractHash, Is.Not.EqualTo(baseline.ContractHash));
            Assert.That(changedHorizon.ContractHash, Is.Not.EqualTo(baseline.ContractHash));
        }

        [Test]
        public void Create_UsesDistinctContractsAndEffectiveHorizonsForTacticalAndDuelModes()
        {
            var config = new EnvConfig { MaxSteps = 123 };

            var tactical = MlContract.Create(config, MlEnvironmentKind.Tactical);
            var duel = MlContract.Create(config, MlEnvironmentKind.Duel);

            Assert.That(tactical.EnvironmentKind, Is.EqualTo("tactical"));
            Assert.That(duel.EnvironmentKind, Is.EqualTo("duel"));
            Assert.That(tactical.Board["max_steps"], Is.EqualTo(123));
            Assert.That(duel.Board["max_steps"], Is.EqualTo(246));
            Assert.That(duel.ContractHash, Is.Not.EqualTo(tactical.ContractHash));
        }
    }
}
