using System;
using System.Linq;
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class UnitArtTests
    {
        static readonly UnitStats Stats = new UnitStats(3,4,0,2,1,5,1,2,1);
        static GameState Start() => new GameState(
            new Board(new[] {new Tile(new HexCoord(0,0),0,TerrainType.Plains),new Tile(new HexCoord(1,0),0,TerrainType.Plains)},
                zone0:new[]{new HexCoord(0,0)},zone1:new[]{new HexCoord(1,0)}),GameConfig.Default(),
            new[]{new PlayerState(PlayerId.Player0,200),new PlayerState(PlayerId.Player1,200)},PlayerId.Player0,1,1);
        [Test]
        public void ChoiceSurvivesCommandCatalogDeployMoveDamageAndReplay()
        {
            var created=GameEngine.Apply(Start(),CommandWire.Read(CommandWire.Write(new CreateUnit(PlayerId.Player0,Stats,"Watcher","halo-01"))));
            Assert.That(created.Success,Is.True);
            var catalog=BarracksWire.Read(BarracksWire.Write(created.NewState.Player(PlayerId.Player0).Barracks));
            Assert.That(catalog[0].ArtId,Is.EqualTo("halo-01"));
            var deployed=GameEngine.Apply(created.NewState,new DeployUnit(PlayerId.Player0,0,new HexCoord(0,0)));
            Assert.That(deployed.Success,Is.True);
            var unit=deployed.NewState.Player(PlayerId.Player0).UnitsOnBoard.Single();
            Assert.That(unit.WithCell(new HexCoord(1,0),2).WithDamage(1).ArtId,Is.EqualTo("halo-01"));
            var recovered=ReplayFile.Read(ReplayFile.Write(deployed.NewState,Array.Empty<Command>())).Start;
            Assert.That(recovered.Player(PlayerId.Player0).UnitsOnBoard.Single().ArtId,Is.EqualTo("halo-01"));
            Assert.That(recovered.Player(PlayerId.Player0).Barracks[0].ArtId,Is.EqualTo("halo-01"));
            var removed=GameEngine.Apply(deployed.NewState,new DeleteTemplate(PlayerId.Player0,0));
            Assert.That(removed.NewState.Player(PlayerId.Player0).UnitsOnBoard.Single().ArtId,Is.EqualTo("halo-01"));
        }
        [Test]
        public void LegacyBytesAndUnknownAssetFallbackRemainUsable()
        {
            Assert.That(CommandWire.Write(new CreateUnit(PlayerId.Player0,Stats,"A")),Does.StartWith("C 0 "));
            Assert.That(BarracksWire.Write(new[]{new UnitTemplate("A",Stats)}),Does.StartWith("V1\n"));
            Assert.That(ReplayFile.Write(Start(),Array.Empty<Command>()),Does.StartWith("HEXWARS-REPLAY 1\n"));
            Assert.That(UnitArt.Resolve("missing-99",Stats),Is.EqualTo("lance-01"));
            Assert.That(new UnitTemplate("A",Stats,"../../file").ArtId,Is.Empty);
            Assert.That(((CreateUnit)CommandWire.Read("C 0 3 4 0 2 1 5 1 2 1 A")).ArtId,Is.Empty);
        }
        [Test]
        public void ManualSelectionDoesNotAlterStatsOrPointCost()
        {
            var a=GameEngine.Apply(Start(),new CreateUnit(PlayerId.Player0,Stats,"A"));
            var b=GameEngine.Apply(Start(),new CreateUnit(PlayerId.Player0,Stats,"A","halo-01"));
            Assert.That(a.Success,Is.EqualTo(b.Success));
            Assert.That(a.NewState.Player(PlayerId.Player0).Points,Is.EqualTo(b.NewState.Player(PlayerId.Player0).Points));
            Assert.That(a.NewState.Player(PlayerId.Player0).Barracks[0].Stats,Is.EqualTo(b.NewState.Player(PlayerId.Player0).Barracks[0].Stats));
        }
        [Test]
        public void ReplacementAndEmptyNameHaveUnambiguousVersionedRoundTrips()
        {
            var command=new ReplaceTemplate(PlayerId.Player0,2,Stats,"","crux-01");
            Assert.That(CommandWire.Read(CommandWire.Write(command)),Is.EqualTo(command));
            Assert.That(CommandWire.TryRead("C2 0 3 4 0 2 1 5 1 2 1",out _),Is.False);
            var templates=BarracksCatalog.Normalize(new[]{new UnitTemplate("A",Stats,"halo-01"),new UnitTemplate("A",Stats,"crux-01")});
            Assert.That(templates.Count,Is.EqualTo(2));
        }
    }
}
