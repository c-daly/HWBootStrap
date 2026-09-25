using System.Linq;
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Presentation.Tests
{
    public class TacticalForecastTests
    {
        static GameState Position(bool biomes=true,bool fog=false,int damage=5,int vision=8,int[] attacked=null)
        {
            var board=new Board(Enumerable.Range(0,5).Select(q=>new Tile(new HexCoord(q,0),q==0?1:0,q==3?TerrainType.Forest:TerrainType.Plains)).ToArray());
            var attacker=new Unit(1,PlayerId.Player0,new UnitStats(4,damage,0,3,2,6,2,vision,2),new HexCoord(0,0),1,"Longshot","lance-01");
            var target=new Unit(2,PlayerId.Player1,new UnitStats(8,2,2,1,1,2,1,2,1),new HexCoord(3,0),0,"Bulwark","atlas-01");
            return new GameState(board,GameConfig.Default(biomesEnabled:biomes,fogOfWar:fog),new[]{
                new PlayerState(PlayerId.Player0,20,unitsOnBoard:new[]{attacker}),new PlayerState(PlayerId.Player1,20,unitsOnBoard:new[]{target})},PlayerId.Player0,1,3,attackedUnitIds:attacked);
        }
        [TestCase(true,3,1)]
        [TestCase(false,4,0)]
        public void ForecastMatchesAppliedDamageAndRespectsTerrainSetting(bool biomes,int damage,int cover)
        {
            var state=Position(biomes);
            Assert.That(TacticalForecast.TryCreate(state,PlayerId.Player0,1,2,out var forecast),Is.True);
            Assert.That(forecast.Damage,Is.EqualTo(damage));Assert.That(forecast.Cover,Is.EqualTo(cover));
            var applied=GameEngine.Apply(state,new AttackUnit(PlayerId.Player0,1,2));
            Assert.That(applied.Success,Is.True);
            Assert.That(applied.NewState.Player(PlayerId.Player1).UnitsOnBoard.Single().CurrentHp,Is.EqualTo(forecast.RemainingHealth));
            Assert.That(state.Player(PlayerId.Player1).UnitsOnBoard.Single().CurrentHp,Is.EqualTo(8),"A preview must not mutate state.");
        }
        [Test]
        public void NonCombatantDoesNotReceiveHighGroundDamage()
        {
            Assert.That(TacticalForecast.TryCreate(Position(damage:0),PlayerId.Player0,1,2,out var f),Is.True);
            Assert.That(f.Damage,Is.Zero);Assert.That(f.HeightBonus,Is.Zero);
        }
        [Test]
        public void HiddenTargetsCannotBeInspectedOrForecast()
        {
            var state=Position(fog:true,vision:0);
            Assert.That(TacticalForecast.FindVisible(state,2,PlayerId.Player0),Is.Null);
            Assert.That(TacticalForecast.TryCreate(state,PlayerId.Player0,1,2,out _),Is.False);
        }
        [Test]
        public void UsedAttackCannotBeForecastAgain()
        {Assert.That(TacticalForecast.TryCreate(Position(attacked:new[]{1}),PlayerId.Player0,1,2,out _),Is.False);}
        [Test]
        public void OpponentCannotCommandTheSelectedUnit()
        {Assert.That(TacticalForecast.TryCreate(Position(),PlayerId.Player1,1,2,out _),Is.False);}
        [Test]
        public void NullOrUnknownSelectionHasNoForecast()
        {Assert.That(TacticalForecast.TryCreate(null,PlayerId.Player0,1,2,out _),Is.False);Assert.That(TacticalForecast.TryCreate(Position(),PlayerId.Player0,99,2,out _),Is.False);}
    }
}
