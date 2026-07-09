using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    /// <summary>
    /// The networked wire-format for a single <see cref="Command"/> must round-trip every command type
    /// exactly (issuer + payload), so the same code on server and client never drifts. This is the
    /// foundation the multiplayer relay sends move-by-move.
    /// </summary>
    public class CommandWireTests
    {
        private const PlayerId P0 = PlayerId.Player0;
        private const PlayerId P1 = PlayerId.Player1;

        private static void RoundTrips(Command c)
        {
            string wire = CommandWire.Write(c);
            Command back = CommandWire.Read(wire);
            Assert.That(back, Is.EqualTo(c), $"round-trip changed the command (wire='{wire}')");
        }

        [Test] public void EndTurn_RoundTrips() => RoundTrips(new EndTurn(P1));
        [Test] public void MoveUnit_RoundTrips() => RoundTrips(new MoveUnit(P0, 7, new HexCoord(3, -2)));
        [Test] public void AttackUnit_RoundTrips() => RoundTrips(new AttackUnit(P0, 7, 12));
        [Test] public void CreateUnit_RoundTrips() => RoundTrips(new CreateUnit(P1, new UnitStats(3, 4, 1, 2, 1, 2, 1, 3, 1)));

        [Test]
        public void CreateUnit_RoundTrips_WithName() =>
            RoundTrips(new CreateUnit(P1, new UnitStats(3, 4, 1, 2, 1, 2, 1, 3, 1), "Doom Turtle"));

        [Test]
        public void CreateUnit_Read_OldFormatMissingNameToken_DefaultsToEmptyName()
        {
            var cmd = (CreateUnit)CommandWire.Read("C 1 3 4 1 2 1 2 1 3 1");
            Assert.That(cmd.Name, Is.EqualTo(""));
        }

        [Test]
        public void CreateUnit_Write_EncodesSpacesAsUnderscores()
        {
            var wire = CommandWire.Write(new CreateUnit(P0, new UnitStats(1, 0, 0, 0, 0, 0, 0, 0, 0), "Doom Turtle"));
            Assert.That(wire, Does.EndWith(" Doom_Turtle"));
        }

        [Test]
        public void CreateUnit_RoundTrips_SpacedName_Exactly()
        {
            // spaces survive the underscore transport encoding both ways (sanitized names can never
            // contain a literal underscore, so decode's underscore -> space is lossless)
            var cmd = (CreateUnit)CommandWire.Read(CommandWire.Write(
                new CreateUnit(P0, new UnitStats(2, 2, 0, 2, 2, 6, 1, 4, 1), "Long Shot Mk-2")));
            Assert.That(cmd.Name, Is.EqualTo("Long Shot Mk-2"));
        }

        [Test] public void DeployUnit_RoundTrips() => RoundTrips(new DeployUnit(P0, 2, new HexCoord(-1, 4)));
        [Test] public void DeployGenerator_RoundTrips() => RoundTrips(new DeployGenerator(P1, new HexCoord(0, 0)));
        [Test] public void CaptureHex_RoundTrips() => RoundTrips(new CaptureHex(P0, new HexCoord(5, -3)));
        [Test] public void BuildGenerator_RoundTrips() => RoundTrips(new BuildGenerator(P1, new HexCoord(2, 2)));

        [Test]
        public void Read_UnknownToken_Throws()
        {
            Assert.That(() => CommandWire.Read("Z 0"), Throws.Exception);
        }
    }
}
