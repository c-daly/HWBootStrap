using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    /// <summary>
    /// The line-of-text envelope both ends speak over the WebSocket: "TYPE payload". WebSocket preserves
    /// message boundaries, so a payload may itself be multi-line (the START start-state dump). Parsing is
    /// "split on the first space"; the typed builders keep server and client in lockstep.
    /// </summary>
    public class NetProtocolTests
    {
        private const PlayerId P0 = PlayerId.Player0;
        private const PlayerId P1 = PlayerId.Player1;

        [Test]
        public void Apply_RoundTripsThroughParseAndCommandWire()
        {
            var cmd = new MoveUnit(P0, 7, new HexCoord(3, -2));
            var msg = NetProtocol.Parse(NetProtocol.Apply(cmd));
            Assert.That(msg.Type, Is.EqualTo("APPLY"));
            Assert.That(CommandWire.Read(msg.Payload), Is.EqualTo(cmd));
        }

        [Test]
        public void Cmd_RoundTripsThroughParseAndCommandWire()
        {
            var cmd = new AttackUnit(P1, 2, 5);
            var msg = NetProtocol.Parse(NetProtocol.Cmd(cmd));
            Assert.That(msg.Type, Is.EqualTo("CMD"));
            Assert.That(CommandWire.Read(msg.Payload), Is.EqualTo(cmd));
        }

        [Test]
        public void Start_PreservesMultiLinePayloadExactly()
        {
            string startState = "META 3 0 1 0\nTILES 1\n0 0 0 0\nPLAYER 0 10 0 0 0";
            var msg = NetProtocol.Parse(NetProtocol.Start(startState));
            Assert.That(msg.Type, Is.EqualTo("START"));
            Assert.That(msg.Payload, Is.EqualTo(startState));
        }

        [Test]
        public void Seat_FormatsAndParses()
        {
            var msg = NetProtocol.Parse(NetProtocol.Seat(P1));
            Assert.That(msg.Type, Is.EqualTo("SEAT"));
            Assert.That(msg.Payload, Is.EqualTo("1"));
        }

        [Test]
        public void Reject_CarriesReasonName()
        {
            var msg = NetProtocol.Parse(NetProtocol.Reject(RejectionReason.NotYourTurn));
            Assert.That(msg.Type, Is.EqualTo("REJECT"));
            Assert.That(msg.Payload, Is.EqualTo("NotYourTurn"));
        }
    }
}
