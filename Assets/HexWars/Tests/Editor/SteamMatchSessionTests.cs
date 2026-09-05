#nullable enable
using System.Collections.Generic;
using NUnit.Framework;

namespace HexWars.Presentation.Tests
{
    [TestFixture]
    public sealed class SteamMatchSessionTests
    {
        const string MatchId = "match-77";
        const string Credential = "join-credential";
        const double Clock = 1000;

        SteamMatchSession _sut = null!;

        [SetUp]
        public void SetUp()
        {
            _sut = new SteamMatchSession(new SteamMatchTicket(MatchId, "wss://match.invalid/ws/v2", Credential, 0));
        }

        [Test]
        public void OnOpen_TheVeryFirstFrameIsAuth()
        {
            _sut.Opened();

            var outputs = _sut.Drain();
            Assert.That(outputs, Has.Count.EqualTo(1));
            Assert.That(outputs[0].Kind, Is.EqualTo(SteamMatchSessionOutputKind.Send));
            Assert.That(outputs[0].Text, Is.EqualTo("AUTH " + MatchId + " " + Credential));
            Assert.That(_sut.State, Is.EqualTo(SteamMatchSessionState.Authenticating));
        }

        [Test]
        public void NoFrameBeforeSeat_ReachesTheGame()
        {
            _sut.Opened();
            _sut.Drain();

            _sut.Frame("START 0 9 7");
            _sut.Frame("APPLY EndTurn 0");
            _sut.Frame("REJECT Illegal");
            _sut.Frame("CATALOG?");

            Assert.That(_sut.Drain(), Is.Empty, "an unauthenticated socket must not deal a board");

            _sut.Frame("SEAT 1");
            var seated = _sut.Drain();
            Assert.That(seated, Has.Count.EqualTo(1));
            Assert.That(seated[0].Kind, Is.EqualTo(SteamMatchSessionOutputKind.Seat));
            Assert.That(seated[0].Seat, Is.EqualTo(1));
            Assert.That(_sut.State, Is.EqualTo(SteamMatchSessionState.Seated));

            _sut.Frame("START 0 9 7");
            Assert.That(Kinds(_sut.Drain()), Is.EqualTo(new[] { SteamMatchSessionOutputKind.Start }));
        }

        [Test]
        public void AHungHandshake_EndsTheAttemptAndRetriesAfterFifteenSeconds()
        {
            _sut.Opened();
            _sut.Drain();

            _sut.Tick(Clock);                                   // the first tick anchors the deadline
            _sut.Tick(Clock + SteamMatchSession.HandshakeTimeoutSeconds - 0.1);
            Assert.That(_sut.Drain(), Is.Empty);
            Assert.That(_sut.State, Is.EqualTo(SteamMatchSessionState.Authenticating));

            _sut.Tick(Clock + SteamMatchSession.HandshakeTimeoutSeconds);

            var outputs = _sut.Drain();
            Assert.That(Kinds(outputs), Is.EqualTo(new[]
            {
                SteamMatchSessionOutputKind.Reconnecting, SteamMatchSessionOutputKind.Retry,
            }));
            Assert.That(outputs[1].DelaySeconds, Is.EqualTo(1));
            Assert.That(_sut.State, Is.EqualTo(SteamMatchSessionState.Closed));
        }

        [Test]
        public void AuthFail_GivesUpWithTheServerCode()
        {
            _sut.Opened();
            _sut.Drain();

            _sut.Frame("AUTH FAIL expired");

            var outputs = _sut.Drain();
            Assert.That(Kinds(outputs), Is.EqualTo(new[] { SteamMatchSessionOutputKind.AuthFailed }));
            Assert.That(outputs[0].Text, Is.EqualTo("expired"));
            Assert.That(_sut.AuthFailed, Is.True);

            _sut.Closed();
            Assert.That(_sut.Drain(), Is.Empty, "a refused credential must never be retried");
        }

        [Test]
        public void Ping_IsAnsweredWithPong()
        {
            _sut.Opened();
            _sut.Frame("SEAT 0");
            _sut.Drain();

            _sut.Frame(SteamMatchProtocol.Ping);

            var outputs = _sut.Drain();
            Assert.That(Kinds(outputs), Is.EqualTo(new[] { SteamMatchSessionOutputKind.Send }));
            Assert.That(outputs[0].Text, Is.EqualTo(SteamMatchProtocol.Pong));
        }

        [Test]
        public void APingBeforeSeat_IsStillAnswered()
        {
            _sut.Opened();
            _sut.Drain();

            _sut.Frame(SteamMatchProtocol.Ping);

            Assert.That(_sut.Drain()[0].Text, Is.EqualTo(SteamMatchProtocol.Pong));
        }

        [Test]
        public void AnAnnouncedRestart_RetriesImmediatelyAndThenBacksOff()
        {
            _sut.Opened();
            _sut.Frame("SEAT 0");
            _sut.Frame("START 0 9 7");
            _sut.Drain();

            _sut.Frame(SteamMatchProtocol.ServerRestart);
            _sut.Closed();

            var planned = _sut.Drain();
            Assert.That(Kinds(planned), Is.EqualTo(new[]
            {
                SteamMatchSessionOutputKind.Reconnecting, SteamMatchSessionOutputKind.Retry,
            }));
            Assert.That(planned[1].DelaySeconds, Is.Zero, "a planned restart reconnects at once");

            _sut.BeginAttempt();
            _sut.Opened();
            _sut.Drain();
            _sut.Closed();

            var unplanned = _sut.Drain();
            Assert.That(unplanned[1].Kind, Is.EqualTo(SteamMatchSessionOutputKind.Retry));
            Assert.That(unplanned[1].DelaySeconds, Is.EqualTo(2), "the backoff resumes where it was");
        }

        [Test]
        public void RepeatedDropsBackOffOneTwoFourEightThenFifteen()
        {
            _sut.Opened();
            _sut.Frame("SEAT 0");
            _sut.Frame("START 0 9 7");
            _sut.Drain();

            var delays = new List<double>();
            for (var i = 0; i < 6; i++)
            {
                _sut.Closed();
                foreach (var output in _sut.Drain())
                {
                    if (output.Kind == SteamMatchSessionOutputKind.Retry) delays.Add(output.DelaySeconds);
                }
                _sut.BeginAttempt();
                _sut.Opened();
                _sut.Drain();
            }

            Assert.That(delays, Is.EqualTo(new double[] { 1, 2, 4, 8, 15, 15 }));
        }

        [Test]
        public void AFailedCredentialRefresh_GivesUp()
        {
            _sut.Opened();
            _sut.Frame("SEAT 0");
            _sut.Frame("START 0 9 7");
            _sut.Closed();
            _sut.Drain();

            _sut.CredentialRefreshed(null);

            Assert.That(Kinds(_sut.Drain()), Is.EqualTo(new[] { SteamMatchSessionOutputKind.GiveUp }));
            Assert.That(_sut.State, Is.EqualTo(SteamMatchSessionState.Closed));
        }

        [Test]
        public void AFreshTicket_IsUsedForTheNextAuthFrame()
        {
            _sut.Opened();
            _sut.Frame("SEAT 0");
            _sut.Frame("START 0 9 7");
            _sut.Closed();
            _sut.Drain();

            _sut.CredentialRefreshed(new SteamMatchTicket(MatchId, "wss://match.invalid/ws/v2", "second-credential", 0));
            _sut.BeginAttempt();
            _sut.Opened();

            Assert.That(_sut.Drain()[0].Text, Is.EqualTo("AUTH " + MatchId + " second-credential"));
        }

        [Test]
        public void AFirstDropBeforeAnyGame_StopsInsteadOfReconnecting()
        {
            _sut.Opened();
            _sut.Drain();

            _sut.Closed();

            Assert.That(Kinds(_sut.Drain()), Is.EqualTo(new[] { SteamMatchSessionOutputKind.Closed }));
        }

        [Test]
        public void AFullMatch_ReportsSeatFullAndStops()
        {
            _sut.Opened();
            _sut.Drain();

            _sut.Frame("SEAT FULL");

            Assert.That(Kinds(_sut.Drain()), Is.EqualTo(new[] { SteamMatchSessionOutputKind.SeatFull }));

            _sut.Closed();
            Assert.That(_sut.Drain(), Is.Empty);
        }

        [Test]
        public void ASeatAfterADrop_ReportsReconnectedFirst()
        {
            _sut.Opened();
            _sut.Frame("SEAT 0");
            _sut.Frame("START 0 9 7");
            _sut.Closed();
            _sut.BeginAttempt();
            _sut.Opened();
            _sut.Drain();

            _sut.Frame("SEAT 0");

            Assert.That(Kinds(_sut.Drain()), Is.EqualTo(new[]
            {
                SteamMatchSessionOutputKind.Reconnected, SteamMatchSessionOutputKind.Seat,
            }));
            Assert.That(_sut.Attempt, Is.Zero);
        }

        [Test]
        public void AConnectThatNeverEvenOpens_IsAbandonedAfterFifteenSeconds()
        {
            // No Opened, no frame, no close: the socket is simply hanging on connect. Before the
            // deadline covered connect, nothing at all ended this attempt and the game sat on
            // "Connecting..." for good.
            _sut.BeginAttempt();

            _sut.Tick(Clock);
            _sut.Tick(Clock + SteamMatchSession.HandshakeTimeoutSeconds - 0.1);
            Assert.That(_sut.Drain(), Is.Empty);

            _sut.Tick(Clock + SteamMatchSession.HandshakeTimeoutSeconds);

            Assert.That(Kinds(_sut.Drain()), Is.EqualTo(new[]
            {
                SteamMatchSessionOutputKind.Reconnecting, SteamMatchSessionOutputKind.Retry,
            }));
            Assert.That(_sut.State, Is.EqualTo(SteamMatchSessionState.Closed));
        }

        [Test]
        public void OpeningTheSocket_DoesNotRestartTheDeadline()
        {
            _sut.BeginAttempt();
            _sut.Tick(Clock);                     // anchored while the socket is still connecting

            _sut.Tick(Clock + 10);                // ten seconds of connect, then the socket opens
            _sut.Opened();
            _sut.Drain();

            _sut.Tick(Clock + SteamMatchSession.HandshakeTimeoutSeconds);

            Assert.That(Kinds(_sut.Drain()), Is.EqualTo(new[]
            {
                SteamMatchSessionOutputKind.Reconnecting, SteamMatchSessionOutputKind.Retry,
            }), "one deadline covers connect, AUTH and SEAT together");
        }

        [Test]
        public void AMalformedSeat_IsAProtocolViolationAndGivesUp()
        {
            _sut.Opened();
            _sut.Drain();

            _sut.Frame("SEAT garbage");

            var outputs = _sut.Drain();
            Assert.That(Kinds(outputs), Is.EqualTo(new[]
            {
                SteamMatchSessionOutputKind.ProtocolViolation, SteamMatchSessionOutputKind.GiveUp,
            }));
            Assert.That(outputs[0].Text, Is.EqualTo(SteamMatchProtocol.ProtocolCloseCode));
            Assert.That(_sut.State, Is.EqualTo(SteamMatchSessionState.Closed));
        }

        [Test]
        public void ASeatOutOfRange_NeverDealsASeat()
        {
            _sut.Opened();
            _sut.Drain();

            _sut.Frame("SEAT 2");

            var kinds = Kinds(_sut.Drain());
            Assert.That(kinds, Has.No.Member(SteamMatchSessionOutputKind.Seat),
                "a seat index the game has no player for must never reach the barracks cache");
            Assert.That(kinds, Is.EqualTo(new[]
            {
                SteamMatchSessionOutputKind.ProtocolViolation, SteamMatchSessionOutputKind.GiveUp,
            }));
        }

        [Test]
        public void ASeatBeforeOurAuthWentOut_IsIgnoredAndTheDeadlineStillFires()
        {
            _sut.BeginAttempt();                  // connecting: AUTH has not been sent yet

            _sut.Frame("SEAT 0");

            Assert.That(_sut.Drain(), Is.Empty, "a socket that was never authenticated must not seat");
            Assert.That(_sut.State, Is.EqualTo(SteamMatchSessionState.Connecting));

            _sut.Tick(Clock);
            _sut.Tick(Clock + SteamMatchSession.HandshakeTimeoutSeconds);

            Assert.That(Kinds(_sut.Drain()), Is.EqualTo(new[]
            {
                SteamMatchSessionOutputKind.Reconnecting, SteamMatchSessionOutputKind.Retry,
            }), "a premature SEAT must never disarm the handshake deadline");
        }

        [Test]
        public void AnUnknownAuthFailCode_IsReportedAsUnknownAndTheAuthFrameIsNeverRendered()
        {
            _sut.Opened();
            var authSend = _sut.Drain()[0];

            _sut.Frame("AUTH FAIL " + Credential + " leaked\nsecond line");

            var outputs = _sut.Drain();
            Assert.That(Kinds(outputs), Is.EqualTo(new[] { SteamMatchSessionOutputKind.AuthFailed }));
            Assert.That(outputs[0].Text, Is.EqualTo(SteamMatchProtocol.AuthFailUnknown));
            Assert.That(outputs[0].Text, Does.Not.Contain(Credential));

            var rendered = authSend.ToString();
            Assert.That(rendered, Does.Contain("AUTH <redacted>"));
            Assert.That(rendered, Does.Not.Contain(Credential));
            Assert.That(rendered, Does.Not.Contain(MatchId));
        }

        [Test]
        public void EveryAttemptHasItsOwnId()
        {
            var first = _sut.CurrentAttempt;
            var second = _sut.BeginAttempt();
            var third = _sut.BeginAttempt();

            Assert.That(second, Is.Not.EqualTo(first));
            Assert.That(third, Is.Not.EqualTo(second));
            Assert.That(_sut.CurrentAttempt, Is.EqualTo(third));

            _sut.Opened(second);
            _sut.Frame(second, "SEAT 0");
            _sut.Closed(second);

            Assert.That(_sut.Drain(), Is.Empty);
            Assert.That(_sut.StaleEventsIgnored, Is.EqualTo(3));
        }

        static SteamMatchSessionOutputKind[] Kinds(IReadOnlyList<SteamMatchSessionOutput> outputs)
        {
            var kinds = new SteamMatchSessionOutputKind[outputs.Count];
            for (var i = 0; i < outputs.Count; i++) kinds[i] = outputs[i].Kind;
            return kinds;
        }
    }
}
