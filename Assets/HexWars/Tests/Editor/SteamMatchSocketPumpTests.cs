#nullable enable
using System.Collections.Generic;
using NUnit.Framework;

namespace HexWars.Presentation.Tests
{
    /// <summary>
    /// Attempt isolation. A socket that is being abandoned keeps firing: its OnClose lands late, and
    /// so can a SEAT the server sent before it noticed. Before attempt ids those events were fed to
    /// whichever attempt happened to be live, so attempt A could close attempt B, seat it, or make it
    /// authenticate with a credential that was already spent. These tests stage exactly that.
    /// </summary>
    [TestFixture]
    public sealed class SteamMatchSocketPumpTests
    {
        const string Url = "wss://match.invalid/ws/v2";
        const string MatchId = "match-77";
        const string FirstCredential = "first-credential";
        const string SecondCredential = "second-credential";
        const double Clock = 1000;

        readonly List<string> _log = new List<string>();

        FakeSteamSocketDriver _driver = null!;
        SteamMatchSession _session = null!;
        SteamMatchSocketPump _pump = null!;

        [SetUp]
        public void SetUp()
        {
            _log.Clear();
            _driver = new FakeSteamSocketDriver();
            _session = new SteamMatchSession(Ticket(FirstCredential));
            _pump = new SteamMatchSocketPump(_session, _driver, message => _log.Add(message));
        }

        [TearDown]
        public void TearDown()
        {
            _pump.Dispose();
        }

        [Test]
        public void AnAbandonedAttempt_CannotClose_Seat_OrAuthenticateTheLiveOne()
        {
            var a = OpenAttempt();
            _driver.RaiseOpened(a);
            PumpAndExecute();
            Assert.That(_driver.SendsFor(a), Is.EqualTo(new[] { "AUTH " + MatchId + " " + FirstCredential }));

            // A opens and then says nothing: the handshake deadline is what ends it.
            _session.Tick(Clock);
            _session.Tick(Clock + SteamMatchSession.HandshakeTimeoutSeconds);
            Assert.That(Kinds(_session.Drain()), Is.EqualTo(new[]
            {
                SteamMatchSessionOutputKind.Reconnecting, SteamMatchSessionOutputKind.Retry,
            }));

            _session.CredentialRefreshed(Ticket(SecondCredential));
            var b = OpenAttempt();
            Assert.That(b, Is.Not.EqualTo(a), "every attempt needs its own id");

            // Everything A still had in flight arrives now, after B has begun.
            _driver.RaiseClosed(a, "late close");
            _driver.RaiseMessage(a, "SEAT 0");
            _driver.RaiseOpened(a);
            PumpAndExecute();

            Assert.That(_session.StaleEventsIgnored, Is.EqualTo(3));
            Assert.That(_session.State, Is.EqualTo(SteamMatchSessionState.Connecting),
                "a stale open must not authenticate the live attempt");
            Assert.That(_driver.SendsFor(b), Is.Empty, "a stale open must not send the live attempt AUTH");

            // B is untouched and runs its own handshake, with its own credential.
            _driver.RaiseOpened(b);
            PumpAndExecute();
            Assert.That(_driver.SendsFor(b), Is.EqualTo(new[] { "AUTH " + MatchId + " " + SecondCredential }));
            Assert.That(_driver.SendsFor(b)[0], Does.Not.Contain(FirstCredential),
                "a spent credential must never be replayed");

            _driver.RaiseMessage(b, "SEAT 1");
            var seated = PumpAndExecute();
            Assert.That(Kinds(seated), Is.EqualTo(new[]
            {
                SteamMatchSessionOutputKind.Reconnected, SteamMatchSessionOutputKind.Seat,
            }));
            Assert.That(seated[1].Seat, Is.EqualTo(1));
        }

        [Test]
        public void NothingReachesTheSessionBeforePump()
        {
            var a = OpenAttempt();

            _driver.RaiseOpened(a);
            _driver.RaiseMessage(a, "SEAT 0");

            Assert.That(_pump.QueuedEvents, Is.EqualTo(2));
            Assert.That(_session.Drain(), Is.Empty);

            var outputs = PumpAndExecute();
            Assert.That(Kinds(outputs), Is.EqualTo(new[]
            {
                SteamMatchSessionOutputKind.Send, SteamMatchSessionOutputKind.Seat,
            }));
            Assert.That(_pump.QueuedEvents, Is.Zero);
        }

        [Test]
        public void SocketErrors_AreLoggedAndNeverFedToTheSession()
        {
            var a = OpenAttempt();

            _driver.RaiseError(a, "tls handshake failed");
            PumpAndExecute();

            Assert.That(_session.Drain(), Is.Empty);
            Assert.That(_log, Has.Count.EqualTo(1));
            Assert.That(_log[0], Does.Contain("tls handshake failed"));
        }

        [Test]
        public void ADisposedPump_DeliversNothingMore()
        {
            var a = OpenAttempt();

            _pump.Dispose();
            _driver.RaiseOpened(a);
            _pump.Pump();

            Assert.That(_session.Drain(), Is.Empty);
            Assert.That(_session.State, Is.EqualTo(SteamMatchSessionState.Connecting));
        }

        [Test]
        public void ACloseThatNeverFinishes_LeavesItsTaskPendingSoTheCallerCanBoundTheWait()
        {
            _driver.CompleteClosesImmediately = false;
            var a = OpenAttempt();

            var closing = _driver.CloseAsync(a);

            Assert.That(closing.IsCompleted, Is.False,
                "reopening must be able to wait on the previous close, and to give up on it");
            _driver.CompleteCloses();
            Assert.That(closing.IsCompleted, Is.True);
        }

        [Test]
        public void AFloodOfFrames_IsAProtocolViolationAndTheQueueIsDropped()
        {
            var a = OpenAttempt();
            _driver.RaiseOpened(a);
            PumpAndExecute();

            // An unbounded queue lets the peer choose how much memory this process spends.
            for (var i = 0; i < 300; i++) _driver.RaiseMessage(a, SteamMatchProtocol.Ping);

            Assert.That(_pump.QueuedEvents, Is.EqualTo(1),
                "everything queued is dropped, leaving only the violation");

            var outputs = PumpAndExecute();
            Assert.That(Kinds(outputs), Is.EqualTo(new[]
            {
                SteamMatchSessionOutputKind.ProtocolViolation, SteamMatchSessionOutputKind.GiveUp,
            }));
            Assert.That(_session.State, Is.EqualTo(SteamMatchSessionState.Closed));
        }

        [Test]
        public void OnePumpReplaysAtMostItsCap_AndTheRestWaitForTheNextFrame()
        {
            var a = OpenAttempt();
            for (var i = 0; i < 100; i++) _driver.RaiseMessage(a, SteamMatchProtocol.Ping);
            Assert.That(_pump.QueuedEvents, Is.EqualTo(100));

            _pump.Pump();
            Assert.That(_pump.QueuedEvents,
                Is.EqualTo(100 - SteamMatchSocketPump.MaxEventsPerPump),
                "a burst costs a few frames of latency, not a stalled main thread");

            _pump.Pump();
            Assert.That(_pump.QueuedEvents, Is.Zero);
        }

        [Test]
        public void AFrameLargerThanTheCap_IsAProtocolViolation()
        {
            var a = OpenAttempt();
            _driver.RaiseOpened(a);
            PumpAndExecute();

            _driver.RaiseMessage(a, new string('x', SteamMatchSocketPump.MaxFrameBytes + 1));

            var outputs = PumpAndExecute();
            Assert.That(Kinds(outputs), Is.EqualTo(new[]
            {
                SteamMatchSessionOutputKind.ProtocolViolation, SteamMatchSessionOutputKind.GiveUp,
            }));
            Assert.That(_log, Has.Count.EqualTo(1));
            Assert.That(_log[0], Does.Contain("protocol violation"));
        }

        [Test]
        public void AnOversizedFrameFromAnAbandonedAttempt_DoesNotSilenceTheLiveOne()
        {
            var a = StartAndAbandonAnAttempt();
            var b = OpenAttempt();

            // A is being torn down and gets one last, enormous frame in.
            _driver.RaiseMessage(a, new string('x', SteamMatchSocketPump.MaxFrameBytes + 1));
            PumpAndExecute();

            Assert.That(_session.State, Is.EqualTo(SteamMatchSessionState.Connecting),
                "a dead attempt violation must not end the live one");

            _driver.RaiseOpened(b);
            PumpAndExecute();
            Assert.That(_driver.SendsFor(b), Is.EqualTo(new[] { "AUTH " + MatchId + " " + SecondCredential }));

            _driver.RaiseMessage(b, "SEAT 1");
            PumpAndExecute();
            Assert.That(_session.State, Is.EqualTo(SteamMatchSessionState.Seated));

            _driver.RaiseMessage(b, "START 0 9 7");
            Assert.That(Kinds(PumpAndExecute()), Is.EqualTo(new[] { SteamMatchSessionOutputKind.Start }));
        }

        [Test]
        public void AFloodFromAnAbandonedAttempt_DoesNotEraseTheLiveAttemptQueue()
        {
            var a = StartAndAbandonAnAttempt();
            var b = OpenAttempt();

            _driver.RaiseOpened(b);                       // B has an event waiting
            for (var i = 0; i < 300; i++) _driver.RaiseMessage(a, SteamMatchProtocol.Ping);

            Assert.That(_pump.QueuedEvents, Is.EqualTo(2),
                "B keeps its event; A keeps only its violation");

            PumpAndExecute();
            Assert.That(_driver.SendsFor(b), Is.EqualTo(new[] { "AUTH " + MatchId + " " + SecondCredential }),
                "the live attempt still authenticates");
            Assert.That(_session.State, Is.EqualTo(SteamMatchSessionState.Authenticating));
        }

        // ----- helpers ------------------------------------------------------------------------

        /// <summary>Runs one attempt to its handshake deadline and refreshes the credential.</summary>
        int StartAndAbandonAnAttempt()
        {
            var abandoned = OpenAttempt();
            _driver.RaiseOpened(abandoned);
            PumpAndExecute();

            _session.Tick(Clock);
            _session.Tick(Clock + SteamMatchSession.HandshakeTimeoutSeconds);
            _session.Drain();
            _session.CredentialRefreshed(Ticket(SecondCredential));
            return abandoned;
        }

        static SteamMatchTicket Ticket(string credential)
        {
            return new SteamMatchTicket(MatchId, Url, credential, 0);
        }

        int OpenAttempt()
        {
            var attemptId = _session.BeginAttempt();
            _driver.Open(Url, attemptId);
            return attemptId;
        }

        /// <summary>Exactly what the connection does each frame: pump, drain, carry out the sends.</summary>
        IReadOnlyList<SteamMatchSessionOutput> PumpAndExecute()
        {
            _pump.Pump();
            var outputs = _session.Drain();
            foreach (var output in outputs)
            {
                if (output.Kind == SteamMatchSessionOutputKind.Send)
                {
                    _driver.Send(_session.CurrentAttempt, output.Text);
                }
            }
            return outputs;
        }

        static SteamMatchSessionOutputKind[] Kinds(IReadOnlyList<SteamMatchSessionOutput> outputs)
        {
            var kinds = new SteamMatchSessionOutputKind[outputs.Count];
            for (var i = 0; i < outputs.Count; i++) kinds[i] = outputs[i].Kind;
            return kinds;
        }
    }
}
