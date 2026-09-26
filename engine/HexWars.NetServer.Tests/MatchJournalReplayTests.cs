using HexWars.Engine;
using HexWars.NetServer.Persistence;
using HexWars.NetServer.Tests.Fixtures;
using Microsoft.Extensions.Logging.Abstractions;
using Npgsql;
using NUnit.Framework;

namespace HexWars.NetServer.Tests
{
    /// <summary>
    /// The exit gate for the persistence layer: a match written by one process is played back byte-identically
    /// by another that shares nothing with it but the database.
    ///
    /// This is the claim the whole design rests on. A match host can be killed mid-game and replaced, and the
    /// replacement must reach the exact state the dead one was in. Testing that inside one object graph would
    /// prove almost nothing, so the reader here builds its own connection pool and its own store, reads the
    /// journal cold, and replays it through the same deterministic engine the client runs.
    ///
    /// The two commands are the pair SelfTest.cs uses, and they are legal on the default seed for the same
    /// reason: the deterministic army placement puts a striker of Player0 within reach of one of Player1.
    /// </summary>
    [TestFixture]
    public class MatchJournalReplayTests
    {
        static readonly DateTimeOffset Created = new(2026, 9, 4, 12, 0, 0, TimeSpan.Zero);

        static readonly Command FirstMove = new MoveUnit(PlayerId.Player0, 2, new HexCoord(3, 0));
        static readonly Command ThenAttack = new AttackUnit(PlayerId.Player0, 2, 5);

        PostgresTestDatabase _db = null!;

        [SetUp]
        public async Task StartFromAFreshSchema()
        {
            _db = await PostgresTestDatabase.GetAsync();
            await _db.ResetAsync();
            await _db.ApplyMigrationsAsync();
        }

        [Test]
        public async Task AMatchWrittenByOneProcess_ReplaysToTheSameStateInAnother()
        {
            Guid matchId;
            string seat0 = "76561190000000001";
            string seat1 = "76561190000000002";

            // ---- the process that played the match ----
            await using (NpgsqlDataSource writerPool = NpgsqlDataSource.Create(_db.ConnectionString))
            {
                var writer = new PostgresMatchStore(writerPool, NullLogger<PostgresMatchStore>.Instance);

                CreateMatchResult created = await writer.CreateMatchForLobbyAsync(new CreateMatchRequest(
                    "109775240000000042", GameSetup.Default.ToWire(), "hexwars-engine/1", 2, "test-build",
                    new[] { (seat0, 0), (seat1, 1) }, Created), CancellationToken.None);
                matchId = created.Match.MatchId;

                string startReplay =
                    ReplayFile.Write(GameFactory.Build(GameSetup.Default), Array.Empty<Command>());
                Assert.That(
                    await writer.TryStartMatchAsync(
                        matchId, startReplay, Created.AddMinutes(1), CancellationToken.None),
                    Is.True);

                Assert.That(
                    (await writer.AppendCommandAsync(matchId, 1, CommandWire.Write(FirstMove), seat0,
                        Created.AddMinutes(2), CancellationToken.None)).Status,
                    Is.EqualTo(AppendStatus.Appended));
                Assert.That(
                    (await writer.AppendCommandAsync(matchId, 2, CommandWire.Write(ThenAttack), seat0,
                        Created.AddMinutes(3), CancellationToken.None)).Status,
                    Is.EqualTo(AppendStatus.Appended));
            }

            // ---- the process that took over ----
            MatchJournal journal;
            await using (NpgsqlDataSource readerPool = NpgsqlDataSource.Create(_db.ConnectionString))
            {
                var reader = new PostgresMatchStore(readerPool, NullLogger<PostgresMatchStore>.Instance);
                journal = (await reader.LoadJournalAsync(matchId, CancellationToken.None))!;
            }

            Assert.That(journal, Is.Not.Null);
            Assert.That(journal.Match.Status, Is.EqualTo(MatchStatus.Active));
            Assert.That(journal.Match.SetupWire, Is.EqualTo(GameSetup.Default.ToWire()));
            Assert.That(journal.Match.StartReplay, Is.Not.Null);
            Assert.That(journal.Commands, Has.Count.EqualTo(2));
            Assert.That(journal.Commands.Select(c => c.Sequence), Is.EqualTo(new[] { 1, 2 }));
            Assert.That(journal.Players.Select(p => p.SteamId), Is.EqualTo(new[] { seat0, seat1 }));

            GameState recovered = ReplayFile.Read(journal.Match.StartReplay!).Start;
            foreach (PersistedCommand stored in journal.Commands)
            {
                var result = GameEngine.Apply(recovered, CommandWire.Read(stored.CommandWire));
                Assert.That(result.Success, Is.True,
                    "a journalled command was rejected on replay: " + stored.CommandWire + " (" + result.Reason + ")");
                recovered = result.NewState;
            }

            // The comparison is the serialised state rather than a handful of fields: it is the same text the
            // server would deal to a reconnecting client, so an equal string means an equal game.
            GameState direct = GameFactory.Build(GameSetup.Default);
            direct = GameEngine.Apply(direct, FirstMove).NewState;
            direct = GameEngine.Apply(direct, ThenAttack).NewState;

            Assert.That(ReplayFile.Write(recovered, Array.Empty<Command>()),
                Is.EqualTo(ReplayFile.Write(direct, Array.Empty<Command>())));
        }
    }
}
