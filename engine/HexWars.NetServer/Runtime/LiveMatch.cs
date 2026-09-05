using System.Collections.Concurrent;
using HexWars.Engine;
using HexWars.NetServer.Persistence;

namespace HexWars.NetServer.Runtime
{
    /// <summary>
    /// One match as this process is currently playing it: the projection of the journal, plus the sockets
    /// watching it and the gate that keeps them from overlapping.
    ///
    /// The projection is a cache, never a source of truth. Everything in it can be rebuilt from the store by
    /// <see cref="FromJournal"/>, which is what makes a restart, a failover and a stale-write recovery the
    /// same operation. <see cref="Stale"/> is the admission that the cache and the database may have parted
    /// company; the coordinator rebuilds before it acts again rather than trying to reconcile.
    ///
    /// <see cref="Gate"/> is what makes the durable ordering rule expressible at all. Evaluating a command,
    /// appending it and advancing the state have to be one indivisible step per match, or two players could
    /// both be told their command was accepted at the same sequence. It is per match rather than global so a
    /// slow write in one game cannot stall every other game in the process.
    /// </summary>
    public sealed class LiveMatch
    {
        LiveMatch(Guid matchId) => MatchId = matchId;

        public Guid MatchId { get; }

        /// <summary>The lobby picks the start state was, or will be, built from.</summary>
        public GameSetup Setup { get; private set; }

        public MatchStatus Status { get; set; }

        /// <summary>Steam id to seat number. Both seats exist from the moment the match is allocated.</summary>
        public IReadOnlyDictionary<string, int> Seats { get; private set; } =
            new Dictionary<string, int>(StringComparer.Ordinal);

        /// <summary>The barracks each seat chose, by seat number. Missing until that player sends one.</summary>
        public Dictionary<int, IReadOnlyList<UnitTemplate>> Catalogs { get; } = new();

        /// <summary>The authoritative start state, once the match is active.</summary>
        public GameState? Start { get; set; }

        /// <summary>The state every accepted command so far has produced.</summary>
        public GameState? State { get; set; }

        /// <summary>Every accepted command, in order. Paired with <see cref="Start"/> it replays to
        /// <see cref="State"/> exactly.</summary>
        public List<Command> Log { get; } = new();

        /// <summary>The highest sequence known to be in the journal. The next append asks for this plus one.</summary>
        public int LastSequence { get; set; }

        /// <summary>
        /// When this match reached a terminal status, as the record has it. Null while it is being played.
        ///
        /// The projection carries it because the reconnect window is judged twice: once by the credential
        /// service, and again under the match gate immediately before a seat is dealt. The second check is
        /// the one that cannot be overtaken by a slow load, and it has nowhere else to read the instant
        /// from without going back to the store while holding the gate.
        /// </summary>
        public DateTimeOffset? CompletedAt { get; set; }

        public string EngineVersion { get; private set; } = string.Empty;

        public int ProtocolVersion { get; private set; }

        /// <summary>
        /// Connection id to Steam id, for every socket currently watching this match.
        ///
        /// Concurrent rather than a plain dictionary because the counters and the shutdown broadcast read it
        /// from outside the gate, and a dictionary being read while another thread seats a player is not a
        /// race that shows up in testing - it shows up once, in production, as a corrupted lookup.
        /// </summary>
        public ConcurrentDictionary<string, string> Connections { get; } = new(StringComparer.Ordinal);

        /// <summary>Held for the whole of every operation that can change this match.</summary>
        public SemaphoreSlim Gate { get; } = new(1, 1);

        /// <summary>Set when a durable write failed or was refused: the projection must be rebuilt from the
        /// journal before anything else is decided from it.</summary>
        public bool Stale { get; set; }

        /// <summary>When a connection was last added or removed. An idle match is swept from memory a while
        /// after its last socket goes.</summary>
        public DateTimeOffset LastConnectionAt { get; set; }

        /// <summary>
        /// Rebuilds the projection from the journal: seats, catalogues, and - once the match is active - the
        /// start state fast-forwarded through every stored command.
        ///
        /// The two refusals are deliberate. A gap in the sequence, or a stored command the engine will not
        /// accept, both mean the journal describes a game that cannot have been played. Carrying on would
        /// deal a state to the clients that disagrees with the record, so this stops instead and lets the
        /// caller decide what to do with a match it cannot honestly host.
        /// </summary>
        /// <exception cref="InvalidOperationException">The journal has a sequence gap, is missing the start
        /// replay of an active match, or holds a command that does not replay.</exception>
        public static LiveMatch FromJournal(MatchJournal journal)
        {
            ArgumentNullException.ThrowIfNull(journal);

            var live = new LiveMatch(journal.Match.MatchId)
            {
                Setup = GameSetup.Parse(journal.Match.SetupWire),
                Status = journal.Match.Status,
                CompletedAt = journal.Match.CompletedAt,
                EngineVersion = journal.Match.EngineVersion,
                ProtocolVersion = journal.Match.ProtocolVersion,
            };

            var seats = new Dictionary<string, int>(StringComparer.Ordinal);
            foreach (PersistedPlayer player in journal.Players)
            {
                seats[player.SteamId] = player.Seat;
                if (player.CatalogWire is not null)
                    live.Catalogs[player.Seat] = ReadCatalog(player.CatalogWire);
            }

            live.Seats = seats;

            // Every journal that has a start state is replayed, not only the active ones. A completed
            // match is still handed to clients - the terminal reconnect window deals its final position -
            // and a projection built by skipping the log would deal the opening of a game that is over.
            if (journal.Match.StartReplay is null)
            {
                if (journal.Match.Status == MatchStatus.Active)
                    throw new InvalidOperationException(
                        "match " + journal.Match.MatchId + " is active with no start replay");

                // Waiting: there is nothing to replay yet, and the seats and catalogues above are the
                // whole projection.
                return live;
            }

            GameState state = ReplayFile.Read(journal.Match.StartReplay).Start;
            live.Start = state;

            int expected = 1;
            foreach (PersistedCommand stored in journal.Commands)
            {
                if (stored.Sequence != expected)
                    throw new InvalidOperationException(
                        "sequence gap in match " + journal.Match.MatchId + ": expected sequence "
                        + expected + " but the journal holds " + stored.Sequence);

                Command command;
                try
                {
                    command = CommandWire.Read(stored.CommandWire);
                }
                catch (FormatException malformed)
                {
                    throw new InvalidOperationException(
                        "replay failed at sequence " + stored.Sequence + " of match "
                        + journal.Match.MatchId + ": the stored command does not parse", malformed);
                }

                // The row says who sent it and the payload says who played it. When they disagree the
                // journal describes a move somebody was not entitled to make, and replaying it would hand
                // that move to whichever seat the payload names. Checked here as well as in the recovery
                // service so the plain loader cannot build a projection the verified one would refuse.
                if (!seats.TryGetValue(stored.IssuerSteamId, out int issuerSeat))
                    throw new InvalidOperationException(
                        "replay failed at sequence " + stored.Sequence + " of match "
                        + journal.Match.MatchId + ": it was written by a player who holds no seat");

                if (issuerSeat != (int)command.Issuer)
                    throw new InvalidOperationException(
                        "replay failed at sequence " + stored.Sequence + " of match "
                        + journal.Match.MatchId + ": it claims seat " + (int)command.Issuer
                        + " but its row belongs to seat " + issuerSeat);

                Result applied = GameEngine.Apply(state, command);
                if (!applied.Success)
                    throw new InvalidOperationException(
                        "replay failed at sequence " + stored.Sequence + " of match "
                        + journal.Match.MatchId + ": the engine refused it with " + applied.Reason);

                state = applied.NewState;
                live.Log.Add(command);
                expected++;
            }

            live.State = state;
            live.LastSequence = expected - 1;
            return live;
        }

        /// <summary>
        /// The START payload for a connect or a reconnect: the start state plus every accepted command.
        ///
        /// Never the live state on its own. The replay start-state encoding assumes fresh, full-health units
        /// and drops the per-turn tracking, which is exact for a state nobody has played yet and silently
        /// wrong for one somebody has - see the class comment on the legacy hub, which learned this the hard
        /// way. Replaying the log through the same engine the client runs reconstructs the position exactly.
        /// </summary>
        public string StartReplayText() => ReplayFile.Write(Start!, Log);

        /// <summary>
        /// Takes on everything a freshly loaded projection knows, in place.
        ///
        /// In place rather than by swapping the object, because the gate being held is what makes a rebuild
        /// safe: a caller that replaced the cached instance would leave every waiter holding a lock on an
        /// object nobody else is using any more. The sockets and the gate belong to the process, not to the
        /// journal, so they survive the rebuild untouched.
        /// </summary>
        public void ReplaceProjectionWith(LiveMatch rebuilt)
        {
            ArgumentNullException.ThrowIfNull(rebuilt);

            if (rebuilt.MatchId != MatchId)
                throw new ArgumentException(
                    "a projection for match " + rebuilt.MatchId + " cannot replace match " + MatchId,
                    nameof(rebuilt));

            Setup = rebuilt.Setup;
            Status = rebuilt.Status;
            CompletedAt = rebuilt.CompletedAt;
            Seats = rebuilt.Seats;
            EngineVersion = rebuilt.EngineVersion;
            ProtocolVersion = rebuilt.ProtocolVersion;
            Start = rebuilt.Start;
            State = rebuilt.State;
            LastSequence = rebuilt.LastSequence;

            Catalogs.Clear();
            foreach (KeyValuePair<int, IReadOnlyList<UnitTemplate>> catalog in rebuilt.Catalogs)
                Catalogs[catalog.Key] = catalog.Value;

            Log.Clear();
            Log.AddRange(rebuilt.Log);

            Stale = false;
        }

        /// <summary>A barracks that no longer parses is not a reason to refuse the match. The same fallback
        /// the legacy hub applies to a garbled CATALOG frame applies to a stored one.</summary>
        static IReadOnlyList<UnitTemplate> ReadCatalog(string catalogWire)
        {
            try
            {
                return BarracksWire.Read(catalogWire);
            }
            catch (FormatException)
            {
                return BarracksCatalog.DefaultTemplates;
            }
        }
    }
}
