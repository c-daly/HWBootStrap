using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEditor;
using UnityEngine;

namespace HexWars.Presentation.EditorTools.MlLab
{
    public enum MlRunState { Unknown, Created, Running, Stopping, Stopped, Completed, Failed }

    public sealed class MlRunStatus
    {
        public bool Ok { get; private set; }
        public string RunDirectory { get; private set; }
        public string RunName { get; private set; }
        public MlRunState State { get; private set; }
        public int Pid { get; private set; }
        public long Step { get; private set; }
        public long TargetStep { get; private set; }
        public string LatestCheckpoint { get; private set; }
        public string LatestEvaluation { get; private set; }
        public double Throughput { get; private set; }
        public bool TrackerDegraded { get; private set; }
        public string LearnerSeat { get; private set; }
        public int Seat0Episodes { get; private set; }
        public int Seat1Episodes { get; private set; }
        public bool SeatAuditReadable { get; private set; }
        public bool SeatAuditBalanced { get; private set; }
        public string SeatAuditWarning { get; private set; }
        public bool SeatAuditBalanceApplicable => string.Equals(
            LearnerSeat, "alternating", StringComparison.OrdinalIgnoreCase);
        public bool SeatAuditShowsWarning => SeatAuditBalanceApplicable && IsTerminal(State) &&
            SeatAuditReadable && !SeatAuditBalanced &&
            !string.IsNullOrWhiteSpace(SeatAuditWarning);
        public bool SeatAuditShowsInfo => !SeatAuditShowsWarning &&
            (!SeatAuditReadable || (SeatAuditBalanceApplicable &&
             !IsTerminal(State) && !SeatAuditBalanced));
        public string Error { get; private set; }

        public static MlRunStatus Parse(string json)
        {
            if (string.IsNullOrWhiteSpace(json)) throw new ArgumentException("Status JSON is empty.", nameof(json));
            var envelope = JsonUtility.FromJson<Envelope>(json);
            if (envelope == null) throw new FormatException("Status JSON could not be parsed.");
            var result = envelope.result ?? new Result();
            var run = result.run ?? envelope.run ?? new Run();
            var config = run.config ?? new Config();
            var seatAudit = result.seat_audit ?? new SeatAudit();
            return new MlRunStatus
            {
                Ok = envelope.ok,
                RunDirectory = result.run_dir ?? string.Empty,
                RunName = First(run.run_name, config.run_name),
                State = ParseState(run.state),
                Pid = run.pid,
                Step = run.timesteps != 0 ? run.timesteps : run.step,
                TargetStep = run.target_step != 0 ? run.target_step : config.total_timesteps,
                LatestCheckpoint = run.latest_checkpoint ?? string.Empty,
                LatestEvaluation = run.latest_evaluation ?? string.Empty,
                Throughput = run.throughput,
                TrackerDegraded = run.tracker_degraded || HasTrackerFailure(run.tracker_status),
                LearnerSeat = config.learner_seat ?? string.Empty,
                Seat0Episodes = seatAudit.seat_0_episodes,
                Seat1Episodes = seatAudit.seat_1_episodes,
                SeatAuditReadable = seatAudit.readable,
                SeatAuditBalanced = seatAudit.balanced,
                SeatAuditWarning = seatAudit.warning ?? string.Empty,
                Error = result.message ?? string.Empty,
            };
        }

        public static MlRunState ParseState(string value)
        {
            switch ((value ?? string.Empty).Trim().ToLowerInvariant())
            {
                case "created": return MlRunState.Created;
                case "running": return MlRunState.Running;
                case "stopping": return MlRunState.Stopping;
                case "stopped": return MlRunState.Stopped;
                case "completed": return MlRunState.Completed;
                case "failed": return MlRunState.Failed;
                default: return MlRunState.Unknown;
            }
        }

        static string First(string first, string second) => !string.IsNullOrEmpty(first) ? first : second ?? string.Empty;

        static bool IsTerminal(MlRunState state) =>
            state == MlRunState.Stopped ||
            state == MlRunState.Completed ||
            state == MlRunState.Failed;

        static bool HasTrackerFailure(Tracker[] trackers)
        {
            if (trackers == null) return false;
            foreach (var tracker in trackers)
                if (tracker != null && (!tracker.ok || string.Equals(
                    tracker.status, "degraded", StringComparison.OrdinalIgnoreCase))) return true;
            return false;
        }

        [Serializable] sealed class Envelope { public bool ok; public Result result; public Run run; }
        [Serializable] sealed class Result
        {
            public string run_dir;
            public Run run;
            public SeatAudit seat_audit;
            public string message;
        }
        [Serializable] sealed class Config
        {
            public string run_name;
            public long total_timesteps;
            public string learner_seat;
        }
        [Serializable] sealed class SeatAudit
        {
            public int seat_0_episodes;
            public int seat_1_episodes;
            public bool readable;
            public bool balanced;
            public string warning;
        }
        [Serializable] sealed class Tracker { public bool ok = true; public string status; }
        [Serializable] sealed class Run
        {
            public string run_name;
            public string state;
            public int pid;
            public long timesteps;
            public long step;
            public long target_step;
            public string latest_checkpoint;
            public string latest_evaluation;
            public double throughput;
            public bool tracker_degraded;
            public Tracker[] tracker_status;
            public Config config;
        }
    }

    public sealed class MlLogBuffer
    {
        readonly int _capacity;
        readonly Queue<string> _lines;
        readonly object _gate = new object();

        public MlLogBuffer(int capacity)
        {
            if (capacity <= 0) throw new ArgumentOutOfRangeException(nameof(capacity));
            _capacity = capacity;
            _lines = new Queue<string>(capacity);
        }

        public string[] Lines { get { lock (_gate) return _lines.ToArray(); } }
        public void Add(string line)
        {
            if (line == null) return;
            lock (_gate)
            {
                while (_lines.Count >= _capacity) _lines.Dequeue();
                _lines.Enqueue(line);
            }
        }

        public void Clear() { lock (_gate) _lines.Clear(); }
    }

    public readonly struct MlRunAttachment
    {
        const string RunKey = "HexWars.MlLab.ActiveRun";
        const string PidKey = "HexWars.MlLab.ActivePid";

        public readonly string RunDirectory;
        public bool Exists => !string.IsNullOrWhiteSpace(RunDirectory);

        MlRunAttachment(string runDirectory)
        {
            RunDirectory = runDirectory ?? string.Empty;
        }

        public static void Remember(string runDirectory)
        {
            if (string.IsNullOrWhiteSpace(runDirectory)) throw new ArgumentException(
                "Active run directory is required.", nameof(runDirectory));
            SessionState.SetString(RunKey, runDirectory);
            SessionState.EraseInt(PidKey);
        }

        public static MlRunAttachment Restore() => new MlRunAttachment(
            SessionState.GetString(RunKey, string.Empty));

        public static void Forget()
        {
            SessionState.EraseString(RunKey);
            SessionState.EraseInt(PidKey);
        }
    }

    public static class MlRunLog
    {
        public static string[] ReadTail(string runDirectory, int maxLines)
        {
            if (string.IsNullOrWhiteSpace(runDirectory) || maxLines <= 0) return Array.Empty<string>();
            return MlSharedFileSnapshot.ReadTail(Path.Combine(runDirectory, "train.log"), maxLines);
        }
    }

    public static class MlSharedFileSnapshot
    {
        public const int TailReadLimitBytes = 256 * 1024;
        public const int LastLineReadLimitBytes = 16 * 1024;

        public static string[] ReadTail(string path, int maxLines) =>
            ReadTail(path, maxLines, out _);

        public static string[] ReadTail(string path, int maxLines, out int bytesRead)
        {
            bytesRead = 0;
            if (string.IsNullOrWhiteSpace(path) || maxLines <= 0) return Array.Empty<string>();
            try
            {
                SuffixSnapshot snapshot = ReadSuffix(path, TailReadLimitBytes);
                bytesRead = snapshot.BytesRead;
                string[] lines = SplitLines(snapshot, true);
                int count = Math.Min(maxLines, lines.Length);
                var tail = new string[count];
                Array.Copy(lines, lines.Length - count, tail, 0, count);
                return tail;
            }
            catch (IOException) { return Array.Empty<string>(); }
            catch (UnauthorizedAccessException) { return Array.Empty<string>(); }
        }

        public static string ReadLastNonEmptyLine(string path) =>
            ReadLastNonEmptyLine(path, out _);

        public static string ReadLastNonEmptyLine(string path, out int bytesRead)
        {
            bytesRead = 0;
            if (string.IsNullOrWhiteSpace(path)) return string.Empty;
            try
            {
                SuffixSnapshot snapshot = ReadSuffix(path, LastLineReadLimitBytes);
                bytesRead = snapshot.BytesRead;
                string[] lines = SplitLines(snapshot, false);
                for (int i = lines.Length - 1; i >= 0; i--)
                    if (!string.IsNullOrWhiteSpace(lines[i])) return lines[i];
                return string.Empty;
            }
            catch (IOException) { return string.Empty; }
            catch (UnauthorizedAccessException) { return string.Empty; }
        }

        static SuffixSnapshot ReadSuffix(string path, int byteLimit)
        {
            using (var stream = new FileStream(path, FileMode.Open, FileAccess.Read,
                       FileShare.ReadWrite | FileShare.Delete))
            {
                long length = stream.Length;
                bool truncated = length > byteLimit;
                long start = truncated ? length - byteLimit : 0;
                stream.Seek(start, SeekOrigin.Begin);
                int requested = (int)Math.Min(byteLimit, length - start);
                var buffer = new byte[requested];
                int bytesRead = 0;
                while (bytesRead < requested)
                {
                    int count = stream.Read(buffer, bytesRead, requested - bytesRead);
                    if (count == 0) break;
                    bytesRead += count;
                }

                int textOffset = truncated && bytesRead > 0 ? 1 : 0;
                bool startsAtLineBoundary = !truncated || (bytesRead > 0 && buffer[0] == (byte)'\n');
                string text = Encoding.UTF8.GetString(buffer, textOffset, bytesRead - textOffset);
                return new SuffixSnapshot(text, startsAtLineBoundary, bytesRead);
            }
        }

        static string[] SplitLines(SuffixSnapshot snapshot, bool includePartialFinalLine)
        {
            string text = snapshot.Text;
            if (!snapshot.StartsAtLineBoundary)
            {
                int firstNewline = text.IndexOf('\n');
                if (firstNewline < 0) return Array.Empty<string>();
                text = text.Substring(firstNewline + 1);
            }
            if (text.Length == 0) return Array.Empty<string>();

            bool endsWithNewline = text[text.Length - 1] == '\n';
            string[] raw = text.Split('\n');
            int count = raw.Length;
            if (endsWithNewline || !includePartialFinalLine) count--;
            if (count <= 0) return Array.Empty<string>();

            var lines = new string[count];
            for (int i = 0; i < count; i++)
                lines[i] = raw[i].EndsWith("\r", StringComparison.Ordinal)
                    ? raw[i].Substring(0, raw[i].Length - 1)
                    : raw[i];
            return lines;
        }

        readonly struct SuffixSnapshot
        {
            public readonly string Text;
            public readonly bool StartsAtLineBoundary;
            public readonly int BytesRead;

            public SuffixSnapshot(string text, bool startsAtLineBoundary, int bytesRead)
            {
                Text = text;
                StartsAtLineBoundary = startsAtLineBoundary;
                BytesRead = bytesRead;
            }
        }
    }
}
