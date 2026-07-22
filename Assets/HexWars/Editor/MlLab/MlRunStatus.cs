using System;
using System.Collections.Generic;
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
        public string Error { get; private set; }

        public static MlRunStatus Parse(string json)
        {
            if (string.IsNullOrWhiteSpace(json)) throw new ArgumentException("Status JSON is empty.", nameof(json));
            var envelope = JsonUtility.FromJson<Envelope>(json);
            if (envelope == null) throw new FormatException("Status JSON could not be parsed.");
            var result = envelope.result ?? new Result();
            var run = result.run ?? envelope.run ?? new Run();
            var config = run.config ?? new Config();
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

        static bool HasTrackerFailure(Tracker[] trackers)
        {
            if (trackers == null) return false;
            foreach (var tracker in trackers)
                if (tracker != null && (!tracker.ok || string.Equals(
                    tracker.status, "degraded", StringComparison.OrdinalIgnoreCase))) return true;
            return false;
        }

        [Serializable] sealed class Envelope { public bool ok; public Result result; public Run run; }
        [Serializable] sealed class Result { public string run_dir; public Run run; public string message; }
        [Serializable] sealed class Config { public string run_name; public long total_timesteps; }
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
        public readonly int Pid;
        public bool Exists => !string.IsNullOrWhiteSpace(RunDirectory);

        MlRunAttachment(string runDirectory, int pid)
        {
            RunDirectory = runDirectory ?? string.Empty;
            Pid = pid;
        }

        public static void Remember(string runDirectory, int pid)
        {
            if (string.IsNullOrWhiteSpace(runDirectory)) throw new ArgumentException(
                "Active run directory is required.", nameof(runDirectory));
            SessionState.SetString(RunKey, runDirectory);
            SessionState.SetInt(PidKey, Math.Max(0, pid));
        }

        public static MlRunAttachment Restore() => new MlRunAttachment(
            SessionState.GetString(RunKey, string.Empty), SessionState.GetInt(PidKey, 0));

        public static void Forget()
        {
            SessionState.EraseString(RunKey);
            SessionState.EraseInt(PidKey);
        }
    }
}
