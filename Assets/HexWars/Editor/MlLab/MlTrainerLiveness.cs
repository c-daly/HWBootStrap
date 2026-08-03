using System;
using System.Diagnostics;
using System.Globalization;
using System.IO;

namespace HexWars.Presentation.EditorTools.MlLab
{
    /// <summary>Outcome of <see cref="MlTrainerLivenessPolicy.Decide"/>: how confidently the Lab
    /// believes the trainer process it launched is still alive. <see cref="TrackedAlive"/> means the
    /// PID persisted at launch time (see <see cref="MlRunAttachment.RememberProcess"/>) still resolves
    /// to a process with a matching name -- the strongest signal available. <see cref="AliveByFiles"/>
    /// is the fallback used whenever that PID handle is unavailable (a genuine editor-process restart
    /// clears Unity's SessionState, or the PID was never recorded) or was reused by an unrelated
    /// process: progress.csv having moved recently is treated as proof of life instead. <see
    /// cref="Dead"/> is everything else -- no valid process handle and no recent file activity.</summary>
    public enum MlTrainerLivenessState { TrackedAlive, AliveByFiles, Dead }

    /// <summary>Pure decision behind D1 ("Lab stops lying about trainers"): before this fix, "is the
    /// trainer alive" was read off <c>MlLabWindowState.Phase</c>, an in-memory field that (a) resets to
    /// <c>Idle</c> on every OnEnable/domain reload with nothing to reconcile it against reality, and
    /// (b) gets knocked to <c>Failed</c> by a single transient status-query subprocess error --
    /// permanently, because <c>MlLabWindow.AttemptWatch</c>'s Start &amp; Watch retry loop treats
    /// "training ended" (<see cref="MlWatchStartPolicy"/>'s <c>GiveUp</c>) as a one-shot,
    /// non-retryable verdict once <c>_pendingWatchRunDirectory</c> is cleared. Grounding the answer in
    /// the PID persisted at launch (existence + name, guarding against PID reuse) with a file-mtime
    /// fallback removes that single point of failure: a status-query hiccup, or even losing the
    /// process handle outright across a restart, no longer looks like "training ended" as long as the
    /// run's own files are still moving.</summary>
    public static class MlTrainerLivenessPolicy
    {
        public static MlTrainerLivenessState Decide(
            bool hasPersistedPid,
            bool processExists,
            bool processIdentityMatches,
            bool progressFresh)
        {
            if (hasPersistedPid && processExists && processIdentityMatches)
                return MlTrainerLivenessState.TrackedAlive;
            return progressFresh ? MlTrainerLivenessState.AliveByFiles : MlTrainerLivenessState.Dead;
        }

        // The watch gate (MlWatchStartPolicy.Decide's trainingAlive) only needs a bool: both
        // TrackedAlive and AliveByFiles mean "still going, keep waiting for the checkpoint."
        public static bool IsAlive(MlTrainerLivenessState state) => state != MlTrainerLivenessState.Dead;
    }

    /// <summary>How long <c>progress.csv</c> may go untouched before the Lab calls a run stalled --
    /// long enough to absorb a slow checkpoint/eval pause without crying wolf, short enough to catch a
    /// genuinely hung or dead trainer well before a human notices via TensorBoard going quiet (D2,
    /// "Lab stops lying about trainers").</summary>
    public static class MlTrainerProgressFreshness
    {
        public const double StalenessThresholdMinutes = 5.0;

        public static bool IsFresh(double minutesSinceLastWrite) =>
            minutesSinceLastWrite < StalenessThresholdMinutes;
    }

    /// <summary>Pure formatter for D2's honest run-status line -- no modal, no interruption of
    /// playback, just a status string surfaced next to the Lab's own run status and, for a live watch,
    /// the Arena identity rows (see <see cref="HexWars.Presentation.ModelDuelDriver.MarkTrainerLivenessStatus"/>).
    /// <paramref name="confirmedExited"/> distinguishes a positively-identified process death (worth
    /// calling "exited", with the last known step) from merely stale files with no process handle to
    /// check (call that "stalled" instead -- honest about not knowing whether it crashed, was killed,
    /// or is just hung).</summary>
    public static class MlTrainerStatusFormatter
    {
        public static string Describe(
            bool confirmedExited,
            double minutesSinceProgress,
            long step,
            long targetStep)
        {
            if (confirmedExited)
                return "trainer exited (step " +
                    step.ToString("N0", CultureInfo.InvariantCulture) + " of " +
                    targetStep.ToString("N0", CultureInfo.InvariantCulture) + ")";
            if (!MlTrainerProgressFreshness.IsFresh(minutesSinceProgress))
                return "trainer stalled — no progress for " +
                    minutesSinceProgress.ToString("0", CultureInfo.InvariantCulture) + " min";
            return string.Empty;
        }
    }

    /// <summary>The only OS-facing piece of D1's reattachment: resolves a persisted PID to a live
    /// process' name (existence + name check -- the documented "at minimum" bar for guarding against
    /// PID reuse, short of reading the full command line). Deliberately not unit-tested against real
    /// processes; <see cref="MlTrainerLivenessPolicy.Decide"/> is the tested seam and takes this
    /// class' output as plain booleans.</summary>
    static class MlTrainerProcessLookup
    {
        public static bool TryGetRunningProcessName(int pid, out string processName)
        {
            processName = string.Empty;
            if (pid <= 0) return false;
            try
            {
                using (Process process = Process.GetProcessById(pid))
                {
                    if (process.HasExited) return false;
                    processName = process.ProcessName;
                    return true;
                }
            }
            catch (ArgumentException) { return false; } // no process with that id
            catch (InvalidOperationException) { return false; } // process exited mid-check
        }

        public static bool MatchesExpectedExecutable(string processName, string expectedExecutablePath)
        {
            string expected = Path.GetFileNameWithoutExtension(expectedExecutablePath ?? string.Empty);
            return !string.IsNullOrEmpty(expected) &&
                string.Equals(processName, expected, StringComparison.OrdinalIgnoreCase);
        }
    }
}
