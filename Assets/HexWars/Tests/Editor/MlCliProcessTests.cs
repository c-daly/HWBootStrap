using System;
using System.Diagnostics;
using HexWars.Presentation.EditorTools.MlLab;
using NUnit.Framework;

namespace HexWars.Presentation.Tests
{
    public sealed class MlCliProcessTests
    {
        [TestCase("plain", "plain")]
        [TestCase("two words", "\"two words\"")]
        [TestCase("", "\"\"")]
        [TestCase("a\\\"b", "\"a\\\\\\\"b\"")]
        [TestCase("C:\\path with spaces\\", "\"C:\\path with spaces\\\\\"")]
        public void QuoteArgument_UsesWindowsCommandLineRules(string value, string expected) =>
            Assert.That(MlCliProcess.QuoteArgument(value), Is.EqualTo(expected));

        [Test]
        public void BuildStartInfo_UsesDirectPythonWithRedirectedAsyncStreams()
        {
            var info = MlCliProcess.BuildStartInfo(
                @"C:\Python Env\python.exe",
                @"C:\Hex Wars\python\hexwars_ml.py",
                "status --run \"C:\\runs\\one\" --json",
                @"C:\Hex Wars\python");

            Assert.That(info.FileName, Is.EqualTo(@"C:\Python Env\python.exe"));
            Assert.That(info.Arguments, Does.StartWith("\"C:\\Hex Wars\\python\\hexwars_ml.py\" status"));
            Assert.That(info.WorkingDirectory, Is.EqualTo(@"C:\Hex Wars\python"));
            Assert.That(info.UseShellExecute, Is.False);
            Assert.That(info.CreateNoWindow, Is.True);
            Assert.That(info.RedirectStandardOutput, Is.True);
            Assert.That(info.RedirectStandardError, Is.True);
        }

        [Test]
        public void BuildDetachedStartInfo_DoesNotCreateUnityOwnedOutputPipes()
        {
            var info = MlCliProcess.BuildDetachedStartInfo(
                @"C:\Python Env\python.exe",
                @"C:\Hex Wars\python\hexwars_ml.py",
                "train --run detached --no-console-output",
                @"C:\Hex Wars\python");

            Assert.That(info.UseShellExecute, Is.False);
            Assert.That(info.CreateNoWindow, Is.True);
            Assert.That(info.RedirectStandardOutput, Is.False);
            Assert.That(info.RedirectStandardError, Is.False);
        }

        [Test]
        public void StopCommands_TargetRunAndDifferentiateCheckpointFromImmediateStop()
        {
            Assert.That(MlCliProcess.BuildStopArguments(@"C:\runs\one", false),
                Is.EqualTo("stop C:\\runs\\one --after-checkpoint --json"));
            Assert.That(MlCliProcess.BuildStopArguments(@"C:\runs\one", true),
                Is.EqualTo("stop C:\\runs\\one --now --json"));
        }

        [Test]
        public void Start_DrainsBothStreamsAndPublishesExitWithoutKillingOnDispose()
        {
            var adapter = new FakeProcessAdapter();
            var process = new MlCliProcess(new FakeProcessFactory(adapter), 3);
            int exitCode = -1;
            process.Exited += code => exitCode = code;

            process.Start(new ProcessStartInfo("python.exe"));
            adapter.EmitOutput("one");
            adapter.EmitError("bad");
            adapter.EmitOutput("three");
            adapter.EmitOutput("four");
            adapter.EmitExit(7);

            Assert.That(process.Log.Lines, Is.EqualTo(new[] { "ERROR: bad", "three", "four" }));
            Assert.That(exitCode, Is.EqualTo(7));
            process.Dispose();
            Assert.That(adapter.KillCalled, Is.False, "domain reload must not kill headless training");
            Assert.That(adapter.DisposeCalled, Is.True);
        }

        [Test]
        public void FailedStart_CleansAdapterAndAllowsRetry()
        {
            var failed = new FakeProcessAdapter { StartError = new InvalidOperationException("no python") };
            var replacement = new FakeProcessAdapter();
            var process = new MlCliProcess(new SequenceProcessFactory(failed, replacement));

            Assert.Throws<InvalidOperationException>(() => process.Start(new ProcessStartInfo("missing.exe")));
            Assert.That(failed.DisposeCalled, Is.True);
            Assert.DoesNotThrow(() => process.Start(new ProcessStartInfo("python.exe")));
            Assert.That(process.ProcessId, Is.EqualTo(42));
            process.Dispose();
        }

        [Test]
        public void StartAndReloadQuery_RestoresRunAndPublishesDurableStatus()
        {
            MlRunAttachment.Forget();
            var launched = new FakeProcessAdapter();
            var owner = new MlCliProcess(new FakeProcessFactory(launched));
            owner.Start(new ProcessStartInfo("python.exe"), @"C:\runs\active");
            owner.Dispose();

            Assert.That(MlRunAttachment.Restore().RunDirectory, Is.EqualTo(@"C:\runs\active"));
            Assert.That(MlRunAttachment.Restore().Pid, Is.EqualTo(42),
                "Start must persist the launched process' own pid (D1: reattach after a reload)");

            var query = new FakeProcessAdapter();
            var reattached = new MlCliProcess(new FakeProcessFactory(query));
            MlRunStatus received = null;
            reattached.StatusReceived += status => received = status;

            Assert.That(reattached.TryQueryAttachedRun(
                @"C:\Python Env\python.exe", @"C:\Hex Wars\hexwars_ml.py", @"C:\Hex Wars"), Is.True);
            Assert.That(query.StartInfo.Arguments, Does.Contain("status C:\\runs\\active --json"));
            query.EmitOutput("{\"ok\":true,\"result\":{\"run_dir\":\"C:/runs/active\",\"run\":{\"state\":\"running\",\"pid\":42,\"timesteps\":64}}}");
            Assert.That(received, Is.SameAs(reattached.LastStatus));
            Assert.That(received.State, Is.EqualTo(MlRunState.Running));
            Assert.That(received.Step, Is.EqualTo(64));
            reattached.Dispose();
            MlRunAttachment.Forget();
        }

        [Test]
        public void ReloadQuery_PublishesFailedStatusEnvelope()
        {
            MlRunAttachment.Forget();
            MlRunAttachment.Remember(@"C:\runs\missing");
            var query = new FakeProcessAdapter();
            var process = new MlCliProcess(new FakeProcessFactory(query));
            MlRunStatus received = null;
            process.StatusReceived += status => received = status;
            Assert.That(process.TryQueryAttachedRun("python.exe", "hexwars_ml.py", @"C:\HexWars"), Is.True);

            query.EmitOutput("{\"ok\":false,\"result\":{\"error\":\"FileNotFoundError\",\"message\":\"run missing\"}}");

            Assert.That(received, Is.SameAs(process.LastStatus));
            Assert.That(received.Ok, Is.False);
            Assert.That(received.Error, Is.EqualTo("run missing"));
            process.Dispose();
            MlRunAttachment.Forget();
        }

        [Test]
        public void RememberProcess_PersistsPidAlongsideRunDirectory()
        {
            MlRunAttachment.Forget();

            MlRunAttachment.RememberProcess(@"C:\runs\active", 4242);

            MlRunAttachment restored = MlRunAttachment.Restore();
            Assert.That(restored.RunDirectory, Is.EqualTo(@"C:\runs\active"));
            Assert.That(restored.Pid, Is.EqualTo(4242));
            Assert.That(restored.HasPid, Is.True);
            MlRunAttachment.Forget();
        }

        [Test]
        public void Remember_SwitchingRunDirectoryClearsAnyPreviouslyRecordedPid()
        {
            MlRunAttachment.Forget();
            MlRunAttachment.RememberProcess(@"C:\runs\active", 4242);

            MlRunAttachment.Remember(@"C:\runs\different");

            MlRunAttachment restored = MlRunAttachment.Restore();
            Assert.That(restored.RunDirectory, Is.EqualTo(@"C:\runs\different"));
            Assert.That(restored.HasPid, Is.False,
                "a pid recorded for a different run must not survive selecting a different run");
            MlRunAttachment.Forget();
        }

        [Test]
        public void Remember_SameRunDirectoryPreservesThePersistedPid()
        {
            MlRunAttachment.Forget();
            MlRunAttachment.RememberProcess(@"C:\runs\active", 4242);

            // Mirrors the once-a-second status poll and manual re-selection, both of which
            // re-remember the same run directory without ever knowing its launch-time pid.
            MlRunAttachment.Remember(@"C:\runs\active");

            Assert.That(MlRunAttachment.Restore().Pid, Is.EqualTo(4242),
                "re-remembering the same run must not clobber the launch-time pid");
            MlRunAttachment.Forget();
        }

        sealed class FakeProcessFactory : IMlProcessFactory
        {
            readonly IMlProcessAdapter _adapter;
            public FakeProcessFactory(IMlProcessAdapter adapter) => _adapter = adapter;
            public IMlProcessAdapter Create() => _adapter;
        }

        sealed class SequenceProcessFactory : IMlProcessFactory
        {
            readonly IMlProcessAdapter[] _adapters;
            int _index;
            public SequenceProcessFactory(params IMlProcessAdapter[] adapters) => _adapters = adapters;
            public IMlProcessAdapter Create() => _adapters[_index++];
        }

        sealed class FakeProcessAdapter : IMlProcessAdapter
        {
            public event Action<string> OutputLine;
            public event Action<string> ErrorLine;
            public event Action<int> ProcessExited;
            public int Id => 42;
            public bool IsRunning { get; private set; }
            public bool KillCalled { get; private set; }
            public bool DisposeCalled { get; private set; }
            public Exception StartError;
            public ProcessStartInfo StartInfo;
            public void Start(ProcessStartInfo startInfo)
            {
                if (StartError != null) throw StartError;
                StartInfo = startInfo;
                IsRunning = true;
            }
            public void Kill() { KillCalled = true; IsRunning = false; }
            public void Dispose() => DisposeCalled = true;
            public void EmitOutput(string line) => OutputLine?.Invoke(line);
            public void EmitError(string line) => ErrorLine?.Invoke(line);
            public void EmitExit(int code) { IsRunning = false; ProcessExited?.Invoke(code); }
        }
    }
}
