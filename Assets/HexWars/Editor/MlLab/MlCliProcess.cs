using System;
using System.Diagnostics;
using System.Text;
using System.Threading;

namespace HexWars.Presentation.EditorTools.MlLab
{
    public interface IMlProcessAdapter : IDisposable
    {
        event Action<string> OutputLine;
        event Action<string> ErrorLine;
        event Action<int> ProcessExited;
        int Id { get; }
        bool IsRunning { get; }
        void Start(ProcessStartInfo startInfo);
        void Kill();
    }

    public interface IMlProcessFactory
    {
        IMlProcessAdapter Create();
    }

    sealed class SystemMlProcessFactory : IMlProcessFactory
    {
        public IMlProcessAdapter Create() => new SystemMlProcessAdapter();
    }

    sealed class SystemMlProcessAdapter : IMlProcessAdapter
    {
        Process _process;
        public event Action<string> OutputLine;
        public event Action<string> ErrorLine;
        public event Action<int> ProcessExited;
        public int Id => _process != null && IsRunning ? _process.Id : 0;
        public bool IsRunning
        {
            get
            {
                if (_process == null) return false;
                try { return !_process.HasExited; }
                catch (InvalidOperationException) { return false; }
            }
        }

        public void Start(ProcessStartInfo startInfo)
        {
            if (_process != null) throw new InvalidOperationException("Process adapter is single-use.");
            _process = new Process { StartInfo = startInfo, EnableRaisingEvents = true };
            _process.OutputDataReceived += OnOutput;
            _process.ErrorDataReceived += OnError;
            _process.Exited += OnExited;
            if (!_process.Start()) throw new InvalidOperationException("Python process did not start.");
            if (startInfo.RedirectStandardOutput) _process.BeginOutputReadLine();
            if (startInfo.RedirectStandardError) _process.BeginErrorReadLine();
        }

        public void Kill() { if (IsRunning) _process.Kill(); }
        void OnOutput(object sender, DataReceivedEventArgs args) { if (args.Data != null) OutputLine?.Invoke(args.Data); }
        void OnError(object sender, DataReceivedEventArgs args) { if (args.Data != null) ErrorLine?.Invoke(args.Data); }
        void OnExited(object sender, EventArgs args)
        {
            int code = -1;
            try
            {
                var exited = sender as Process;
                if (exited != null)
                {
                    // Required after redirected async reads: this waits for final OutputDataReceived /
                    // ErrorDataReceived callbacks before the owner observes ProcessExited.
                    exited.WaitForExit();
                    code = exited.ExitCode;
                }
            }
            catch (InvalidOperationException) { }
            ProcessExited?.Invoke(code);
        }

        public void Dispose()
        {
            if (_process == null) return;
            _process.OutputDataReceived -= OnOutput;
            _process.ErrorDataReceived -= OnError;
            _process.Exited -= OnExited;
            _process.Dispose();
            _process = null;
        }
    }

    public sealed class MlCliProcess : IDisposable
    {
        readonly IMlProcessFactory _factory;
        readonly SynchronizationContext _context;
        readonly MlLogBuffer _log;
        IMlProcessAdapter _process;

        public event Action Changed;
        public event Action<int> Exited;
        public event Action<MlRunStatus> StatusReceived;
        public int ProcessId => _process != null && _process.IsRunning ? _process.Id : 0;
        public bool IsRunning => _process != null && _process.IsRunning;
        public MlLogBuffer Log => _log;

        public MlCliProcess(int maxLogLines = 400) : this(null, maxLogLines) { }

        public MlCliProcess(IMlProcessFactory factory, int maxLogLines = 400)
        {
            _factory = factory ?? new SystemMlProcessFactory();
            _context = SynchronizationContext.Current;
            _log = new MlLogBuffer(maxLogLines);
        }

        public MlRunStatus LastStatus { get; private set; }

        public void Start(ProcessStartInfo startInfo, string activeRunDirectory = null)
        {
            if (startInfo == null) throw new ArgumentNullException(nameof(startInfo));
            if (IsRunning) throw new InvalidOperationException("An ML command is already running.");
            DisposeProcess();
            _process = _factory.Create();
            _process.OutputLine += OnOutput;
            _process.ErrorLine += OnError;
            _process.ProcessExited += OnExited;
            try { _process.Start(startInfo); }
            catch
            {
                DisposeProcess();
                throw;
            }
            if (!string.IsNullOrWhiteSpace(activeRunDirectory))
                MlRunAttachment.RememberProcess(activeRunDirectory, _process.Id);
            NotifyChanged();
        }

        public bool TryQueryAttachedRun(
            string pythonExe, string scriptPath, string workingDirectory)
        {
            var attachment = MlRunAttachment.Restore();
            if (!attachment.Exists) return false;
            Start(BuildStartInfo(
                pythonExe, scriptPath, BuildStatusArguments(attachment.RunDirectory), workingDirectory));
            return true;
        }

        public void Kill() { if (IsRunning) _process.Kill(); }

        public static ProcessStartInfo BuildStartInfo(
            string pythonExe, string scriptPath, string commandArguments, string workingDirectory)
            => BuildStartInfo(pythonExe, scriptPath, commandArguments, workingDirectory, true);

        public static ProcessStartInfo BuildDetachedStartInfo(
            string pythonExe, string scriptPath, string commandArguments, string workingDirectory)
            => BuildStartInfo(pythonExe, scriptPath, commandArguments, workingDirectory, false);

        static ProcessStartInfo BuildStartInfo(
            string pythonExe, string scriptPath, string commandArguments, string workingDirectory,
            bool redirectOutput)
        {
            return new ProcessStartInfo
            {
                FileName = pythonExe,
                Arguments = QuoteArgument(scriptPath) + " " + (commandArguments ?? string.Empty),
                WorkingDirectory = workingDirectory,
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardOutput = redirectOutput,
                RedirectStandardError = redirectOutput,
            };
        }

        public static string BuildStatusArguments(string runPath) =>
            "status " + QuoteArgument(runPath) + " --json";

        public static string BuildStopArguments(string runPath, bool immediate) =>
            "stop " + QuoteArgument(runPath) + (immediate ? " --now --json" : " --after-checkpoint --json");

        public static string QuoteArgument(string value)
        {
            if (value == null) value = string.Empty;
            bool quote = value.Length == 0;
            for (int i = 0; i < value.Length && !quote; i++)
                quote = char.IsWhiteSpace(value[i]) || value[i] == '"';
            if (!quote) return value;

            var output = new StringBuilder(value.Length + 2).Append('"');
            int slashes = 0;
            foreach (char ch in value)
            {
                if (ch == '\\') { slashes++; continue; }
                if (ch == '"')
                {
                    output.Append('\\', slashes * 2 + 1).Append('"');
                    slashes = 0;
                    continue;
                }
                output.Append('\\', slashes).Append(ch);
                slashes = 0;
            }
            output.Append('\\', slashes * 2).Append('"');
            return output.ToString();
        }

        void OnOutput(string line) => AddLine(line);
        void OnError(string line) => AddLine(line == null ? null : "ERROR: " + line);
        void AddLine(string line)
        {
            if (line == null) return;
            _log.Add(line);
            if (line.Length > 1 && line[0] == '{')
            {
                try
                {
                    var status = MlRunStatus.Parse(line);
                    if (status != null)
                    {
                        LastStatus = status;
                        Post(() => StatusReceived?.Invoke(status));
                    }
                }
                catch (Exception) { /* Non-status JSON and partial log lines remain ordinary output. */ }
            }
            NotifyChanged();
        }

        void OnExited(int code)
        {
            Post(() =>
            {
                Changed?.Invoke();
                Exited?.Invoke(code);
            });
        }

        // Closing the window or reloading assemblies must not kill a headless training run.
        public void Dispose() => DisposeProcess();
        void NotifyChanged() => Post(() => Changed?.Invoke());

        void Post(Action action)
        {
            if (_context == null || SynchronizationContext.Current == _context) action();
            else _context.Post(_ => action(), null);
        }

        void DisposeProcess()
        {
            if (_process == null) return;
            _process.OutputLine -= OnOutput;
            _process.ErrorLine -= OnError;
            _process.ProcessExited -= OnExited;
            _process.Dispose();
            _process = null;
        }
    }
}
