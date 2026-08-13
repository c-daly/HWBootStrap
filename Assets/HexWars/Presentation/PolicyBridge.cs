using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace HexWars.Presentation
{
    public sealed class PolicySeatInfo
    {
        public int Seat { get; internal set; }
        public string Kind { get; internal set; }
        public string InferenceMode { get; internal set; }
        public string Path { get; internal set; }
        public string Algorithm { get; internal set; }
        public long Step { get; internal set; }
        public bool HasStep { get; internal set; }
        public string ContractVersion { get; internal set; }
        public string ContractHash { get; internal set; }
        public string Environment { get; internal set; }
        public string EncodingHash { get; internal set; }
        public string CapacityHash { get; internal set; }
    }

    public sealed class PolicyReadyResult
    {
        public bool Ready { get; internal set; }
        public string Error { get; internal set; }
        public int[] ModelSeats { get; internal set; } = Array.Empty<int>();
        public PolicySeatInfo[] Seats { get; internal set; } = Array.Empty<PolicySeatInfo>();
    }

    public sealed class PolicyActionResult
    {
        public int Action { get; internal set; }
        public string Error { get; internal set; }

        public int RequireAction(string stderrTail)
        {
            if (string.IsNullOrWhiteSpace(Error)) return Action;
            string detail = string.IsNullOrWhiteSpace(stderrTail) ? Error : Error + "\n" + stderrTail;
            throw new InvalidOperationException(detail);
        }
    }

    public sealed class PolicyCandidateResult
    {
        public long DecisionId { get; internal set; }
        public int CandidateId { get; internal set; }
    }

    public sealed class PolicyReloadResult
    {
        public int[] ReloadedSeats { get; internal set; } = Array.Empty<int>();
        public PolicySeatInfo[] Seats { get; internal set; } = Array.Empty<PolicySeatInfo>();
        public string Error { get; internal set; }
    }

    /// <summary>Structured JSONL bridge to the Python policy server. Editor/developer use only.</summary>
    public sealed class PolicyBridge : IDisposable
    {
        const int MaxStderrLines = 40;
        public const int DefaultStartupTimeoutMs = 30000;
        readonly Queue<string> _stderr = new Queue<string>();
        readonly object _stderrGate = new object();
        Process _proc;

        public PolicyReadyResult ReadyInfo { get; private set; }
        public PolicySeatInfo Seat0 => FindSeat(0);
        public PolicySeatInfo Seat1 => FindSeat(1);
        public string StderrTail
        {
            get { lock (_stderrGate) return string.Join("\n", _stderr); }
        }

        public async Task<bool> StartAsync(
            string pythonExe, string serverScript, string p0Spec, string p1Spec, string workingDir,
            string expectedEnvironment, string expectedContractVersion, string expectedEncodingHash,
            string expectedCapacityHash = null,
            int timeoutMs = DefaultStartupTimeoutMs, CancellationToken cancellationToken = default)
        {
            if (timeoutMs <= 0) throw new ArgumentOutOfRangeException(nameof(timeoutMs));
            Dispose();
            lock (_stderrGate) _stderr.Clear();
            ReadyInfo = null;
            var psi = new ProcessStartInfo
            {
                FileName = pythonExe,
                Arguments = BuildArguments(serverScript, p0Spec, p1Spec,
                    expectedEnvironment, expectedContractVersion, expectedEncodingHash,
                    expectedCapacityHash),
                WorkingDirectory = workingDir,
                RedirectStandardInput = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            try
            {
                var process = new Process { StartInfo = psi };
                _proc = process;
                process.ErrorDataReceived += OnError;
                if (!process.Start()) throw new InvalidOperationException("policy server did not start");
                process.BeginErrorReadLine();
                Task<string> read = process.StandardOutput.ReadLineAsync();
                Task delay = Task.Delay(timeoutMs, cancellationToken);
                if (await Task.WhenAny(read, delay) != read)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    throw new TimeoutException($"policy server did not become ready within {timeoutMs / 1000.0:0.#} seconds");
                }
                string line = await read;
                cancellationToken.ThrowIfCancellationRequested();
                if (!ReferenceEquals(process, _proc)) throw new OperationCanceledException("policy startup was stopped");
                ReadyInfo = ParseReady(line);
                if (!ReadyInfo.Ready)
                    throw new InvalidOperationException(WithStderr(ReadyInfo.Error ?? "policy server was not ready"));
                return true;
            }
            catch (OperationCanceledException)
            {
                Dispose();
                return false;
            }
            catch (Exception error)
            {
                string message = error.Message;
                Dispose();
                UnityEngine.Debug.LogError("PolicyBridge: " + WithStderr(message));
                return false;
            }
        }

        public int Act(int seat, float[] obs, bool[] mask)
        {
            EnsureRunning();
            var request = new ActionRequest { seat = seat, obs = obs, mask = mask };
            _proc.StandardInput.WriteLine(JsonUtility.ToJson(request));
            _proc.StandardInput.Flush();
            string response = _proc.StandardOutput.ReadLine();
            if (response == null) throw new InvalidOperationException(WithStderr("policy server closed unexpectedly"));
            return ParseAction(response).RequireAction(StderrTail);
        }

        public PolicyCandidateResult ActStructured(
            int seat, TacticalV3ViewDto decision)
        {
            if (decision == null) throw new ArgumentNullException(nameof(decision));
            if (seat != decision.seat)
                throw new InvalidOperationException(
                    "structured policy request seat does not match decision seat");
            EnsureRunning();
            var request = new TacticalV3PolicyRequestDto
            {
                seat = seat,
                decision = decision,
            };
            _proc.StandardInput.WriteLine(JsonUtility.ToJson(request));
            _proc.StandardInput.Flush();
            string response = _proc.StandardOutput.ReadLine();
            if (response == null)
                throw new InvalidOperationException(WithStderr(
                    "policy server closed unexpectedly"));
            try
            {
                return ParseStructuredAction(response, decision.decision_id);
            }
            catch (InvalidOperationException error)
            {
                throw new InvalidOperationException(
                    WithStderr(error.Message), error);
            }
        }

        public PolicyReloadResult Reload()
        {
            if (!IsRunning()) return new PolicyReloadResult { Error = WithStderr("policy server is not running") };
            _proc.StandardInput.WriteLine("{\"cmd\":\"reload\"}");
            _proc.StandardInput.Flush();
            string response = _proc.StandardOutput.ReadLine();
            if (response == null) return new PolicyReloadResult { Error = WithStderr("policy server closed during reload") };
            var result = ParseReload(response);
            if (ReadyInfo != null) ReadyInfo.Seats = SeatsAfterReload(ReadyInfo.Seats, result);
            return result;
        }

        public static PolicySeatInfo[] SeatsAfterReload(PolicySeatInfo[] current, PolicyReloadResult result)
        {
            if (result == null || !string.IsNullOrWhiteSpace(result.Error)) return current;
            return result.Seats ?? Array.Empty<PolicySeatInfo>();
        }

        public static PolicyReadyResult ParseReady(string json)
        {
            var dto = Parse<ReadyDto>(json, "ready");
            return new PolicyReadyResult
            {
                Ready = dto.ready,
                Error = dto.error,
                ModelSeats = dto.model_seats ?? Array.Empty<int>(),
                Seats = ConvertSeats(dto.seat_models),
            };
        }

        public static PolicyActionResult ParseAction(string json)
        {
            var dto = Parse<ActionDto>(json, "action");
            return new PolicyActionResult { Action = dto.action, Error = dto.error };
        }

        public static PolicyReloadResult ParseReload(string json)
        {
            var dto = Parse<ReloadDto>(json, "reload");
            return new PolicyReloadResult
            {
                ReloadedSeats = dto.reloaded ?? Array.Empty<int>(),
                Seats = ConvertSeats(dto.seat_models),
                Error = dto.error,
            };
        }

        static T Parse<T>(string json, string messageKind) where T : class
        {
            if (string.IsNullOrWhiteSpace(json))
                throw new InvalidOperationException("policy server returned no " + messageKind + " message");
            try
            {
                var value = JsonUtility.FromJson<T>(json);
                if (value == null) throw new FormatException("empty JSON object");
                return value;
            }
            catch (Exception error)
            {
                throw new InvalidOperationException("invalid policy " + messageKind + " message: " + error.Message);
            }
        }

        static PolicySeatInfo[] ConvertSeats(SeatDto[] seats)
        {
            if (seats == null) return Array.Empty<PolicySeatInfo>();
            var result = new PolicySeatInfo[seats.Length];
            for (int i = 0; i < seats.Length; i++)
            {
                var seat = seats[i] ?? new SeatDto();
                result[i] = new PolicySeatInfo
                {
                    Seat = seat.seat,
                    Kind = seat.kind ?? string.Empty,
                    InferenceMode = string.IsNullOrWhiteSpace(seat.inference_mode)
                        ? "deterministic"
                        : seat.inference_mode,
                    Path = seat.path ?? string.Empty,
                    Algorithm = seat.algorithm ?? string.Empty,
                    Step = seat.step,
                    HasStep = seat.step >= 0,
                    ContractVersion = seat.contract_version ?? string.Empty,
                    ContractHash = seat.contract_hash ?? string.Empty,
                    Environment = seat.environment ?? string.Empty,
                    EncodingHash = seat.encoding_hash ?? string.Empty,
                    CapacityHash = seat.capacity_hash ?? string.Empty,
                };
            }
            return result;
        }

        PolicySeatInfo FindSeat(int seat)
        {
            if (ReadyInfo?.Seats == null) return null;
            foreach (var info in ReadyInfo.Seats) if (info.Seat == seat) return info;
            return null;
        }

        void EnsureRunning()
        {
            if (!IsRunning()) throw new InvalidOperationException(WithStderr("policy server is not running"));
        }

        bool IsRunning()
        {
            if (_proc == null) return false;
            try { return !_proc.HasExited; }
            catch (InvalidOperationException) { return false; }
        }

        void OnError(object sender, DataReceivedEventArgs args)
        {
            if (string.IsNullOrWhiteSpace(args.Data)) return;
            lock (_stderrGate)
            {
                while (_stderr.Count >= MaxStderrLines) _stderr.Dequeue();
                _stderr.Enqueue(args.Data);
            }
            UnityEngine.Debug.LogWarning("[policy_server] " + args.Data);
        }

        string WithStderr(string message) => string.IsNullOrWhiteSpace(StderrTail)
            ? message
            : message + "\n" + StderrTail;

        public void Dispose()
        {
            var process = _proc;
            _proc = null;
            if (process == null) return;
            try
            {
                if (!process.HasExited)
                {
                    process.StandardInput.WriteLine("{\"cmd\":\"close\"}");
                    process.StandardInput.Flush();
                    if (!process.WaitForExit(1000))
                    {
                        process.Kill();
                        process.WaitForExit(1000);
                    }
                }
            }
            catch { try { if (!process.HasExited) process.Kill(); } catch { } }
            process.ErrorDataReceived -= OnError;
            process.Dispose();
        }

        public static string BuildArguments(string serverScript, string p0Spec, string p1Spec,
            string expectedEnvironment, string expectedContractVersion,
            string expectedEncodingHash, string expectedCapacityHash = null)
        {
            if (string.IsNullOrWhiteSpace(expectedEnvironment))
                throw new ArgumentException("expected environment is required", nameof(expectedEnvironment));
            if (string.IsNullOrWhiteSpace(expectedContractVersion))
                throw new ArgumentException("expected contract version is required", nameof(expectedContractVersion));
            if (!IsLowerSha256(expectedEncodingHash))
                throw new ArgumentException("expected encoding hash must be lowercase SHA-256", nameof(expectedEncodingHash));
            bool structured = string.Equals(
                expectedEnvironment, "tactical-v3", StringComparison.Ordinal);
            if (structured && !IsLowerSha256(expectedCapacityHash))
                throw new ArgumentException(
                    "tactical-v3 expected capacity hash must be lowercase SHA-256",
                    nameof(expectedCapacityHash));
            if (!structured && expectedCapacityHash != null)
                throw new ArgumentException(
                    "expected capacity hash is valid only for tactical-v3",
                    nameof(expectedCapacityHash));
            var args = new List<string> { QuoteArgument(serverScript) };
            if (!string.IsNullOrEmpty(p0Spec)) { args.Add("--p0"); args.Add(QuoteArgument(p0Spec)); }
            if (!string.IsNullOrEmpty(p1Spec)) { args.Add("--p1"); args.Add(QuoteArgument(p1Spec)); }
            args.Add("--expected-environment"); args.Add(QuoteArgument(expectedEnvironment));
            args.Add("--expected-contract-version"); args.Add(QuoteArgument(expectedContractVersion));
            args.Add("--expected-encoding-hash"); args.Add(QuoteArgument(expectedEncodingHash));
            if (structured)
            {
                args.Add("--expected-capacity-hash");
                args.Add(QuoteArgument(expectedCapacityHash));
            }
            return string.Join(" ", args);
        }

        public static PolicyCandidateResult ParseStructuredAction(
            string json, long expectedDecisionId)
        {
            if (string.IsNullOrWhiteSpace(json))
                throw new InvalidOperationException(
                    "policy server returned no structured action message");
            const string decisionFirst =
                @"^[ \t\r\n]*\{[ \t\r\n]*""decision_id""[ \t\r\n]*:[ \t\r\n]*(-?(?:0|[1-9][0-9]*))[ \t\r\n]*,[ \t\r\n]*""candidate_id""[ \t\r\n]*:[ \t\r\n]*(-?(?:0|[1-9][0-9]*))[ \t\r\n]*\}[ \t\r\n]*$";
            const string candidateFirst =
                @"^[ \t\r\n]*\{[ \t\r\n]*""candidate_id""[ \t\r\n]*:[ \t\r\n]*(-?(?:0|[1-9][0-9]*))[ \t\r\n]*,[ \t\r\n]*""decision_id""[ \t\r\n]*:[ \t\r\n]*(-?(?:0|[1-9][0-9]*))[ \t\r\n]*\}[ \t\r\n]*$";
            Match match = Regex.Match(json, decisionFirst, RegexOptions.CultureInvariant);
            bool reversed = false;
            if (!match.Success)
            {
                match = Regex.Match(json, candidateFirst, RegexOptions.CultureInvariant);
                reversed = true;
            }
            if (!match.Success)
                throw new InvalidOperationException(
                    "invalid structured policy action message: expected exactly decision_id and candidate_id integers");

            string decisionText = match.Groups[reversed ? 2 : 1].Value;
            string candidateText = match.Groups[reversed ? 1 : 2].Value;
            if (!long.TryParse(decisionText, NumberStyles.AllowLeadingSign,
                CultureInfo.InvariantCulture, out long decisionId))
                throw new InvalidOperationException(
                    "invalid structured policy action decision_id");
            if (!int.TryParse(candidateText, NumberStyles.AllowLeadingSign,
                CultureInfo.InvariantCulture, out int candidateId))
                throw new InvalidOperationException(
                    "invalid structured policy action candidate_id");
            if (decisionId != expectedDecisionId)
                throw new InvalidOperationException(
                    "structured policy action decision id does not match request");
            return new PolicyCandidateResult
            {
                DecisionId = decisionId,
                CandidateId = candidateId,
            };
        }

        static bool IsLowerSha256(string value)
        {
            if (value == null || value.Length != 64) return false;
            foreach (char ch in value)
                if (!((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f'))) return false;
            return true;
        }

        static string QuoteArgument(string value)
        {
            if (value == null) value = string.Empty;
            bool quote = value.Length == 0;
            for (int i = 0; i < value.Length && !quote; i++)
                quote = char.IsWhiteSpace(value[i]) || value[i] == '"';
            if (!quote) return value;
            var output = new StringBuilder().Append('"');
            int slashes = 0;
            foreach (char ch in value)
            {
                if (ch == '\\') { slashes++; continue; }
                if (ch == '"') { output.Append('\\', slashes * 2 + 1).Append('"'); slashes = 0; continue; }
                output.Append('\\', slashes).Append(ch); slashes = 0;
            }
            output.Append('\\', slashes * 2).Append('"');
            return output.ToString();
        }

        [Serializable] sealed class ActionRequest { public int seat; public float[] obs; public bool[] mask; }
        [Serializable] sealed class ReadyDto { public bool ready; public string error; public int[] model_seats; public SeatDto[] seat_models; }
        [Serializable] sealed class ActionDto { public int action; public string error; }
        [Serializable] sealed class ReloadDto { public int[] reloaded; public SeatDto[] seat_models; public string error; }
        [Serializable] sealed class SeatDto
        {
            public int seat;
            public string kind;
            public string inference_mode;
            public string path;
            public string algorithm;
            public long step = -1;
            public string contract_version;
            public string contract_hash;
            public string environment;
            public string encoding_hash;
            public string capacity_hash;
        }
    }
}
