using System;
using System.Diagnostics;
using System.Globalization;
using System.Text;
using UnityEngine;

namespace HexWars.Presentation
{
    /// <summary>
    /// Runs trained SB3 models as opponents by talking to the Windows-venv <c>policy_server.py</c> over
    /// stdin/stdout (localhost subprocess — no WSL networking). Unity computes the observation + legal mask
    /// with the shared engine codec and sends them; the server returns the model's action. Editor/dev tool.
    /// </summary>
    public sealed class PolicyBridge : IDisposable
    {
        Process _proc;

        /// <summary>Starts the server with the given per-seat model specs (e.g. "ppo:sp6base.zip"); pass null
        /// for a seat Unity drives itself (greedy/random). Returns true once the server reports ready.</summary>
        public bool Start(string pythonExe, string serverScript, string p0Spec, string p1Spec, string workingDir)
        {
            var args = new StringBuilder();
            args.Append('"').Append(serverScript).Append('"');
            if (!string.IsNullOrEmpty(p0Spec)) args.Append(" --p0 \"").Append(p0Spec).Append('"');
            if (!string.IsNullOrEmpty(p1Spec)) args.Append(" --p1 \"").Append(p1Spec).Append('"');

            var psi = new ProcessStartInfo
            {
                FileName = pythonExe,
                Arguments = args.ToString(),
                WorkingDirectory = workingDir,
                RedirectStandardInput = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            };

            try { _proc = Process.Start(psi); }
            catch (Exception e) { UnityEngine.Debug.LogError($"PolicyBridge: failed to launch python: {e.Message}"); return false; }

            _proc.ErrorDataReceived += (_, e) => { if (!string.IsNullOrEmpty(e.Data)) UnityEngine.Debug.LogWarning("[policy_server] " + e.Data); };
            _proc.BeginErrorReadLine();

            // server prints one ready line after the model(s) load (torch import + load can take a few seconds)
            string ready = _proc.StandardOutput.ReadLine();
            if (ready == null || !ready.Contains("\"ready\""))
            {
                UnityEngine.Debug.LogError("PolicyBridge: server did not report ready (got: " + (ready ?? "<eof>") + ")");
                return false;
            }
            return true;
        }

        /// <summary>Send one observation + mask for <paramref name="seat"/> and block for the model's action.
        /// Localhost round-trip is sub-millisecond, fine for turn-based stepping.</summary>
        public int Act(int seat, float[] obs, bool[] mask)
        {
            var sb = new StringBuilder(obs.Length * 8 + mask.Length * 6 + 32);
            sb.Append("{\"seat\":").Append(seat).Append(",\"obs\":[");
            for (int i = 0; i < obs.Length; i++) { if (i > 0) sb.Append(','); sb.Append(obs[i].ToString("R", CultureInfo.InvariantCulture)); }
            sb.Append("],\"mask\":[");
            for (int i = 0; i < mask.Length; i++) { if (i > 0) sb.Append(','); sb.Append(mask[i] ? "true" : "false"); }
            sb.Append("]}");

            _proc.StandardInput.WriteLine(sb.ToString());
            _proc.StandardInput.Flush();

            string resp = _proc.StandardOutput.ReadLine();
            if (resp == null) throw new InvalidOperationException("policy_server closed unexpectedly");
            int c = resp.IndexOf(':'); int e = resp.IndexOf('}', c < 0 ? 0 : c);
            return int.Parse(resp.Substring(c + 1, e - c - 1).Trim(), CultureInfo.InvariantCulture);
        }

        public void Dispose()
        {
            try
            {
                if (_proc != null && !_proc.HasExited)
                {
                    _proc.StandardInput.WriteLine("{\"cmd\":\"close\"}");
                    _proc.StandardInput.Flush();
                    if (!_proc.WaitForExit(1000)) _proc.Kill();
                }
            }
            catch { /* best effort */ }
            _proc?.Dispose();
            _proc = null;
        }
    }
}
