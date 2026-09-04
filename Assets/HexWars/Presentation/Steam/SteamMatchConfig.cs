#nullable enable
using System;
using UnityEngine;

namespace HexWars.Presentation
{
    /// <summary>
    /// The Unity half of the match-service configuration: where the base URL comes from at runtime.
    /// Order is command line, then environment, then the shipped <c>Resources/HexWarsSteamConfig</c>
    /// asset. The asset ships with the <c>OWNER-INPUT</c> placeholder, so an unconfigured build
    /// resolves to <see cref="SteamMatchSettings.NotConfigured"/> and the title screen can say Steam
    /// play is unavailable instead of failing at the first request. Validation and parsing live in
    /// <c>SteamMatchApiContracts.cs</c> so they are covered by dotnet tests.
    /// </summary>
    public static partial class SteamMatchConfig
    {
        static SteamMatchSettings? _cached;

        /// <summary>Resolves once per process; call <see cref="Invalidate"/> to force a re-read.</summary>
        public static SteamMatchSettings Resolve()
        {
            if (_cached == null) _cached = ResolveUncached();
            return _cached;
        }

        public static void Invalidate()
        {
            _cached = null;
        }

        static SteamMatchSettings ResolveUncached()
        {
            var fromCommandLine = FromBaseUrl(CommandLineValue(CommandLineFlag));
            if (fromCommandLine.IsConfigured) return fromCommandLine;

            var fromEnvironment = FromBaseUrl(EnvironmentValue(EnvironmentVariable));
            if (fromEnvironment.IsConfigured) return fromEnvironment;

            var asset = Resources.Load<TextAsset>(ResourceName);
            if (asset != null)
            {
                var fromAsset = ParseJson(asset.text);
                if (fromAsset.IsConfigured) return fromAsset;
            }

            return SteamMatchSettings.NotConfigured;
        }

        /// <summary>Reads <c>-hexwars-match-url &lt;url&gt;</c>. Unavailable on WebGL, hence the guard.</summary>
        static string? CommandLineValue(string flag)
        {
            string[] args;
            try { args = Environment.GetCommandLineArgs(); }
            catch (Exception) { return null; }
            if (args == null) return null;

            for (var i = 0; i < args.Length - 1; i++)
            {
                if (string.Equals(args[i], flag, StringComparison.Ordinal)) return args[i + 1];
            }
            return null;
        }

        static string? EnvironmentValue(string name)
        {
            try { return Environment.GetEnvironmentVariable(name); }
            catch (Exception) { return null; }
        }
    }
}
