using System.IO;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEngine;

namespace HexWars.Presentation.EditorTools
{
    /// <summary>
    /// Builds the HexWars scene to a browser (WebGL) player. Compression is disabled so the output is
    /// plain static files any HTTP server (or tunnel) can serve without Content-Encoding configuration.
    /// Run from the menu (HexWars ▸ Build WebGL) or headless:
    ///   Unity.exe -quit -batchmode -projectPath . -buildTarget WebGL \
    ///     -executeMethod HexWars.Presentation.EditorTools.WebGLBuild.Build -logFile build.log
    /// Output: Build/WebGL/ (open index.html via a server, not file://).
    /// </summary>
    public static class WebGLBuild
    {
        const string OutputDir = "Build/WebGL";
        static readonly string[] Scenes = { "Assets/Scenes/HexWars.unity" };

        [MenuItem("HexWars/Build WebGL")]
        public static void Build()
        {
            // Gzip + decompression fallback: ~15 MB instead of ~54 MB, and still serves from any static
            // host (incl. our Kestrel) without Content-Encoding header config. Full exceptions for now.
            PlayerSettings.WebGL.compressionFormat = WebGLCompressionFormat.Gzip;
            PlayerSettings.WebGL.decompressionFallback = true;
            PlayerSettings.WebGL.exceptionSupport = WebGLExceptionSupport.FullWithStacktrace;

            var fullOut = Path.GetFullPath(OutputDir);
            Directory.CreateDirectory(fullOut);

            var opts = new BuildPlayerOptions
            {
                scenes = Scenes,
                locationPathName = OutputDir,
                target = BuildTarget.WebGL,
                targetGroup = BuildTargetGroup.WebGL,
                options = BuildOptions.None,
            };

            BuildReport report = BuildPipeline.BuildPlayer(opts);
            BuildSummary summary = report.summary;
            Debug.Log($"[WebGLBuild] result={summary.result} sizeBytes={summary.totalSize} " +
                      $"time={summary.totalTime} errors={summary.totalErrors} out={fullOut}");

            if (summary.result != BuildResult.Succeeded)
            {
                Debug.LogError($"[WebGLBuild] FAILED: {summary.result} ({summary.totalErrors} errors)");
                if (Application.isBatchMode) EditorApplication.Exit(1);
                return;
            }
            Debug.Log("[WebGLBuild] SUCCESS");
            if (Application.isBatchMode) EditorApplication.Exit(0);
        }
    }
}
