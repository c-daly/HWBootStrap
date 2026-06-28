using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEngine;
using UnityEngine.Rendering;

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
            PlayerSettings.WebGL.template = "PROJECT:HexWarsFull"; // full-window canvas → sharp at any size / in fullscreen

            // Board/unit materials are created at runtime via Shader.Find — WebGL strips shaders that
            // nothing references at build time, so they'd render magenta. Force-include them.
            EnsureShadersIncluded(
                "Universal Render Pipeline/Lit",
                "Universal Render Pipeline/Unlit",
                "HexWars/Matcap",
                "HexWars/IconUnlit",
                "Unlit/Color",
                "Unlit/Texture",
                "Skybox/Panoramic");

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

        /// <summary>Add shaders to Graphics ▸ Always Included Shaders so runtime Shader.Find works in the build.</summary>
        static void EnsureShadersIncluded(params string[] names)
        {
            var so = new SerializedObject(GraphicsSettings.GetGraphicsSettings());
            var arr = so.FindProperty("m_AlwaysIncludedShaders");
            var have = new HashSet<string>();
            for (int i = 0; i < arr.arraySize; i++)
                if (arr.GetArrayElementAtIndex(i).objectReferenceValue is Shader s) have.Add(s.name);

            foreach (var n in names)
            {
                if (have.Contains(n)) continue;
                var sh = Shader.Find(n);
                if (sh == null) { Debug.LogWarning("[WebGLBuild] shader not found to include: " + n); continue; }
                arr.InsertArrayElementAtIndex(arr.arraySize);
                arr.GetArrayElementAtIndex(arr.arraySize - 1).objectReferenceValue = sh;
                Debug.Log("[WebGLBuild] +AlwaysIncluded: " + n);
            }
            so.ApplyModifiedProperties();
        }
    }
}
