using System.IO;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace HexWars.Presentation.EditorTools
{
    public static class GraphitePreviewBuild
    {
        [MenuItem("HexWars/Build Graphite Windows Preview")]
        public static void Build()
        {
            WebGLBuild.EnsureShadersIncluded("Universal Render Pipeline/Lit", "Universal Render Pipeline/Unlit",
                "HexWars/Matcap", "HexWars/IconUnlit", "Unlit/Color", "Unlit/Texture", "Skybox/Panoramic");
            PlayerSettings.defaultScreenWidth=1600;PlayerSettings.defaultScreenHeight=900;
            PlayerSettings.fullScreenMode=FullScreenMode.Windowed;PlayerSettings.resizableWindow=true;
            var scene=EditorSceneManager.OpenScene("Assets/Scenes/HexWars.unity");
            var camera=Camera.main;
            if(camera!=null) { camera.clearFlags=CameraClearFlags.SolidColor;camera.backgroundColor=UiKit.Bg;EditorUtility.SetDirty(camera); }
            EditorSceneManager.MarkSceneDirty(scene);EditorSceneManager.SaveScene(scene);
            AssetDatabase.SaveAssets();
            Directory.CreateDirectory("Build/GraphitePreview");
            var result=BuildPipeline.BuildPlayer(new BuildPlayerOptions {
                scenes=new[]{"Assets/Scenes/HexWars.unity"},locationPathName="Build/GraphitePreview/HexWars.exe",
                target=BuildTarget.StandaloneWindows64,options=BuildOptions.None,extraScriptingDefines=new[]{"DISABLESTEAMWORKS"} });
            Debug.Log($"[GraphitePreviewBuild] {result.summary.result}; errors={result.summary.totalErrors}; size={result.summary.totalSize}");
            if(Application.isBatchMode) EditorApplication.Exit(result.summary.result==BuildResult.Succeeded ? 0 : 1);
        }
    }
}
