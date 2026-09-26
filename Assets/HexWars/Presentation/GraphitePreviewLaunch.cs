using System;
using System.Collections;
using System.IO;
using HexWars.Engine;
using UnityEngine;

namespace HexWars.Presentation
{
    /// <summary>Opt-in preview launch and repeatable captures of the built player.</summary>
    public sealed class GraphitePreviewLaunch : MonoBehaviour
    {
        IEnumerator Start()
        {
            yield return null;
            GraphiteWorkshop.Open(GetComponent<GameBootstrap>(), true);
            var args=Environment.GetCommandLineArgs();int at=Array.IndexOf(args,"-graphite-capture");
            if(at<0 || at+1>=args.Length) yield break;
            string folder=args[at+1];Directory.CreateDirectory(folder);
            yield return new WaitForSecondsRealtime(3);
            yield return new WaitForEndOfFrame();
            ScreenCapture.CaptureScreenshot(Path.Combine(folder,"unity-graphite-collection.png"));
            yield return new WaitForSecondsRealtime(1);
            var gallery=GetComponent<GraphiteWorkshop>();if(gallery!=null) Destroy(gallery);
            GetComponent<GameBootstrap>().StartLocalGame(new GameSetup(GameMode.Annihilation,9,7,200,7),false);
            yield return new WaitForSecondsRealtime(2);
            var designer=FindAnyObjectByType<DesignPanel>();designer.SelectArt("halo-01");
            TipBubble.Dismiss();
            yield return new WaitForSecondsRealtime(.25f);
            yield return new WaitForEndOfFrame();
            ScreenCapture.CaptureScreenshot(Path.Combine(folder,"unity-graphite-battlefield.png"));
            Debug.Log("[GraphitePreview] Captured collection and local battlefield; manual Halo selected.");
            yield return new WaitForSecondsRealtime(1);
            Application.Quit();
        }
    }
}
