#if UNITY_STANDALONE_WIN || UNITY_EDITOR
using System;
using System.Collections;
using System.IO;
using System.Linq;
using HexWars.Engine;
using UnityEngine;

namespace HexWars.Presentation
{
    /// <summary>Opt-in native evidence capture. Uses ordinary matches and engine commands.</summary>
    public sealed class TacticalPreviewCapture : MonoBehaviour
    {
        IEnumerator Start()
        {
            var args=Environment.GetCommandLineArgs();int at=Array.IndexOf(args,"-tactical-capture");
            if(at<0||at+1>=args.Length)yield break;
            string folder=args[at+1];Directory.CreateDirectory(folder);
            Application.runInBackground=true;
            yield return null;
            var game=GetComponent<GameBootstrap>();var input=FindAnyObjectByType<UnitInputController>();
            // Keep this opt-in capture deterministic even if the preview window receives input.
            input.enabled=false;
            var events=FindAnyObjectByType<UnityEngine.EventSystems.EventSystem>();if(events!=null)events.enabled=false;
            FindAnyObjectByType<CameraRig>().enabled=false;
            game.StartLocalGame(new GameSetup(GameMode.Annihilation,7,5,200,7),false);
            TipBubble.Dismiss();
            yield return new WaitForSecondsRealtime(2);
            FindAnyObjectByType<CameraRig>().Frame();
            var units=game.State.Player(PlayerId.Player0).UnitsOnBoard;
            input.SelectById(units[units.Count-1].Id);
            yield return Capture(folder,"tactical-native-board");
            var route=input.Routes.OrderBy(p=>p.Value.HorizontalCost).FirstOrDefault();
            if(input.Routes.Count>0){input.PreviewMove(route.Key);yield return Capture(folder,"tactical-native-move");input.ClearPreview();}
            // A smaller ordinary match brings both armies into the same view for the forecast check.
            game.StartLocalGame(new GameSetup(GameMode.Annihilation,5,5,200,7),false);
            yield return null;
            var attack=LegalMoves.For(game.State).OfType<AttackUnit>().FirstOrDefault();
            if(attack!=null){input.SelectById(attack.AttackerId);input.PreviewAttack(attack.TargetId);yield return Capture(folder,"tactical-native-attack");}
            else Debug.LogWarning("[TacticalCapture] No legal attack in capture seed.");
            GetComponent<TacticalHud>().SetWorkshop(true);
            FindAnyObjectByType<DesignPanel>().SelectArt("halo-01");TipBubble.Dismiss();
            yield return Capture(folder,"tactical-native-designer");
            GetComponent<TacticalHud>().SetWorkshop(false);
            GraphiteWorkshop.Open(game,false);yield return Capture(folder,"tactical-native-machines");
            Debug.Log("[TacticalCapture] Native evidence complete.");Application.Quit();
        }
        static IEnumerator Capture(string folder,string name)
        {
            yield return new WaitForSecondsRealtime(1.5f);yield return new WaitForEndOfFrame();
            var snapshot=ScreenCapture.CaptureScreenshotAsTexture();
            File.WriteAllBytes(Path.Combine(folder,name+".png"),snapshot.EncodeToPNG());Destroy(snapshot);
            yield return new WaitForSecondsRealtime(.3f);
        }
    }
}
#endif
