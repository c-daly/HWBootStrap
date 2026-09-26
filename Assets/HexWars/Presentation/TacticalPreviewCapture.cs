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
            if(attack==null)
            {
                // Correctly mirrored backlines can start beyond range. Advance with a legal move.
                foreach(var move in LegalMoves.For(game.State).OfType<MoveUnit>())
                {
                    var next=GameEngine.Apply(game.State,move);
                    if(!next.Success)continue;
                    attack=LegalMoves.For(next.NewState).OfType<AttackUnit>().FirstOrDefault();
                    if(attack==null)continue;
                    game.TryApply(move);game.Presenter.FastForward();break;
                }
            }
            if(attack==null)throw new InvalidOperationException("Capture fixture has no reachable attack.");
            input.SelectById(attack.AttackerId);input.SetMode(UnitInputController.Intent.Attack);
            yield return Capture(folder,"tactical-native-range");
            input.PreviewAttack(attack.TargetId);yield return Capture(folder,"tactical-native-attack");
            input.ClearPreview();game.TryApply(new EndTurn(game.State.ActivePlayer));game.Presenter.FastForward();
            input.SelectById(game.State.Player(PlayerId.Player1).UnitsOnBoard[0].Id);
            yield return Capture(folder,"tactical-native-player2");
            GetComponent<TacticalHud>().SetWorkshop(true);
            FindAnyObjectByType<DesignPanel>().SelectArt("halo-01");TipBubble.Dismiss();
            yield return Capture(folder,"tactical-native-designer");
            GetComponent<TacticalHud>().SetWorkshop(false);
            game.StartLocalGame(new GameSetup(GameMode.Annihilation,9,7,40,7,manualPlacement:true),false);
            TipBubble.Dismiss();yield return null;
            input.SelectById(game.State.Players[0].UnitsOnBoard[0].Id);
            input.PreviewMove(input.PlacementCells.OrderByDescending(c=>game.State.Board.TileAt(c).Elevation).First());
            yield return Capture(folder,"tactical-native-placement");
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
