using HexWars.Engine;
using UnityEngine;
using UnityEngine.UI;

namespace HexWars.Presentation
{
    /// <summary>Inspectable collection of the actual meshes used in a match.</summary>
    public sealed class GraphiteWorkshop : MonoBehaviour
    {
        public static bool IsOpen { get; private set; }
        bool _offerNewMatch;
        GameObject _canvas; RawImage _hero; Text _name, _description;
        readonly Button[] _choices=new Button[8];
        static readonly string[] Descriptions={"Utility rover with a cab and articulated tool arm.","Armored crawler with a broad, sloped hull.","Assault carrier with a twin-barrel turret.","Shield carrier on a planted tracked chassis.","Recon buggy with exposed wheels and a light cab.","Four-legged walker with jointed climbing feet.","Tracked long-range gun with a recoil sleeve.","Sensor rover with a radar dish and mast."};
        public static void Open(GameBootstrap game, bool offerNewMatch = false)
        {
            if (game.GetComponent<GraphiteWorkshop>() != null) return;
            var workshop = game.gameObject.AddComponent<GraphiteWorkshop>();
            workshop._offerNewMatch = offerNewMatch;
        }
        void Awake() => IsOpen = true;
        void Start()
        {
            _canvas=UiKit.Canvas("GraphiteWorkshop",UiKit.OrderMenu+20,transform);
            var backdrop=UiKit.Panel(_canvas.transform,"Backdrop",UiKit.Bg);UiKit.Stretch(backdrop.rectTransform);
            var panel=UiKit.Panel(_canvas.transform,"Collection",UiKit.Surface);
            var r=panel.rectTransform;r.anchorMin=r.anchorMax=new Vector2(.5f,.5f);r.pivot=new Vector2(.5f,.5f);r.sizeDelta=new Vector2(1120,740);r.anchoredPosition=Vector2.zero;
            HexBrandMark.Add(panel.transform,-486,-32,48);
            UiKit.Label(panel.transform,"HEXWARS",-329,-37,238,32,28,TextAnchor.MiddleLeft);
            UiKit.Label(panel.transform,"UNIT COLLECTION",305,-42,400,26,14,TextAnchor.MiddleRight,UiKit.TextDim);
            UiKit.Button(panel.transform,"Close",485,-95,90,32,()=>Destroy(this),UiKit.ButtonStyle.Secondary,14);
            _hero=Portrait(panel.transform,0,-300,-172,365);
            _name=UiKit.Label(panel.transform,"",-290,-528,400,45,36,TextAnchor.MiddleCenter);
            _description=UiKit.Label(panel.transform,"",-290,-580,410,56,16,TextAnchor.UpperCenter,UiKit.TextDim);
            _description.horizontalOverflow=HorizontalWrapMode.Wrap;
            UiKit.Label(panel.transform,"Choose your unit’s appearance.",267,-130,486,35,27,TextAnchor.MiddleLeft);
            UiKit.Label(panel.transform,"Choose a machine in the unit designer.\nYour stats still determine what the unit can do.",267,-181,486,48,17,TextAnchor.UpperLeft,UiKit.TextDim);
            for(int i=0;i<8;i++)
            {
                int idx=i;float x=87+(i%4)*120,y=-261-(i/4)*139;
                var button=UiKit.Button(panel.transform,"",x,y,109,126,()=>Select(idx),UiKit.ButtonStyle.Secondary,13);_choices[i]=button;
                Portrait(button.transform,i,0,-4,95);
                UiKit.Label(button.transform,UnitArt.Names[i],0,-98,105,24,14,TextAnchor.MiddleCenter);
            }
            UiKit.Label(panel.transform,"MINT / PLAYER 1     ·     AMBER / PLAYER 2",267,-564,486,26,12,TextAnchor.MiddleLeft,UiKit.Accent);
            UiKit.Label(panel.transform,"Player 2 uses a broken rim; Player 1 uses a solid rim.\nTeam identity also has a distinct shape.",267,-599,486,44,15,TextAnchor.UpperLeft,UiKit.TextDim);
            var currentGame = GetComponent<GameBootstrap>();
            bool startMatch = _offerNewMatch || currentGame.State == null || currentGame.DemoMode;
            UiKit.Button(panel.transform,startMatch ? "Try on the battlefield" : "Return to match",267,-660,486,46,()=>
            {
                var game=GetComponent<GameBootstrap>();
                if (startMatch) game.StartLocalGame(new GameSetup(GameMode.Annihilation,9,7,200,7),false);
                Destroy(this);
            },UiKit.ButtonStyle.Cta,21);
            Select(0);
        }
        internal static RawImage Portrait(Transform parent,int index,float x,float y,float size)
        {
            var go=new GameObject("Portrait "+UnitArt.Names[index]);go.transform.SetParent(parent,false);
            var image=go.AddComponent<RawImage>();image.raycastTarget=false;
            go.AddComponent<UnitPortrait>().SetArt(index, size > 128);
            UiKit.SetRect(image.rectTransform,x,y,size,size);return image;
        }
        void Select(int i)
        {
            _hero.GetComponent<UnitPortrait>().SetArt(i, true);_name.text=UnitArt.Names[i];_description.text=Descriptions[i];
            for(int j=0;j<8;j++) UiKit.SetToggled(_choices[j],i==j);
        }
        void OnDestroy(){IsOpen=false;if(_canvas!=null)Destroy(_canvas);}
    }
}
