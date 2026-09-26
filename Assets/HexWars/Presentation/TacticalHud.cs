using System;
using System.Collections.Generic;
using HexWars.Engine;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.InputSystem;

namespace HexWars.Presentation
{
    /// <summary>Decision-focused match UI. Reads authoritative state and never applies preview mutations.</summary>
    public sealed class TacticalHud : MonoBehaviour
    {
        public static bool ModalOpen { get; private set; }
        public bool WorkshopOpen { get; private set; }
        GameBootstrap _game; UnitInputController _input;
        DesignPanel _designer; BarracksPanel _barracks;
        GameObject _canvas, _dialog; RectTransform _panel, _squadContent;
        Text _turn, _round, _mission, _context, _name, _role, _hp, _stats, _availability;
        Text _moveText, _attackText, _decision, _target, _result, _math, _note, _left, _cycleLabel;
        Button _move, _attack, _confirm, _end, _workshop, _previous, _next, _undo;
        Image _health, _targetHealth, _targetLoss;
        UnitPortrait _portrait;
        readonly List<Button> _squad = new List<Button>();
        readonly List<HexCoord> _destinations = new List<HexCoord>();
        readonly List<int> _targets = new List<int>();
        GameState _lastState, _dialogState; int _lastSelected=-2, _lastArt=-1;
        Vector2 _size; bool _detailsOpen; bool _dirty=true, _lastWaiting; int _statusBits=-1;
        bool _territoryLayout;
        const float W=348;
        static readonly Color Panel = new Color32(25,38,48,255);
        static readonly Color Line = new Color32(51,70,81,255);
        static readonly Color Muted = new Color32(161,181,189,255);
        static readonly Color Copper = new Color32(236,174,139,255);
        static readonly Color Ink = new Color32(229,238,235,255);

        void Start()
        {
            _game=GetComponent<GameBootstrap>();_input=FindAnyObjectByType<UnitInputController>();
            _designer=FindAnyObjectByType<DesignPanel>();_barracks=FindAnyObjectByType<BarracksPanel>();
            if(_input!=null)_input.PresentationChanged+=Dirty;
            _game.StateChanged+=Dirty; _game.CommandRejected+=Dirty;
            SetWorkshop(false); Build(); Refresh();
        }
        void OnDestroy()
        {
            if(_game!=null){_game.StateChanged-=Dirty;_game.CommandRejected-=Dirty;}
            if(_input!=null){_input.PresentationChanged-=Dirty;_input.SetTerritoryActionHost(null);}
            ModalOpen=false;if(_canvas!=null)Destroy(_canvas);if(_dialog!=null)Destroy(_dialog);
            if(Camera.main!=null)Camera.main.rect=new Rect(0,0,1,1);
        }
        void Dirty()=>_dirty=true;
        void Update()
        {
            if(_game==null||_canvas==null)return;
            var size=((RectTransform)_canvas.transform).rect.size;
            if(Vector2.Distance(size,_size)>2){Build();_dirty=true;}
            int status = (_game.Reconnecting?1:0) | (_input!=null&&_input.CanCommand?2:0)
                | (_input!=null&&_input.AwaitingServer?4:0) | (_game.DemoMode?8:0);
            if(status!=_statusBits){_statusBits=status;_dirty=true;}
            if(ModalOpen && !UiKit.EscapeHandledThisFrame && DeviceInput.FocusProbe() && Keyboard.current!=null && Keyboard.current.escapeKey.wasPressedThisFrame)
            {UiKit.MarkInputEscapeHandled();CloseDialog();}
            if(_dirty)Refresh();
        }

        static void Place(RectTransform r,float x,float y,float w,float h)
        {r.anchorMin=r.anchorMax=new Vector2(0,1);r.pivot=new Vector2(0,1);r.anchoredPosition=new Vector2(x,-y);r.sizeDelta=new Vector2(w,h);}
        static Text Label(Transform p,string text,float x,float y,float w,float h,int size,Color? color=null)
        {
            var t=UiKit.Label(p,text,0,0,w,h,size,TextAnchor.MiddleLeft,color??Ink);
            Place(t.rectTransform,x,y,w,h);t.horizontalOverflow=HorizontalWrapMode.Wrap;t.verticalOverflow=VerticalWrapMode.Truncate;
            t.supportRichText=false;return t;
        }
        static Image Surface(Transform p,string name,float x,float y,float w,float h,Color c)
        {var i=UiKit.Panel(p,name,c);Place(i.rectTransform,x,y,w,h);return i;}
        static Button Button(Transform p,string name,string text,float x,float y,float w,float h,Action click,int size=16)
        {var b=UiKit.Button(p,text,0,0,w,h,click,UiKit.ButtonStyle.Secondary,size);b.name=name;Place((RectTransform)b.transform,x,y,w,h);return b;}
        static void Tint(Button b,Color c)
        {var v=b.colors;v.normalColor=c;v.selectedColor=c;v.highlightedColor=Color.Lerp(c,Color.white,.12f);v.pressedColor=c*.82f;v.disabledColor=new Color(.18f,.23f,.27f,.6f);b.colors=v;}
        static void Glyph(Transform p,TacticalGlyph.Shape kind,float x,float y,float size,Color color)
        {var go=new GameObject(kind.ToString());go.transform.SetParent(p,false);var g=go.AddComponent<TacticalGlyph>();g.Symbol=kind;g.color=color;g.raycastTarget=false;Place(g.rectTransform,x,y,size,size);}

        void Build()
        {
            // Preserve the existing action button when a resize replaces the HUD canvas.
            if(_input!=null)_input.SetTerritoryActionHost(null);
            if(_canvas!=null){_canvas.SetActive(false);Destroy(_canvas);}
            _squad.Clear();_lastState=null;_lastSelected=-2;_lastArt=-1;
            _canvas=UiKit.Canvas("TacticalHUD",UiKit.OrderPanels-10,transform);
            _size=((RectTransform)_canvas.transform).rect.size;
            float width=_size.x,height=_size.y;bool narrow=width<1100;
            // A partial camera viewport does not clear the rest of the framebuffer. Paint every
            // surrounding pixel each frame so changing text cannot leave old glyphs behind.
            Rect view=BoardViewport(_size);
            void Backdrop(float x,float y,float w,float h)
            {var bg=Surface(_canvas.transform,"Viewport surround",x,y,w,h,UiKit.Bg);bg.sprite=null;bg.raycastTarget=false;}
            Backdrop(0,0,width,height*(1-view.yMax));
            Backdrop(0,height*(1-view.yMin),width,height*view.yMin);
            Backdrop(0,height*(1-view.yMax),width*view.xMin,height*view.height);
            Backdrop(width*view.xMax,height*(1-view.yMax),width*(1-view.xMax),height*view.height);
            var bar=Surface(_canvas.transform,"Match header",0,0,width,58,UiKit.Bg);
            var mark=HexBrandMark.Add(bar.transform,0,0,33);Place(mark.rectTransform,22,12,33,33);
            Label(bar.transform,"HEXWARS",69,13,160,31,23);
            _turn=Label(bar.transform,"",narrow?230:width*.40f,9,narrow?200:280,23,17,GraphitePieces.Mint);
            _round=Label(bar.transform,"",narrow?230:width*.40f,32,narrow?200:340,17,12,Muted);
            float fieldW=narrow?width:width-W-54;
            _mission=Label(_canvas.transform,"",26,77,fieldW-50,24,15,Muted);
            Button(bar.transform,"Game menu","Menu",width-108,13,84,32,()=>FindAnyObjectByType<EscapeMenu>()?.Toggle(),14);
            _undo=Button(bar.transform,"Undo move","Undo move",width-385,13,119,32,()=>_input?.UndoLastMove(),14);
            float panelY=narrow?height*.51f:76,panelH=narrow?height*.49f-16:height-99;
            float panelW=narrow?width-32:W;
            _territoryLayout=_game.State!=null&&_game.State.Config.TerritoryMode&&!_game.State.PlacingStartingUnits;
            float territoryHeight=_territoryLayout?64:0;
            _panel=Surface(_canvas.transform,"Selected unit panel",narrow?16:width-W-24,panelY,panelW,panelH,Panel).rectTransform;
            var viewport=Surface(_panel,"Decision scroll",0,0,panelW,panelH-161-territoryHeight,Panel);
            viewport.gameObject.AddComponent<RectMask2D>();var scroll=viewport.gameObject.AddComponent<ScrollRect>();
            scroll.horizontal=false;scroll.vertical=true;scroll.viewport=viewport.rectTransform;scroll.movementType=ScrollRect.MovementType.Clamped;
            var content=new GameObject("Decision content",typeof(RectTransform));content.transform.SetParent(viewport.transform,false);
            var body=(RectTransform)content.transform;Place(body,0,0,panelW,_detailsOpen?638:568);scroll.content=body;scroll.scrollSensitivity=30;
            _role=Label(body,"",20,17,panelW-40,20,12,GraphitePieces.Mint);
            _name=Label(body,"Select a unit",20,43,panelW-155,49,27);
            var portrait=GraphiteWorkshop.Portrait(body,0,0,0,132);Place(portrait.rectTransform,panelW-147,20,132,132);_portrait=portrait.GetComponent<UnitPortrait>();
            _hp=Label(body,"",20,98,panelW-168,27,15,GraphitePieces.Mint);
            Surface(body,"Health track",20,132,panelW-40,4,Line);
            _health=Surface(body,"Health fill",20,132,panelW-40,4,GraphitePieces.Mint);
            var details=Button(body,"Unit details",_detailsOpen?"Hide details":"+ Unit details",20,154,panelW-40,28,()=>{_detailsOpen=!_detailsOpen;Build();Dirty();},13);
            Tint(details,Panel);
            _stats=Label(body,"",20,190,panelW-40,66,14,Muted);_stats.gameObject.SetActive(_detailsOpen);
            float extra=_detailsOpen?70:0;
            _availability=Label(body,"",20,186+extra,panelW-40,20,12,Muted);
            float bw=(panelW-49)/2;
            _move=Button(body,"Move mode","",20,214+extra,bw,55,()=>_input?.SetMode(UnitInputController.Intent.Move));
            _attack=Button(body,"Attack mode","",29+bw,214+extra,bw,55,()=>_input?.SetMode(UnitInputController.Intent.Attack));
            Glyph(_move.transform,TacticalGlyph.Shape.Move,10,17,21,GraphitePieces.Mint);
            Glyph(_attack.transform,TacticalGlyph.Shape.Attack,10,17,21,Copper);
            _moveText=Label(_move.transform,"Move [M]",40,4,bw-43,47,14);
            _attackText=Label(_attack.transform,"Attack [F]",40,4,bw-43,47,14);
            _decision=Label(body,"",20,290+extra,panelW-40,22,12,Muted);
            _target=Label(body,"",20,317+extra,panelW-40,29,20);
            _result=Label(body,"",20,356+extra,panelW-40,46,26,Copper);
            _targetHealth=Surface(body,"Target health remaining",20,411+extra,0,5,Copper);
            _targetLoss=Surface(body,"Target health lost",20,411+extra,0,5,new Color(.48f,.30f,.23f));
            _math=Label(body,"",20,437+extra,panelW-40,76,14,Muted);
            _previous=Button(body,"Previous preview","<",20,530+extra,38,30,()=>Cycle(-1),14);
            _cycleLabel=Label(body,"Choose on the board",67,530+extra,panelW-134,30,12,Muted);
            _next=Button(body,"Next preview",">",panelW-58,530+extra,38,30,()=>Cycle(1),14);
            _confirm=Button(_panel,"Confirm action","Choose a target",20,panelH-146-territoryHeight,panelW-40,44,()=>_input?.ConfirmPreview(),17);
            _note=Label(_panel,"",20,panelH-96-territoryHeight,panelW-40,27,12,Muted);_note.alignment=TextAnchor.MiddleCenter;
            if(_territoryLayout)
            {
                var territory=Surface(_panel,"Territory action",20,panelH-124,panelW-40,56,Panel);
                territory.raycastTarget=false;
                if(_input!=null)_input.SetTerritoryActionHost(territory.rectTransform);
            }
            Surface(_panel,"End divider",20,panelH-59,panelW-40,1,Line);
            _left=Label(_panel,"",20,panelH-50,panelW-160,36,13,Muted);
            _end=Button(_panel,"End turn","End turn",panelW-132,panelH-48,112,35,RequestEndTurn,15);
            float squadY=narrow?height*.51f-106:height-102;
            _context=Label(_canvas.transform,"",26,squadY-35,fieldW-40,25,13,Muted);
            _workshop=Button(_canvas.transform,"Army workshop","Design army",width-256,13,136,32,()=>SetWorkshop(!WorkshopOpen),14);
            var squadViewport=Surface(_canvas.transform,"Squad viewport",22,squadY+12,fieldW-43,70,UiKit.Bg);
            squadViewport.gameObject.AddComponent<RectMask2D>();var squadScroll=squadViewport.gameObject.AddComponent<ScrollRect>();
            squadScroll.horizontal=true;squadScroll.vertical=false;squadScroll.viewport=squadViewport.rectTransform;squadScroll.scrollSensitivity=32;
            squadScroll.movementType=ScrollRect.MovementType.Clamped;
            var row=new GameObject("Squad",typeof(RectTransform));row.transform.SetParent(squadViewport.transform,false);
            _squadContent=(RectTransform)row.transform;Place(_squadContent,0,0,fieldW-43,68);squadScroll.content=_squadContent;
            ApplyCamera();
        }

        static Rect BoardViewport(Vector2 size) => size.x<1100
            ? new Rect(0,.55f,1,.34f) : new Rect(.015f,.165f,(size.x-W-58)/size.x,.70f);

        void ApplyCamera()
        {
            if(Camera.main==null||_game==null)return;
            bool active=_game.State!=null&&!_game.DemoMode;
            Rect rect=!active?new Rect(0,0,1,1):BoardViewport(_size);
            if(Camera.main.rect!=rect){Camera.main.rect=rect;Camera.main.GetComponent<CameraRig>()?.Frame();}
        }

        public void SetWorkshop(bool open)
        {
            WorkshopOpen=open;
            if(_designer!=null)_designer.SetExpanded(open);
            if(_barracks!=null)_barracks.SetExpanded(open);
            _input?.ClearPreview();_dirty=true;
        }

        internal void DismissWorkshop()
        {
            if(_designer!=null)WebGlInputBridge.CancelFocusedEdit(_designer.transform);
            SetWorkshop(false);
        }

        PlayerId Seat
        {
            get
            {
                if(_game.Networked) return _game.Seat??_game.State.ActivePlayer;
                var ai=_game.GetComponent<AiOpponent>();
                if(ai!=null) return ai.AiSeat==PlayerId.Player0?PlayerId.Player1:PlayerId.Player0;
                return _game.State.ActivePlayer;
            }
        }
        void Refresh()
        {
            _dirty=false;bool active=_game.State!=null&&!_game.DemoMode;
            if(active&&_territoryLayout!=(_game.State.Config.TerritoryMode&&!_game.State.PlacingStartingUnits))Build();
            _canvas.SetActive(active);ApplyCamera();if(!active){CloseDialog();return;}
            var s=_game.State;bool changed=!ReferenceEquals(_lastState,s);
            bool battleStarting=_lastState!=null&&_lastState.PlacingStartingUnits&&!s.PlacingStartingUnits;
            if(_dialog!=null&&!ReferenceEquals(_dialogState,s))CloseDialog();
            if(s.PlacingStartingUnits&&WorkshopOpen)SetWorkshop(false);
            var viewer=Seat;bool waiting=_game.WaitingHumanSeat()!=null;
            _turn.text=s.PlacingStartingUnits?$"Player {(int)s.ActivePlayer+1}: starting positions":s.IsGameOver?"Match complete":_game.Reconnecting?"Reconnecting...":waiting?"Opponent's turn":_game.Networked||_game.GetComponent<AiOpponent>()!=null?"Your turn":$"Player {(int)s.ActivePlayer+1}'s turn";
            _turn.color=GraphitePieces.TeamColor(s.ActivePlayer);
            string pace=s.Config.TurnPolicy.RemainingActions(s)?.ToString();
            _round.text=s.PlacingStartingUnits?$"SETUP   /   {s.Player(viewer).Points} points":$"ROUND {s.Round:00}   /   {s.Player(viewer).Points} points"+(pace!=null?$"   /   {pace} actions left":"");
            _mission.text=s.PlacingStartingUnits?"Arrange your army inside the highlighted starting area":WorkshopOpen?"":s.Config.TerritoryMode?"Territory · control the battlefield":"Annihilation · eliminate the opposing army";
            _panel.gameObject.SetActive(!WorkshopOpen);
            _workshop.GetComponentInChildren<Text>().text=WorkshopOpen?"Back to battle":"Design army";
            _workshop.interactable=!s.PlacingStartingUnits;
            _end.GetComponentInChildren<Text>().text=s.PlacingStartingUnits?"Ready":s.IsGameOver?"Main menu":"End turn";
            _end.interactable=s.IsGameOver||(_input!=null&&_input.CanCommand);
            if(changed||_lastSelected!=(_input?.SelectedId??-1)||_lastWaiting!=waiting)RebuildSquad();
            _lastState=s;_lastSelected=_input?.SelectedId??-1;_lastWaiting=waiting;
            if(_input==null)return;
            _undo.interactable=_input.CanUndoMove;
            if(changed && !waiting && (_input.SelectedId<0 || battleStarting || (s.PlacingStartingUnits && _input.SelectedUnit?.Owner!=viewer)))
                foreach(var first in s.Player(viewer).UnitsOnBoard) if(first.IsAlive){_input.SelectById(first.Id);break;}
            var chosen=_input.SelectedUnit;
            _confirm.interactable=false;_destinations.Clear();_targets.Clear();
            _confirm.gameObject.SetActive(true);
            _targetHealth.rectTransform.sizeDelta=new Vector2(0,5);_targetLoss.rectTransform.sizeDelta=new Vector2(0,5);
            if(!chosen.HasValue)
            {
                _name.text="Select a unit";_role.text="YOUR NEXT DECISION";_hp.text="";_stats.text="Choose a machine on the board or in your squad.";
                _portrait.gameObject.SetActive(false);_health.gameObject.SetActive(false);
                _availability.text=waiting?"Waiting for the opponent":"Inspect a unit to see its actions";
                _move.interactable=_attack.interactable=false;_decision.text="BATTLEFIELD";_target.text="";_result.text="";_math.text="Choose a unit from your squad.";
                _context.text=WorkshopOpen?"Choose or create a design, then select a deployment hex.":"Select a unit to begin.";
                _confirm.GetComponentInChildren<Text>().text="Select a unit";_note.text="";_previous.interactable=_next.interactable=false;return;
            }
            var u=chosen.Value;int artIndex=UnitArt.Index(UnitArt.Resolve(u.ArtId,u.Stats));
            _portrait.gameObject.SetActive(true);_health.gameObject.SetActive(true);
            if(_lastArt!=artIndex||_portrait.Owner!=u.Owner){_lastArt=artIndex;_portrait.SetArt(artIndex,true,u.Owner);}
            _role.color=_hp.color=_health.color=GraphitePieces.TeamColor(u.Owner);
            _role.text=$"PLAYER {(int)u.Owner+1}  /  "+(u.Owner==viewer?"YOUR UNIT":"OPPONENT");
            _name.text=u.DisplayName;_hp.text=$"HULL   {u.CurrentHp} / {u.Stats.Health}";
            _health.rectTransform.sizeDelta=new Vector2((_panel.rect.width-40)*u.CurrentHp/Mathf.Max(1f,u.Stats.Health),4);
            _stats.text=$"{Roles.Dominant(u.Stats)} · Damage {u.Stats.Damage} · Defense {u.Stats.Defense}\nRange {u.Stats.Range} / arc {u.Stats.RangeArc} · Vision {u.Stats.Vision} / arc {u.Stats.VisionArc}\n"+(s.Config.BiomesEnabled?"Terrain bonuses enabled":"Terrain bonuses off");
            if(s.PlacingStartingUnits)
            {
                if(_input.Mode!=UnitInputController.Intent.Move)_input.SetMode(UnitInputController.Intent.Move);
                bool arrange=_input.CanCommand&&u.Owner==viewer;
                _destinations.AddRange(_input.PlacementCells);
                _move.interactable=_attack.interactable=false;_moveText.text="Place unit";_attackText.text="Battle not started";
                _availability.text=waiting?"Opponent is arranging their army":"Free placement";
                _decision.text="STARTING POSITION";_result.text="";_math.text="Select a unit, then choose an empty highlighted hex.";
                _target.text=_input.Destination.HasValue?"Cell "+Cell(_input.Destination.Value):"Choose a starting hex";
                _confirm.GetComponentInChildren<Text>().text="Place here";_confirm.interactable=arrange&&_input.Destination.HasValue;
                _confirm.gameObject.SetActive(false);
                _note.text="Click a hex to place. No movement or points spent.";
                _context.text="Raised hexes are available too. Select Ready when your army is arranged.";
                _left.text="Arrange your army";_previous.interactable=_next.interactable=arrange&&_destinations.Count>0;
                _cycleLabel.text=$"{_destinations.Count} starting hexes";return;
            }
            bool spent=TacticalForecast.HasAttacked(s,u.Id),mine=u.Owner==viewer,command=_input.CanCommand&&mine;
            var budget=s.MovementSpent.TryGetValue(u.Id,out var used)?used:(H:0,V:0);
            int horizontal=spent?0:Math.Max(0,u.Stats.Movement-budget.H),vertical=spent?0:Math.Max(0,u.Stats.VerticalMovement-budget.V);
            foreach(var pair in _input.Routes)_destinations.Add(pair.Key);
            _destinations.Sort((a,b)=>a.Q==b.Q?a.R.CompareTo(b.R):a.Q.CompareTo(b.Q));
            if(command)foreach(var candidate in AttackPreviewTargets.Resolve(s,u,null,viewer))
                if(TacticalForecast.TryCreate(s,viewer,u.Id,candidate.UnitId,out _))_targets.Add(candidate.UnitId);
            _move.interactable=command&&_destinations.Count>0;_attack.interactable=command&&!spent;
            _moveText.text=$"Move [M]\n{horizontal} move / {vertical} climb";_attackText.text=spent?"Attack [F]\nUsed":$"Attack [F]\n{_targets.Count} in range";
            _availability.text=!mine?"Opponent inspected":_input.AwaitingServer?"Waiting for server confirmation":waiting?"Waiting for your turn":spent?"Actions used":"";
            Tint(_move,_input.Mode==UnitInputController.Intent.Move?new Color(.20f,.38f,.34f):new Color(.15f,.21f,.25f));
            Tint(_attack,_input.Mode==UnitInputController.Intent.Attack?new Color(.39f,.28f,.22f):new Color(.15f,.21f,.25f));
            bool moving=_input.Mode==UnitInputController.Intent.Move;
            Tint(_confirm,moving?new Color(.28f,.53f,.46f):new Color(.65f,.40f,.27f));
            _note.text=_input.AwaitingServer?"Waiting for the server...":moving?
                (s.Config.FogOfWar?"Fog is on: moves cannot be undone.":"Click a hex to move. Ctrl+Z undoes the last move."):
                "Click the target again to fire. Movement then ends.";
            _decision.text="PREVIEW";
            _target.text=moving?"Choose a destination":"Choose a target";_result.text="";
            _math.text=spent?"This unit has already attacked.":"";
            _confirm.GetComponentInChildren<Text>().text=moving?"Choose a destination":spent?"Attack used":"Choose a target";
            _context.text=$"Cell {Cell(u.Cell)} · Height {u.Elevation}"+(s.Config.BiomesEnabled?$" · {s.Board.TileAt(u.Cell).Terrain}":"");
            _confirm.gameObject.SetActive(!moving || _input.LockedRoute!=null);
            if(moving&&_input.HoverRoute!=null&&_input.HoverDestination.HasValue)
            {
                var route=_input.HoverRoute;string dest=Cell(_input.HoverDestination.Value);
                _target.text=$"Cell {dest}";_result.text=$"{route.HorizontalCost} move  /  {route.VerticalCost} climb";_result.color=GraphitePieces.Mint;
                _math.text=$"Remaining: {route.HorizontalRemaining} move · {route.VerticalRemaining} climb"+(_detailsOpen?$"\nDestination height {s.Board.TileAt(_input.HoverDestination.Value).Elevation}\nAttack stays ready until you fire.":"");
                _confirm.GetComponentInChildren<Text>().text="Move to "+dest;_confirm.interactable=command;
                _context.text=$"Route to {dest}";
                var preview=GameEngine.Apply(s,new MoveUnit(s.ActivePlayer,u.Id,_input.HoverDestination.Value));
                if(preview.Success&&preview.NewState.ActivePlayer!=s.ActivePlayer)_note.text="This move ends your turn and cannot be undone.";
            }
            else if(!moving&&TacticalForecast.TryCreate(s,viewer,u.Id,_input.TargetId,out var forecast))
            {
                _target.text=forecast.Target.DisplayName;_result.text=$"{forecast.Damage} damage";_result.color=Copper;
                string hp=forecast.RemainingHealth==0?"Target destroyed":$"{forecast.RemainingHealth} / {forecast.Target.Stats.Health} hull after hit";
                _math.text=hp+(_detailsOpen?$"\nWeapon {forecast.Weapon} + height {forecast.HeightBonus}\nArmor {forecast.Armor} + terrain {forecast.Cover}"+(forecast.Weapon>0&&s.Config.DamageFloor>0?$" · minimum {s.Config.DamageFloor}":""):"");
                float bar=(_panel.rect.width-40)/Mathf.Max(1,forecast.Target.Stats.Health);
                _targetHealth.rectTransform.sizeDelta=new Vector2(bar*forecast.RemainingHealth,5);
                Place(_targetLoss.rectTransform,20+bar*forecast.RemainingHealth,411+(_detailsOpen?70:0),bar*(forecast.Target.CurrentHp-forecast.RemainingHealth),5);
                _confirm.GetComponentInChildren<Text>().text=$"Fire · {forecast.Damage} damage";_confirm.interactable=command;
                _context.text="";
            }
            int choices=moving?_destinations.Count:_targets.Count;
            _previous.interactable=_next.interactable=command&&choices>0;
            _cycleLabel.text=$"{choices} {(moving?"destinations":"targets")}";
        }

        void RebuildSquad()
        {
            // Destroy is deferred until frame end; old-seat buttons must stop rendering and
            // accepting input before their replacements are added in this same refresh.
            foreach(var button in _squad)if(button!=null){button.gameObject.SetActive(false);Destroy(button.gameObject);}_squad.Clear();
            var s=_game.State;int canAct=0;float x=0;
            foreach(var u in s.Player(Seat).UnitsOnBoard)
            {
                if(!u.IsAlive)continue;int id=u.Id;
                bool spent=TacticalForecast.HasAttacked(s,id),active=s.ActivePlayer==u.Owner&&!s.IsGameOver;
                bool canMove=active&&!spent&&MovementService.Routes(s,u).Count>0;
                bool canAttack=active&&!spent&&AttackPreviewTargets.Resolve(s,u,null,Seat).Count>0;
                if(canMove||canAttack)canAct++;
                var b=Button(_squadContent,"Select unit "+id,"",x,0,186,66,()=>_input?.SelectReadyUnitOfKind(id));_squad.Add(b);
                Tint(b,_input!=null&&_input.SelectedId==id?new Color(.22f,.35f,.35f):Panel);
                var p=GraphiteWorkshop.Portrait(b.transform,UnitArt.Index(UnitArt.Resolve(u.ArtId,u.Stats)),0,0,58,u.Owner);Place(p.rectTransform,2,2,58,58);
                Label(b.transform,u.DisplayName,65,7,110,22,14);
                Label(b.transform,$"{u.CurrentHp}/{u.Stats.Health}",65,32,55,22,12,Muted);
                Glyph(b.transform,TacticalGlyph.Shape.Move,126,37,14,canMove?GraphitePieces.Mint:Line);
                Glyph(b.transform,TacticalGlyph.Shape.Attack,150,37,14,canAttack?Copper:Line);
                x+=196;
            }
            _squadContent.sizeDelta=new Vector2(Mathf.Max(x,_squadContent.parent.GetComponent<RectTransform>().rect.width),68);
            _left.text=$"{canAct} units can act";
        }
        static string Cell(HexCoord c)=>$"{c.Q+1},{c.R+1}";
        void Cycle(int delta)
        {
            if(_input.Mode==UnitInputController.Intent.Move&&_destinations.Count>0)
            {int i=_input.Destination.HasValue?_destinations.IndexOf(_input.Destination.Value):-1;i=(i+delta+_destinations.Count)%_destinations.Count;_input.PreviewMove(_destinations[i]);}
            else if(_targets.Count>0){int i=(_targets.IndexOf(_input.TargetId)+delta+_targets.Count)%_targets.Count;_input.PreviewAttack(_targets[i]);}
        }
        void RequestEndTurn()
        {
            if(_game.State.PlacingStartingUnits){_input?.FinishStartingPlacement();return;}
            if(_game.State.IsGameOver){_game.ReturnToMenu();return;}
            if(_input==null||!_input.CanCommand)return;
            _dialogState=_game.State;ModalOpen=true;
            _dialog=UiKit.Canvas("Confirm end turn",UiKit.OrderEscape+5,transform);
            var shade=UiKit.Panel(_dialog.transform,"Modal backdrop",new Color(0.02f,.04f,.06f,.88f));UiKit.Stretch(shade.rectTransform);
            var panel=UiKit.Panel(_dialog.transform,"End turn confirmation",Panel);var r=panel.rectTransform;
            r.anchorMin=r.anchorMax=r.pivot=new Vector2(.5f,.5f);r.anchoredPosition=Vector2.zero;r.sizeDelta=new Vector2(470,238);
            Label(r,"Finish this turn?",26,22,418,42,28);
            Label(r,_left.text+"\nUnused actions do not carry over.",26,79,418,64,17,Muted);
            Button(r,"Keep playing","Keep playing",26,171,196,40,CloseDialog,16);
            Button(r,"Confirm end turn","End turn",242,171,202,40,()=>
            {var expected=_dialogState;CloseDialog();if(ReferenceEquals(expected,_game.State)&&_input.CanCommand)_game.TryApply(new EndTurn(_game.State.ActivePlayer));},16);
        }
        public void CloseDialog(){ModalOpen=false;if(_dialog!=null){_dialog.SetActive(false);Destroy(_dialog);}_dialog=null;_dialogState=null;}
    }
}
