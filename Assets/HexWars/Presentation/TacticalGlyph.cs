using UnityEngine;
using UnityEngine.UI;

namespace HexWars.Presentation
{
    /// <summary>Small code-native action symbols, shared by the inspector and squad.</summary>
    [RequireComponent(typeof(CanvasRenderer))]
    public sealed class TacticalGlyph : MaskableGraphic
    {
        public enum Shape { Move, Attack, Shield, Eye }
        public Shape Symbol;
        protected override void OnPopulateMesh(VertexHelper vh)
        {
            vh.Clear(); var rect = rectTransform.rect;
            void Line(float x,float y,float x2,float y2)
            {
                var a=new Vector2(rect.x+x*rect.width,rect.y+y*rect.height);
                var b=new Vector2(rect.x+x2*rect.width,rect.y+y2*rect.height);
                var n=new Vector2(-(b-a).y,(b-a).x).normalized * Mathf.Min(rect.width,rect.height)*.038f;
                int i=vh.currentVertCount;
                vh.AddVert(a+n,color,Vector2.zero);vh.AddVert(b+n,color,Vector2.zero);
                vh.AddVert(b-n,color,Vector2.zero);vh.AddVert(a-n,color,Vector2.zero);
                vh.AddTriangle(i,i+1,i+2);vh.AddTriangle(i,i+2,i+3);
            }
            if(Symbol==Shape.Move){Line(.12f,.20f,.40f,.20f);Line(.4f,.2f,.4f,.78f);Line(.4f,.78f,.88f,.78f);Line(.68f,.98f,.88f,.78f);Line(.88f,.78f,.68f,.58f);}
            if(Symbol==Shape.Shield){Line(.5f,.94f,.87f,.76f);Line(.87f,.76f,.82f,.36f);Line(.82f,.36f,.5f,.06f);Line(.5f,.06f,.18f,.36f);Line(.18f,.36f,.13f,.76f);Line(.13f,.76f,.5f,.94f);}
            if(Symbol==Shape.Attack || Symbol==Shape.Eye)
            {
                int n=24;float radius=Symbol==Shape.Attack?.27f:.17f;
                for(int j=0;j<n;j++){float a=j*Mathf.PI*2/n,b=(j+1)*Mathf.PI*2/n;Line(.5f+Mathf.Cos(a)*radius,.5f+Mathf.Sin(a)*radius,.5f+Mathf.Cos(b)*radius,.5f+Mathf.Sin(b)*radius);}
                if(Symbol==Shape.Attack){Line(.5f,.02f,.5f,.28f);Line(.5f,.72f,.5f,.98f);Line(.02f,.5f,.28f,.5f);Line(.72f,.5f,.98f,.5f);}
                else {Line(.02f,.5f,.28f,.79f);Line(.28f,.79f,.72f,.79f);Line(.72f,.79f,.98f,.5f);Line(.98f,.5f,.72f,.21f);Line(.72f,.21f,.28f,.21f);Line(.28f,.21f,.02f,.5f);}
            }
        }
    }
}
