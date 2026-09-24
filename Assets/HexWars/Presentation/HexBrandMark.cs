using UnityEngine;
using UnityEngine.UI;

namespace HexWars.Presentation
{
    /// <summary>The H-in-hex mark from the design study, drawn as resolution-independent UI geometry.</summary>
    public sealed class HexBrandMark : MaskableGraphic
    {
        public static HexBrandMark Add(Transform parent, float x, float y, float size)
        {
            var go = new GameObject("HexWars mark"); go.transform.SetParent(parent, false);
            var mark = go.AddComponent<HexBrandMark>(); mark.color = UiKit.Accent; mark.raycastTarget = false;
            UiKit.SetRect(mark.rectTransform,x,y,size,size);
            return mark;
        }
        protected override void OnPopulateMesh(VertexHelper vh)
        {
            vh.Clear(); var r=rectTransform.rect;
            Vector2 P(float x,float y) => new Vector2(r.xMin+x/40*r.width,r.yMax-y/40*r.height);
            void Line(Vector2 a,Vector2 b)
            {
                var n=new Vector2(-(b-a).y,(b-a).x).normalized*r.width*.03125f; int i=vh.currentVertCount;
                vh.AddVert(a+n,color,Vector2.zero);vh.AddVert(b+n,color,Vector2.zero);
                vh.AddVert(b-n,color,Vector2.zero);vh.AddVert(a-n,color,Vector2.zero);
                vh.AddTriangle(i,i+1,i+2);vh.AddTriangle(i,i+2,i+3);
            }
            var points=new[]{P(20,2),P(36,11),P(36,29),P(20,38),P(4,29),P(4,11)};
            for(int i=0;i<6;i++) Line(points[i],points[(i+1)%6]);
            Line(P(12,13),P(12,27));Line(P(28,13),P(28,27));Line(P(12,20),P(28,20));
        }
    }
}
