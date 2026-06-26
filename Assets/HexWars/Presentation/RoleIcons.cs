using System.Collections.Generic;
using UnityEngine;
using HexWars.Engine;

namespace HexWars.Presentation
{
    /// <summary>
    /// Procedurally drawn role-icon textures (light symbol on a transparent background), one per
    /// <see cref="UnitRole"/>, so players can tell units apart at a glance. Cached.
    /// Swap for hand-drawn icon art later without changing callers.
    /// </summary>
    public static class RoleIcons
    {
        const int N = 48;
        static readonly Dictionary<UnitRole, Texture2D> Cache = new Dictionary<UnitRole, Texture2D>();
        static readonly Color32 Glyph = new Color32(245, 245, 250, 255); // light symbol — reads on the colored discs
        static readonly Color32 Clear = new Color32(0, 0, 0, 0);          // transparent — the icon sits on the piece

        public static Texture2D For(UnitRole role)
        {
            if (Cache.TryGetValue(role, out var t)) return t;
            t = Build(role);
            Cache[role] = t;
            return t;
        }

        static Texture2D Build(UnitRole role)
        {
            var px = new Color32[N * N];
            for (int i = 0; i < px.Length; i++) px[i] = Clear; // badge background

            switch (role)
            {
                case UnitRole.Striker: // sword: blade + guard
                    Rect(px, 22, 6, 26, 38);
                    Rect(px, 13, 30, 35, 34);
                    Disc(px, 24, 40, 4);
                    break;
                case UnitRole.Sniper: // target: rings + dot
                    Ring(px, 24, 24, 18, 3);
                    Ring(px, 24, 24, 9, 3);
                    Disc(px, 24, 24, 3);
                    break;
                case UnitRole.Spotter: // eye: outline + pupil
                    Ring(px, 24, 24, 16, 3);
                    Disc(px, 24, 24, 6);
                    break;
                case UnitRole.Runner: // chevrons ">>"
                    Line(px, 12, 10, 26, 24, 4); Line(px, 26, 24, 12, 38, 4);
                    Line(px, 22, 10, 36, 24, 4); Line(px, 36, 24, 22, 38, 4);
                    break;
                case UnitRole.Climber: // up arrow
                    Rect(px, 22, 14, 26, 40);
                    Line(px, 24, 7, 11, 22, 4); Line(px, 24, 7, 37, 22, 4);
                    break;
                case UnitRole.Bulwark: // shield / armor block (thick square ring)
                    Rect(px, 10, 10, 38, 38);
                    Rect(px, 16, 16, 32, 32, Clear);
                    break;
                case UnitRole.Brute: // plus (health)
                    Rect(px, 21, 9, 27, 39);
                    Rect(px, 9, 21, 39, 27);
                    break;
                case UnitRole.Generalist: // filled block (no dominant stat)
                    Rect(px, 15, 15, 33, 33);
                    break;
            }

            var tex = new Texture2D(N, N, TextureFormat.RGBA32, false) { filterMode = FilterMode.Bilinear };
            tex.SetPixels32(px);
            tex.Apply();
            return tex;
        }

        static void Rect(Color32[] px, int x0, int y0, int x1, int y1) => Rect(px, x0, y0, x1, y1, Glyph);
        static void Rect(Color32[] px, int x0, int y0, int x1, int y1, Color32 c)
        {
            for (int y = Mathf.Max(0, y0); y <= Mathf.Min(N - 1, y1); y++)
                for (int x = Mathf.Max(0, x0); x <= Mathf.Min(N - 1, x1); x++)
                    px[y * N + x] = c;
        }

        static void Disc(Color32[] px, int cx, int cy, int r)
        {
            for (int y = -r; y <= r; y++)
                for (int x = -r; x <= r; x++)
                    if (x * x + y * y <= r * r) Plot(px, cx + x, cy + y, Glyph);
        }

        static void Ring(Color32[] px, int cx, int cy, int r, int t)
        {
            int outer = r * r, inner = (r - t) * (r - t);
            for (int y = -r; y <= r; y++)
                for (int x = -r; x <= r; x++)
                {
                    int d = x * x + y * y;
                    if (d <= outer && d >= inner) Plot(px, cx + x, cy + y, Glyph);
                }
        }

        static void Line(Color32[] px, int x0, int y0, int x1, int y1, int thick)
        {
            int steps = Mathf.Max(Mathf.Abs(x1 - x0), Mathf.Abs(y1 - y0));
            if (steps == 0) steps = 1;
            int h = thick / 2;
            for (int i = 0; i <= steps; i++)
            {
                float u = (float)i / steps;
                int px0 = Mathf.RoundToInt(Mathf.Lerp(x0, x1, u));
                int py0 = Mathf.RoundToInt(Mathf.Lerp(y0, y1, u));
                for (int dy = -h; dy <= h; dy++)
                    for (int dx = -h; dx <= h; dx++)
                        Plot(px, px0 + dx, py0 + dy, Glyph);
            }
        }

        static void Plot(Color32[] px, int x, int y, Color32 c)
        {
            if (x >= 0 && x < N && y >= 0 && y < N) px[y * N + x] = c;
        }
    }
}
