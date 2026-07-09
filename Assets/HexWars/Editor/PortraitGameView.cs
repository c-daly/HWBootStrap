using System;
using System.Reflection;
using UnityEditor;
using UnityEngine;

namespace HexWars.EditorTools
{
    /// <summary>
    /// Verification helper: forces the Game view to a fixed 390×844 logical resolution (iPhone-ish CSS
    /// px) so Screen.width/height — and therefore every CanvasScaler in the game — matches what a phone
    /// in portrait actually sees. Outside Play Mode, Screen.width/height don't track the Game view size
    /// at all (confirmed empirically), so this only matters, and is only meant to be called, while
    /// playing. GameViewSizes has no public API for adding/selecting a custom size, hence the reflection.
    /// Resizing the Game view's EditorWindow via .position does NOT work reliably (a docked window
    /// ignores explicit position edits) — this uses GameViewSizes' own fixed-resolution selection instead,
    /// which was confirmed to force Screen.width/height exactly to the target.
    /// </summary>
    public static class PortraitGameView
    {
        const string Label = "HexWars Portrait";
        const int W = 390, H = 844;
        const int RestoreIndex = 3; // "Full HD (1920x1080)" in the default size list

        static object _gvs, _group;

        public static string Enter()
        {
            var window = GameViewWindow();
            int idx = FindOrAddSize(out object size);
            SizeSelectionCallback(window, idx, size);
            window.Repaint();
            return $"Screen={Screen.width}x{Screen.height}";
        }

        /// <summary>Back to a normal desktop size when the sweep is done.</summary>
        public static string Exit()
        {
            var window = GameViewWindow();
            object size = GetGameViewSize(RestoreIndex);
            SizeSelectionCallback(window, RestoreIndex, size);
            window.Repaint();
            return $"Screen={Screen.width}x{Screen.height}";
        }

        static void EnsureGroup()
        {
            if (_group != null) return;
            var asm = typeof(Editor).Assembly;
            var sizesType = asm.GetType("UnityEditor.GameViewSizes");
            var single = typeof(ScriptableSingleton<>).MakeGenericType(sizesType);
            _gvs = single.GetProperty("instance", BindingFlags.Public | BindingFlags.Static).GetValue(null, null);
            var currentGroupType = sizesType.GetProperty("currentGroupType").GetValue(_gvs, null);
            _group = sizesType.GetMethod("GetGroup").Invoke(_gvs, new object[] { currentGroupType });
        }

        static object GetGameViewSize(int index)
        {
            EnsureGroup();
            return _group.GetType().GetMethod("GetGameViewSize").Invoke(_group, new object[] { index });
        }

        static int FindOrAddSize(out object size)
        {
            EnsureGroup();
            var groupType = _group.GetType();
            var texts = (string[])groupType.GetMethod("GetDisplayTexts").Invoke(_group, null);
            for (int i = 0; i < texts.Length; i++)
                if (texts[i].IndexOf(Label, StringComparison.Ordinal) >= 0)
                { size = GetGameViewSize(i); return i; }

            var asm = typeof(Editor).Assembly;
            var gameViewSizeType = asm.GetType("UnityEditor.GameViewSize");
            var sizeTypeEnum = asm.GetType("UnityEditor.GameViewSizeType");
            object fixedResolution = Enum.Parse(sizeTypeEnum, "FixedResolution");
            var ctor = gameViewSizeType.GetConstructor(new[] { sizeTypeEnum, typeof(int), typeof(int), typeof(string) });
            size = ctor.Invoke(new object[] { fixedResolution, W, H, Label });
            groupType.GetMethod("AddCustomSize").Invoke(_group, new object[] { size });

            texts = (string[])groupType.GetMethod("GetDisplayTexts").Invoke(_group, null);
            for (int i = 0; i < texts.Length; i++)
                if (texts[i].IndexOf(Label, StringComparison.Ordinal) >= 0) return i;
            return texts.Length - 1; // AddCustomSize appends — fallback only if the scan above ever misses
        }

        static EditorWindow GameViewWindow()
        {
            var gameViewType = typeof(Editor).Assembly.GetType("UnityEditor.GameView");
            return EditorWindow.GetWindow(gameViewType);
        }

        static void SizeSelectionCallback(EditorWindow window, int index, object size)
        {
            var method = window.GetType().GetMethod("SizeSelectionCallback",
                BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public);
            method.Invoke(window, new object[] { index, size });
        }
    }
}
