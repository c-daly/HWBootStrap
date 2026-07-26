using System.Linq;
using System.Reflection;
using UnityEngine;
using HexWars.Engine;

namespace HexWars.Presentation
{
    public sealed class ModelArenaIdentityOverlay : MonoBehaviour
    {
        const float EventConsoleWidth = 430f;
        const float LandscapeRowHeight = 32f;
        const float PortraitRowHeight = 58f;
        const float Padding = 8f;

        ModelDuelDriver _driver;
        GUIStyle _p1Style;
        GUIStyle _p2Style;
        GUIStyle _toggleStyle;

        void Awake() => _driver = GetComponent<ModelDuelDriver>();

        void OnGUI()
        {
            if (!ShouldRender(_driver)) return;
            EnsureStyles();

            float scale = Mathf.Max(1f, Screen.height / 1080f);
            var previousMatrix = GUI.matrix;
            try
            {
                GUI.matrix = Matrix4x4.TRS(Vector3.zero, Quaternion.identity, new Vector3(scale, scale, 1f));
                var rows = _driver.IdentitySnapshot();
                bool narrow = Screen.width < Screen.height;
                float logicalWidth = Screen.width / scale;
                for (int index = 0; index < rows.Length; index++)
                {
                    DrawRow(rows[index], RowRect(index, logicalWidth, narrow), narrow);
                }
                bool fogRowShown = DrawFogMarkingToggle(rows.Length, logicalWidth, narrow);
                DrawComfortControls(rows.Length, fogRowShown, logicalWidth, narrow);
            }
            finally { GUI.matrix = previousMatrix; }
        }

        /// <summary>Spec §"Fog-of-War Indicator": the single on/off toggle for the acting-player fog
        /// marking, drawn only while the presented scenario actually trains with fog of war — otherwise
        /// there is nothing for it to hide (spec: "When fog of war is disabled, no marking is drawn").
        /// Placed directly under the identity rows, same corner and row rhythm as
        /// <see cref="RowRect"/>.</summary>
        /// <returns>Whether the row was actually drawn — callers stacking further control rows beneath
        /// it (<see cref="DrawComfortControls"/>) need to know whether to claim its row slot.</returns>
        bool DrawFogMarkingToggle(int rowCount, float logicalWidth, bool narrow)
        {
            GameState presented = _driver.PresentedState;
            if (presented == null || !presented.Config.FogOfWar) return false;

            float height = narrow ? PortraitRowHeight : LandscapeRowHeight;
            var rect = new Rect(Padding, Padding + rowCount * (height + Padding), 240f, 26f);

            var previousColor = GUI.color;
            GUI.color = new Color(0f, 0f, 0f, 0.72f);
            GUI.DrawTexture(rect, Texture2D.whiteTexture);
            GUI.color = previousColor;

            bool show = GUI.Toggle(rect, _driver.ShowFogMarking, " Fog marking (acting player)", _toggleStyle);
            if (show != _driver.ShowFogMarking) _driver.SetShowFogMarking(show);
            return true;
        }

        /// <summary>Spec: viewer comfort controls — Sound (bound to the persisted
        /// <see cref="SoundSettings.MuteAll"/>) and, in the editor, Fullscreen (maximizes/restores the
        /// Game view). Drawn as one row directly beneath the fog-marking toggle when it is shown, or in
        /// its row slot when it is not — same corner and row rhythm as <see cref="RowRect"/>.</summary>
        void DrawComfortControls(int rowCount, bool fogRowShown, float logicalWidth, bool narrow)
        {
            Rect row = ComfortControlsRowRect(rowCount, fogRowShown, logicalWidth, narrow);
            DrawSoundToggle(SoundToggleRect(row));
#if UNITY_EDITOR
            DrawFullscreenToggle(FullscreenToggleRect(row));
#endif
        }

        void DrawSoundToggle(Rect rect)
        {
            var previousColor = GUI.color;
            GUI.color = new Color(0f, 0f, 0f, 0.72f);
            GUI.DrawTexture(rect, Texture2D.whiteTexture);
            GUI.color = previousColor;

            bool soundOn = GUI.Toggle(rect, !SoundSettings.MuteAll, " Sound", _toggleStyle);
            bool wantMute = !soundOn;
            if (wantMute != SoundSettings.MuteAll) SoundSettings.MuteAll = wantMute;
        }

#if UNITY_EDITOR
        void DrawFullscreenToggle(Rect rect)
        {
            var previousColor = GUI.color;
            GUI.color = new Color(0f, 0f, 0f, 0.72f);
            GUI.DrawTexture(rect, Texture2D.whiteTexture);
            GUI.color = previousColor;

            bool maximized = GameViewFullscreen.IsMaximized();
            bool want = GUI.Toggle(rect, maximized, " Fullscreen", _toggleStyle);
            if (want != maximized) GameViewFullscreen.SetMaximized(want);
        }
#endif

        /// <summary>Row slot for the comfort-controls row: right after the identity rows, plus one more
        /// row if the fog-marking toggle claimed a slot above it.</summary>
        public static int ComfortControlsRowIndex(int rowCount, bool fogRowShown) =>
            rowCount + (fogRowShown ? 1 : 0);

        public static Rect ComfortControlsRowRect(int rowCount, bool fogRowShown, float logicalWidth, bool narrow)
        {
            float height = narrow ? PortraitRowHeight : LandscapeRowHeight;
            int index = ComfortControlsRowIndex(rowCount, fogRowShown);
            return new Rect(Padding, Padding + index * (height + Padding), 240f, 26f);
        }

        public static Rect SoundToggleRect(Rect comfortRowRect) =>
            new Rect(comfortRowRect.x, comfortRowRect.y, 112f, comfortRowRect.height);

        public static Rect FullscreenToggleRect(Rect comfortRowRect) =>
            new Rect(comfortRowRect.x + 120f, comfortRowRect.y, 120f, comfortRowRect.height);

        public static bool ShouldRender(ModelDuelDriver driver) => driver != null
            && driver.isActiveAndEnabled && driver.ShouldShowArenaOverlays;

        public static int CharacterBudget(float rowWidth, bool narrow) => !narrow ? 72
            : Mathf.Clamp(Mathf.FloorToInt(rowWidth / 8f), 24, 72);

        public static float RowWidth(float logicalWidth, bool narrow) => narrow
            ? Mathf.Max(160f, logicalWidth - 2f * Padding)
            : Mathf.Max(160f, logicalWidth - EventConsoleWidth - 2f * Padding);

        public static Rect RowRect(int index, float logicalWidth, bool narrow)
        {
            float height = narrow ? PortraitRowHeight : LandscapeRowHeight;
            return new Rect(Padding, Padding + index * (height + Padding), RowWidth(logicalWidth, narrow), height);
        }

        public static string PrefixText(ModelArenaSeatIdentity row) => (row.IsActive ? "▶ " : "  ") + row.Player;

        public static string IdentityText(ModelArenaSeatIdentity row, bool narrow)
        {
            string controller = narrow ? StripZip(row.Controller) : row.Controller;
            string checkpoint = narrow ? StripZip(row.Checkpoint) : row.Checkpoint;
            return string.Join(" · ", new[] { row.Role, controller, row.Algorithm, checkpoint, row.Status }
                .Where(value => !string.IsNullOrWhiteSpace(value)));
        }

        static string StripZip(string value) => !string.IsNullOrWhiteSpace(value)
            && value.EndsWith(".zip", System.StringComparison.OrdinalIgnoreCase)
                ? value.Substring(0, value.Length - 4) : value ?? string.Empty;

        public static string MetricsText(ModelArenaSeatIdentity row) => string.Join(" · ",
            new[] { row.Step, row.Record, PointsText(row) }.Where(value => !string.IsNullOrWhiteSpace(value)));

        /// <summary>Always non-blank (0 included): the identity row displays points continuously, per
        /// spec §"Player Point Totals", not only once a player has scored.</summary>
        static string PointsText(ModelArenaSeatIdentity row) => $"{row.Points} pts";

        public static string[] PortraitLines(ModelArenaSeatIdentity row, int characterBudget) => new[]
        {
            $"{PrefixText(row)}  {ModelArenaIdentity.MiddleTruncate(IdentityText(row, true), characterBudget)}",
            MetricsText(row),
        };

        public static string RowText(ModelArenaSeatIdentity row, int characterBudget)
        {
            return $"{PrefixText(row)}  {ModelArenaIdentity.MiddleTruncate(IdentityText(row, false), characterBudget)}  ·  {MetricsText(row)}";
        }

        void EnsureStyles()
        {
            if (_p1Style != null) return;
            _p1Style = new GUIStyle(GUI.skin.label) { fontSize = 16, alignment = TextAnchor.MiddleLeft };
            _p1Style.normal.textColor = new Color(0.44f, 0.69f, 1f);
            _p2Style = new GUIStyle(_p1Style);
            _p2Style.normal.textColor = new Color(1f, 0.48f, 0.42f);
            _toggleStyle = new GUIStyle(GUI.skin.toggle) { fontSize = 14, alignment = TextAnchor.MiddleLeft };
            _toggleStyle.normal.textColor = Color.white;
            _toggleStyle.onNormal.textColor = Color.white;
        }

        void DrawRow(ModelArenaSeatIdentity row, Rect rect, bool narrow)
        {
            var previousColor = GUI.color;
            GUI.color = new Color(0f, 0f, 0f, 0.72f);
            GUI.DrawTexture(rect, Texture2D.whiteTexture);
            GUI.color = previousColor;

            GUIStyle style = row.Player == "P1" ? _p1Style : _p2Style;
            string prefix = PrefixText(row);
            string identity = IdentityText(row, narrow);
            string metrics = MetricsText(row);
            float contentX = rect.x + Padding;
            float contentWidth = rect.width - 2f * Padding;
            float prefixWidth = style.CalcSize(new GUIContent(prefix + "  ")).x;

            if (narrow)
            {
                GUI.Label(new Rect(contentX, rect.y, prefixWidth, 28f), prefix, style);
                GUI.Label(new Rect(contentX + prefixWidth, rect.y, contentWidth - prefixWidth, 28f),
                    ModelArenaIdentity.MiddleTruncate(identity, CharacterBudget(rect.width, true)), style);
                GUI.Label(new Rect(contentX + prefixWidth, rect.y + 27f, contentWidth - prefixWidth, 28f), metrics, style);
                return;
            }

            float metricsWidth = style.CalcSize(new GUIContent(metrics)).x;
            float metricsX = rect.xMax - Padding - metricsWidth;
            float identityWidth = Mathf.Max(0f, metricsX - contentX - prefixWidth - Padding);
            GUI.Label(new Rect(contentX, rect.y, prefixWidth, rect.height), prefix, style);
            GUI.Label(new Rect(contentX + prefixWidth, rect.y, identityWidth, rect.height),
                ModelArenaIdentity.MiddleTruncate(identity, Mathf.Clamp(Mathf.FloorToInt(identityWidth / 8f), 8, 72)), style);
            GUI.Label(new Rect(metricsX, rect.y, metricsWidth, rect.height), metrics, style);
        }
    }

#if UNITY_EDITOR
    /// <summary>Reads/writes the in-editor "maximize on play" equivalent for the Game view — the closest
    /// runtime analog to OS fullscreen, since a standalone build has no Game view to maximize. Reflection
    /// is required because <c>UnityEditor.GameView</c> is internal; this Presentation asmdef is a normal
    /// runtime assembly (see <see cref="HexWars.Presentation.SpectatorDriver"/> for the same
    /// <c>#if UNITY_EDITOR</c> + <c>UnityEditor.*</c> pattern already in use here), so it compiles only
    /// into editor builds and is never linked into a player.</summary>
    static class GameViewFullscreen
    {
        static System.Type _gameViewType;
        static PropertyInfo _maximizedProperty;
        static bool _resolveFailed;

        static bool TryResolve(out UnityEditor.EditorWindow window, out PropertyInfo maximizedProperty)
        {
            window = null;
            maximizedProperty = null;
            if (_resolveFailed) return false;
            if (_gameViewType == null)
                _gameViewType = typeof(UnityEditor.EditorWindow).Assembly.GetType("UnityEditor.GameView");
            if (_gameViewType == null) { _resolveFailed = true; return false; }
            if (_maximizedProperty == null)
                _maximizedProperty = _gameViewType.GetProperty("maximized", BindingFlags.Public | BindingFlags.Instance);
            if (_maximizedProperty == null) { _resolveFailed = true; return false; }

            window = UnityEditor.EditorWindow.GetWindow(_gameViewType);
            maximizedProperty = _maximizedProperty;
            return window != null;
        }

        public static bool IsMaximized() =>
            TryResolve(out var window, out var property) && (bool)property.GetValue(window);

        public static void SetMaximized(bool maximized)
        {
            if (TryResolve(out var window, out var property)) property.SetValue(window, maximized);
        }
    }
#endif
}
