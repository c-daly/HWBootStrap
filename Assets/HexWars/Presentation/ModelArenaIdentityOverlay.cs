using System.Linq;
using UnityEngine;

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
            }
            finally { GUI.matrix = previousMatrix; }
        }

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
            new[] { row.Step, row.Record }.Where(value => !string.IsNullOrWhiteSpace(value)));

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
}
