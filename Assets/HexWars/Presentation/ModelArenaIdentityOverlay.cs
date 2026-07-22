using System.Linq;
using UnityEngine;

namespace HexWars.Presentation
{
    public sealed class ModelArenaIdentityOverlay : MonoBehaviour
    {
        const float RowWidth = 430f;
        const float RowHeight = 32f;
        const float Padding = 8f;

        ModelDuelDriver _driver;
        GUIStyle _p1Style;
        GUIStyle _p2Style;

        void Awake() => _driver = GetComponent<ModelDuelDriver>();

        void OnGUI()
        {
            EnsureStyles();
            if (_driver == null) _driver = GetComponent<ModelDuelDriver>();
            if (_driver == null) return;

            float scale = Mathf.Max(1f, Screen.height / 1080f);
            var previousMatrix = GUI.matrix;
            GUI.matrix = Matrix4x4.TRS(Vector3.zero, Quaternion.identity, new Vector3(scale, scale, 1f));

            var rows = _driver.IdentitySnapshot();
            bool narrow = Screen.width < Screen.height;
            float logicalWidth = Screen.width / scale;
            for (int index = 0; index < rows.Length; index++)
            {
                float x = Padding + (narrow ? 0f : index * (RowWidth + Padding));
                float y = Padding + (narrow ? index * (RowHeight + Padding) : 0f);
                DrawRow(rows[index], x, y, narrow, logicalWidth);
            }

            GUI.matrix = previousMatrix;
        }

        void EnsureStyles()
        {
            if (_p1Style != null) return;
            _p1Style = new GUIStyle(GUI.skin.label) { fontSize = 16, alignment = TextAnchor.MiddleLeft };
            _p1Style.normal.textColor = new Color(0.44f, 0.69f, 1f);
            _p2Style = new GUIStyle(_p1Style);
            _p2Style.normal.textColor = new Color(1f, 0.48f, 0.42f);
        }

        void DrawRow(ModelArenaSeatIdentity row, float x, float y, bool narrow, float logicalWidth)
        {
            float width = narrow ? logicalWidth - 2f * Padding : RowWidth;
            var rect = new Rect(x, y, width, RowHeight);
            var previousColor = GUI.color;
            GUI.color = new Color(0f, 0f, 0f, 0.72f);
            GUI.DrawTexture(rect, Texture2D.whiteTexture);
            GUI.color = previousColor;

            string marker = row.IsActive ? "▶ " : "  ";
            string model = string.Join(" · ", new[] { row.Controller, row.Algorithm, row.Checkpoint, row.Step }
                .Where(value => !string.IsNullOrWhiteSpace(value)));
            string text = $"{marker}{row.Player}  {ModelArenaIdentity.MiddleTruncate(model, narrow ? 42 : 72)}  ·  {row.Record}";
            GUI.Label(new Rect(x + Padding, y, width - 2f * Padding, RowHeight), text,
                row.Player == "P1" ? _p1Style : _p2Style);
        }
    }
}
