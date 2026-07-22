using UnityEditor;
using HexWars.Presentation.EditorTools.MlLab;

namespace HexWars.Presentation.EditorTools
{
    /// <summary>Compatibility menu alias for the integrated ML Lab.</summary>
    public static class TrainingLauncher
    {
        [MenuItem("HexWars/Start Training...", priority = 21)]
        public static void Open() => MlLabWindow.Open();
    }
}
