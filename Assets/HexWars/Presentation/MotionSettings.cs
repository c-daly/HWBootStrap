using UnityEngine;
namespace HexWars.Presentation
{
    /// <summary>Persisted presentation preference; never changes engine pacing or rules.</summary>
    public static class MotionSettings
    {
        public static bool Reduced
        {
            get => PlayerPrefs.GetInt("HexWars.ReducedMotion", 0) != 0;
            set { PlayerPrefs.SetInt("HexWars.ReducedMotion", value ? 1 : 0); PlayerPrefs.Save(); }
        }
    }
}
