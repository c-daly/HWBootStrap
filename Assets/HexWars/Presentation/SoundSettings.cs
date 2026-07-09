using UnityEngine;

namespace HexWars.Presentation
{
    /// <summary>
    /// Master audio settings — mute and volume — applied through <see cref="AudioListener.volume"/>
    /// (every source routes through it: procedural one-shots, music, ambience, hum) and persisted in
    /// PlayerPrefs (IndexedDB on WebGL, same as the seat token). Distinct from
    /// <see cref="SoundManager.Muted"/>, which is the title-demo battle-SFX gate, not a user setting.
    /// </summary>
    public static class SoundSettings
    {
        const string VolKey = "HexWars.Volume", MuteKey = "HexWars.MuteAll";

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void ApplyOnBoot() => Apply();

        /// <summary>0..1. Kept independent of mute so unmuting restores the previous level.</summary>
        public static float Volume
        {
            get => PlayerPrefs.GetFloat(VolKey, 1f);
            set { PlayerPrefs.SetFloat(VolKey, Mathf.Clamp01(value)); Apply(); }
        }

        public static bool MuteAll
        {
            get => PlayerPrefs.GetInt(MuteKey, 0) == 1;
            set { PlayerPrefs.SetInt(MuteKey, value ? 1 : 0); Apply(); }
        }

        static void Apply() => AudioListener.volume = MuteAll ? 0f : Volume;
    }
}
