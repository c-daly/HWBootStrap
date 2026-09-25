using UnityEngine;

namespace HexWars.Presentation
{
    /// <summary>Independent persisted buses. Existing master volume and mute preferences are retained.</summary>
    public static class SoundSettings
    {
        const string VolKey = "HexWars.Volume", MuteKey = "HexWars.MuteAll";
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void ApplyOnBoot() => Apply();

        public static float Volume
        {
            get => PlayerPrefs.GetFloat(VolKey, 1f);
            set { Save(VolKey, value); Apply(); }
        }
        public static float Effects
        {
            get => PlayerPrefs.GetFloat("HexWars.EffectsVolume", 1f);
            set => Save("HexWars.EffectsVolume", value);
        }
        public static float Atmosphere
        {
            get => PlayerPrefs.GetFloat("HexWars.AtmosphereVolume", .5f);
            set => Save("HexWars.AtmosphereVolume", value);
        }
        public static float Music
        {
            get => PlayerPrefs.GetFloat("HexWars.MusicVolume", .5f);
            set => Save("HexWars.MusicVolume", value);
        }
        public static bool MuteAll
        {
            get => PlayerPrefs.GetInt(MuteKey, 0) == 1;
            set { PlayerPrefs.SetInt(MuteKey, value ? 1 : 0); PlayerPrefs.Save(); Apply(); }
        }
        static void Save(string key, float value) { PlayerPrefs.SetFloat(key, Mathf.Clamp01(value)); PlayerPrefs.Save(); }
        static void Apply() => AudioListener.volume = MuteAll ? 0f : Volume;
    }
}
