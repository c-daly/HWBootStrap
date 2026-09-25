using System.Collections.Generic;
using UnityEngine;

namespace HexWars.Presentation
{
    /// <summary>Six finite effect voices and three gently fading 2D beds. No gameplay state is touched.</summary>
    public sealed class SoundMixDriver : MonoBehaviour
    {
        const int VoiceCount = 6;
        readonly AudioSource[] _voices = new AudioSource[VoiceCount];
        readonly float[] _gains = new float[VoiceCount];
        readonly int[] _priorities = new int[VoiceCount];
        readonly Dictionary<string, double> _lastPlayed = new Dictionary<string, double>();
        AudioSource _music, _ambience, _hum;
        bool _titleWanted, _ambienceWanted, _designerWanted;
        bool _focused = true;
        float _musicLevel, _ambienceLevel, _humLevel;
        float _duckUntil;

        AudioSource Source(string name, bool loop)
        {
            var go = new GameObject(name); go.transform.SetParent(transform, false);
            var src = go.AddComponent<AudioSource>();
            src.playOnAwake = false; src.loop = loop; src.spatialBlend = 0f;
            return src;
        }
        void Awake()
        {
            for (int i = 0; i < VoiceCount; i++) _voices[i] = Source("Effect " + i, false);
            _music = Source("Music", true); _ambience = Source("Ambience", true); _hum = Source("Workshop", true);
        }

        public bool Play(string key, AudioClip clip, float gain, float cooldown, int priority)
        {
            if (!_focused || clip == null || SoundSettings.MuteAll || SoundSettings.Effects <= 0f) return false;
            double now = Time.unscaledTimeAsDouble;
            if (_lastPlayed.TryGetValue(key, out var last) && now - last < cooldown) return false;
            int slot = -1;
            for (int i = 0; i < VoiceCount; i++) if (!_voices[i].isPlaying) { slot = i; break; }
            if (slot < 0)
            {
                // Quiet UI ticks never displace an attack or conclusion; lower-priority chatter can yield.
                for (int i = 0; i < VoiceCount; i++)
                    if (_priorities[i] < priority && (slot < 0 || _priorities[i] < _priorities[slot])) slot = i;
            }
            if (slot < 0) return false;
            _lastPlayed[key] = now;
            _gains[slot] = gain; _priorities[slot] = priority;
            var voice = _voices[slot]; voice.Stop(); voice.clip = clip;
            voice.volume = gain * SoundSettings.Effects; voice.Play();
            if (priority >= 2) _duckUntil = Time.unscaledTime + .6f;
            return true;
        }

        public void Title(bool wanted)
        {
            _titleWanted = wanted;
            if (wanted) { _ambienceWanted = false; _designerWanted = false; }
        }
        public void Ambience(bool wanted)
        {
            _ambienceWanted = wanted;
            if (wanted) _titleWanted = false;
        }
        public void Designer(bool wanted) => _designerWanted = wanted;

        void Update() => Tick(Time.unscaledDeltaTime);
        void Tick(float dt)
        {
            for (int i = 0; i < VoiceCount; i++)
            {
                _voices[i].volume = _gains[i] * SoundSettings.Effects;
                if (SoundSettings.MuteAll || SoundSettings.Effects <= 0f) _voices[i].Stop();
            }
            bool audible = _focused && !SoundSettings.MuteAll;
            float duck = Time.unscaledTime < _duckUntil ? .55f : 1f;
            // Opt-in music also spans setup and online waiting, before board ambience starts.
            bool musicWanted = _titleWanted || SoundSettings.MusicDuringGame;
            Fade(_music, "Tabletop/TitleTheme", ref _musicLevel, audible && musicWanted ? .24f * SoundSettings.Music : 0f, dt, 0f);
            Fade(_ambience, "AmbientBed", ref _ambienceLevel,
                audible && _ambienceWanted ? .1f * SoundSettings.Atmosphere * duck * (_designerWanted ? .3f : 1f) : 0f, dt, 0f);
            Fade(_hum, "DesignerHum", ref _humLevel, audible && _designerWanted ? .18f * SoundSettings.Atmosphere : 0f, dt, .18f);
        }

        static void Fade(AudioSource source, string asset, ref float level, float target, float dt, float seam)
        {
            level = Mathf.Lerp(level, target, 1f - Mathf.Exp(-dt * 4f));
            if (target > 0f && !source.isPlaying)
            {
                if (source.clip == null) source.clip = Resources.Load<AudioClip>("Audio/" + asset);
                if (source.clip != null) { source.volume = 0f; source.Play(); }
            }
            // The supplied hum endpoints do not meet. The new piano theme already has a continuous loop.
            float edge = source.clip != null && seam > 0f ?
                Mathf.SmoothStep(0f, 1f, Mathf.Min(source.time, source.clip.length - source.time) / seam) : 1f;
            source.volume = level * edge;
            if (target == 0f && level < .0001f) { source.Stop(); level = 0f; }
        }

        void OnApplicationFocus(bool focus)
        {
            _focused = focus;
            if (focus) return;
            foreach (var voice in _voices) if (voice != null) voice.Stop();
            // Update may stop while unfocused: silence now and fade back in when focus returns.
            _musicLevel = _ambienceLevel = _humLevel = 0f;
            foreach (var loop in new[] {_music, _ambience, _hum}) if (loop != null) loop.volume = 0f;
        }
    }
}
