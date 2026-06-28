using System.Collections.Generic;
using UnityEngine;

namespace HexWars.Presentation
{
    public enum SoundKind { Move, Attack, Death, EndTurn, Claim, Build, Win }

    /// <summary>
    /// Tiny procedural SFX — clips are synthesized in code (tones / sweeps / a win arpeggio), so there are
    /// no audio assets to ship. One persistent AudioSource plays them as one-shots. Callers just say
    /// <c>SoundManager.Play(SoundKind.Move)</c>.
    /// </summary>
    public static class SoundManager
    {
        const int Rate = 44100;
        static AudioSource _src;
        static readonly Dictionary<SoundKind, AudioClip> _clips = new Dictionary<SoundKind, AudioClip>();

        public static void Play(SoundKind kind)
        {
            Ensure();
            _src.PlayOneShot(Clip(kind), 0.6f);
        }

        static void Ensure()
        {
            if (_src != null) return;
            var go = new GameObject("HexWarsSound");
            Object.DontDestroyOnLoad(go);
            _src = go.AddComponent<AudioSource>();
            _src.playOnAwake = false;
        }

        static AudioClip Clip(SoundKind kind)
        {
            if (!_clips.TryGetValue(kind, out var c)) { c = Build(kind); _clips[kind] = c; }
            return c;
        }

        static AudioClip Build(SoundKind kind)
        {
            switch (kind)
            {
                case SoundKind.Move:    return Tone("move", 520f, 0.10f, 18f, false, 0f);
                case SoundKind.Attack:  return Tone("attack", 170f, 0.18f, 13f, true, 0.30f);
                case SoundKind.Death:   return Sweep("death", 420f, 80f, 0.40f, 6f);
                case SoundKind.EndTurn: return Tone("endturn", 300f, 0.12f, 16f, false, 0f);
                case SoundKind.Claim:   return Sweep("claim", 300f, 640f, 0.22f, 8f);
                case SoundKind.Build:   return Tone("build", 440f, 0.14f, 12f, false, 0f);
                case SoundKind.Win:     return Arp("win", new[] { 392f, 523f, 659f, 784f }, 0.55f);
                default:                return Tone("blip", 440f, 0.10f, 16f, false, 0f);
            }
        }

        static AudioClip Tone(string name, float freq, float dur, float decay, bool square, float noise)
        {
            int n = (int)(Rate * dur);
            var s = new float[n];
            var rng = new System.Random(name.GetHashCode());
            for (int i = 0; i < n; i++)
            {
                float t = i / (float)Rate;
                float env = Mathf.Exp(-t * decay);
                float w = square ? (Mathf.Sin(freq * t * 2f * Mathf.PI) >= 0f ? 1f : -1f)
                                 : Mathf.Sin(freq * t * 2f * Mathf.PI);
                if (noise > 0f) w = Mathf.Lerp(w, (float)(rng.NextDouble() * 2.0 - 1.0), noise);
                s[i] = w * env * 0.4f;
            }
            return Make(name, s);
        }

        static AudioClip Sweep(string name, float f0, float f1, float dur, float decay)
        {
            int n = (int)(Rate * dur);
            var s = new float[n];
            float phase = 0f;
            for (int i = 0; i < n; i++)
            {
                float freq = Mathf.Lerp(f0, f1, i / (float)n);
                phase += freq / Rate * 2f * Mathf.PI;
                float env = Mathf.Exp(-(i / (float)Rate) * decay);
                s[i] = Mathf.Sin(phase) * env * 0.4f;
            }
            return Make(name, s);
        }

        static AudioClip Arp(string name, float[] notes, float dur)
        {
            int n = (int)(Rate * dur);
            var s = new float[n];
            int per = Mathf.Max(1, n / notes.Length);
            for (int i = 0; i < n; i++)
            {
                int note = Mathf.Min(i / per, notes.Length - 1);
                float tIn = (i - note * per) / (float)Rate;
                float env = Mathf.Exp(-tIn * 9f);
                s[i] = Mathf.Sin(notes[note] * (i / (float)Rate) * 2f * Mathf.PI) * env * 0.4f;
            }
            return Make(name, s);
        }

        static AudioClip Make(string name, float[] samples)
        {
            var clip = AudioClip.Create(name, samples.Length, 1, Rate, false);
            clip.SetData(samples, 0);
            return clip;
        }
    }
}
