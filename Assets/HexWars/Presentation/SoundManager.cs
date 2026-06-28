using System.Collections.Generic;
using UnityEngine;

namespace HexWars.Presentation
{
    public enum SoundKind { Move, Attack, Death, EndTurn, Claim, Build, Win }

    /// <summary>
    /// Procedural SFX synthesized in code (no audio assets to ship), but noise/filter-based rather than
    /// arcade beeps: filtered-noise explosions with a low rumble, a swept "rocket" whoosh for attacks, and
    /// soft sine clicks for UI. One persistent AudioSource plays one-shots; callers just say
    /// <c>SoundManager.Play(SoundKind.Attack)</c>.
    /// </summary>
    public static class SoundManager
    {
        const int Rate = 44100;
        static AudioSource _src;
        static readonly Dictionary<SoundKind, AudioClip> _clips = new Dictionary<SoundKind, AudioClip>();

        public static void Play(SoundKind kind)
        {
            Ensure();
            _src.PlayOneShot(Clip(kind));
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
                case SoundKind.Attack:  return Whoosh("attack", 0.34f);          // rocket launch → small burst
                case SoundKind.Death:   return Explosion("death", 0.70f, 0.55f); // bigger boom
                case SoundKind.Move:    return Click("move", 660f, 0.07f, 0.18f);
                case SoundKind.EndTurn: return Click("endturn", 330f, 0.11f, 0.22f);
                case SoundKind.Claim:   return Chime("claim", new[] { 523f, 784f }, 0.16f);
                case SoundKind.Build:   return Click("build", 494f, 0.10f, 0.22f);
                case SoundKind.Win:     return Chime("win", new[] { 523f, 659f, 784f, 1047f }, 0.16f);
                default:                return Click("blip", 600f, 0.08f, 0.2f);
            }
        }

        // ---- explosion: lowpassed white-noise burst + a sub rumble, fast attack, exponential tail ----
        static AudioClip Explosion(string name, float dur, float vol)
        {
            int n = (int)(Rate * dur);
            var s = new float[n];
            var rng = new System.Random(name.GetHashCode());
            float lp = 0f;                       // one-pole lowpass state
            for (int i = 0; i < n; i++)
            {
                float t = i / (float)Rate;
                float attack = Mathf.Clamp01(t / 0.008f);     // ~8ms punch
                float env = attack * Mathf.Exp(-t * 7f);
                float white = (float)(rng.NextDouble() * 2.0 - 1.0);
                lp += 0.12f * (white - lp);                   // boomy lowpass
                float sub = Mathf.Sin(2f * Mathf.PI * 55f * t) * Mathf.Exp(-t * 9f); // low rumble
                s[i] = (lp * 0.9f + sub * 0.6f) * env * vol;
            }
            return Make(name, s);
        }

        // ---- rocket whoosh: noise with a sweeping lowpass that opens then a short tail ----
        static AudioClip Whoosh(string name, float dur)
        {
            int n = (int)(Rate * dur);
            var s = new float[n];
            var rng = new System.Random(name.GetHashCode());
            float lp = 0f;
            for (int i = 0; i < n; i++)
            {
                float u = i / (float)n;
                float t = i / (float)Rate;
                float cutoff = Mathf.Lerp(0.02f, 0.30f, u);   // lowpass opens up = "launch"
                float white = (float)(rng.NextDouble() * 2.0 - 1.0);
                lp += cutoff * (white - lp);
                float env = Mathf.Sin(u * Mathf.PI) * 0.9f + (u > 0.85f ? 0.4f : 0f); // swell + tiny burst at end
                s[i] = lp * env * 0.55f;
            }
            return Make(name, s);
        }

        // ---- soft modern UI click: sine with quick attack/decay and a slight downward pitch glide ----
        static AudioClip Click(string name, float freq, float dur, float vol)
        {
            int n = (int)(Rate * dur);
            var s = new float[n];
            float phase = 0f;
            for (int i = 0; i < n; i++)
            {
                float u = i / (float)n;
                float f = freq * Mathf.Lerp(1.0f, 0.85f, u);   // gentle glide down
                phase += f / Rate * 2f * Mathf.PI;
                float env = Mathf.Min(1f, (1f - u) * 4f) * (1f - u); // soft, no hard edges
                s[i] = Mathf.Sin(phase) * env * vol;
            }
            return Make(name, s);
        }

        // ---- chime: a couple of soft sine notes in sequence (confirm / win) ----
        static AudioClip Chime(string name, float[] notes, float each)
        {
            int per = (int)(Rate * each);
            int n = per * notes.Length;
            var s = new float[n];
            for (int k = 0; k < notes.Length; k++)
            {
                float phase = 0f;
                for (int i = 0; i < per; i++)
                {
                    float u = i / (float)per;
                    phase += notes[k] / Rate * 2f * Mathf.PI;
                    float env = Mathf.Min(1f, u * 6f) * Mathf.Exp(-u * 4f);
                    s[k * per + i] = Mathf.Sin(phase) * env * 0.3f;
                }
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
