using System.Collections.Generic;
using UnityEngine;

namespace HexWars.Presentation
{
    public enum SoundKind { Move, Attack, Death, EndTurn, Claim, Build, Win, Deploy, Design }

    /// <summary>
    /// Procedural SFX synthesized in code as the baseline (no assets required to ship), layered with a
    /// handful of user-supplied clips (title music, ambient bed, deploy door, designer hum, tiered weapon
    /// shots, design-racked chime) loaded from Resources. Every asset-backed sound keeps its procedural
    /// fallback so a missing/failed Resources.Load degrades silently, never throws. Procedural kit:
    /// filtered-noise explosions with a low rumble, a swept "rocket" whoosh for attacks, and soft sine
    /// clicks for UI. One persistent AudioSource plays one-shots; callers just say
    /// <c>SoundManager.Play(SoundKind.Attack)</c>. Three more looping sources on the same GameObject carry
    /// the title music, in-game ambience, and designer hum.
    /// </summary>
    public static class SoundManager
    {
        const int Rate = 44100;
        static AudioSource _src;
        static readonly Dictionary<SoundKind, AudioClip> _clips = new Dictionary<SoundKind, AudioClip>();

        /// <summary>True while the title demo plays — the menu should be calm, not a battle radio.
        /// Deliberately NOT checked by the music loop: the soundtrack owns the title screen even while
        /// battle SFX are muted.</summary>
        public static bool Muted;

        public static void Play(SoundKind kind)
        {
            if (Muted) return;
            Ensure();
            _src.PlayOneShot(Clip(kind));
        }

        // ---- tiered weapon shots: 4 recorded variants per tier, random pick each hit, whoosh fallback ----
        static readonly string[] AttackFamilies = { "Light", "Mid", "Heavy" };
        static readonly Dictionary<int, AudioClip[]> _attackVariants = new Dictionary<int, AudioClip[]>();

        /// <summary>tier 0/1/2 = light/mid/heavy, matching ActionPresenter's projectile-tier thresholds
        /// exactly (same value drives the projectile color/scale). Picks a random one of 4 recorded
        /// variants per tier; a variant that failed to load (or the whole family missing) falls back to
        /// the procedural attack whoosh — never silent, never throws.</summary>
        public static void PlayAttack(int tier)
        {
            if (Muted) return;
            Ensure();
            var variants = AttackVariants(tier);
            var clip = variants[UnityEngine.Random.Range(0, 4)];
            _src.PlayOneShot(clip != null ? clip : Clip(SoundKind.Attack));
        }

        static AudioClip[] AttackVariants(int tier)
        {
            int t = Mathf.Clamp(tier, 0, AttackFamilies.Length - 1);
            if (_attackVariants.TryGetValue(t, out var cached)) return cached;
            var arr = new AudioClip[4];
            for (int i = 0; i < 4; i++)
                arr[i] = Resources.Load<AudioClip>($"Audio/Attack{AttackFamilies[t]}_{i}");
            _attackVariants[t] = arr; // cache regardless of hits/misses — never re-probe Resources per shot
            return arr;
        }

        // ---- looping beds: title music, in-game ambience, designer hum ----
        static AudioSource _music, _ambience, _hum;
        static AudioClip _musicClip, _ambienceClip, _humClip;
        const string MusicPath = "Audio/TitleMusic", AmbiencePath = "Audio/AmbientBed", HumPath = "Audio/DesignerHum";

        /// <summary>Idempotent: already-playing is a no-op, so callers don't need to track state
        /// themselves. Ignores <see cref="Muted"/> on purpose — see its doc comment.</summary>
        public static void StartTitleMusic() => StartLoop(ref _music, ref _musicClip, MusicPath, 0.35f);
        public static void StopTitleMusic() => StopLoop(_music);

        public static void StartAmbience() => StartLoop(ref _ambience, ref _ambienceClip, AmbiencePath, 0.15f);
        public static void StopAmbience() => StopLoop(_ambience);

        public static void StartDesignerHum() => StartLoop(ref _hum, ref _humClip, HumPath, 0.12f);
        public static void StopDesignerHum() => StopLoop(_hum);

        static void StartLoop(ref AudioSource src, ref AudioClip clip, string path, float volume)
        {
            Ensure();
            if (src == null)
            {
                src = _src.gameObject.AddComponent<AudioSource>();
                src.playOnAwake = false;
                src.loop = true;
                src.volume = volume;
            }
            if (src.isPlaying) return; // idempotent: Start while already playing does not restart
            if (clip == null) clip = Resources.Load<AudioClip>(path); // load once, cache — null-safe below
            if (clip == null) return;  // missing asset: silent degrade, never throw
            src.clip = clip;
            src.Play();
        }

        static void StopLoop(AudioSource src)
        {
            if (src != null && src.isPlaying) src.Stop(); // idempotent: Stop while stopped is a no-op
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
                // asset-backed one-shots: recorded clip when available, else the procedural click that
                // best matches the moment (Build's construction thunk) — same fallback pattern as PlayAttack
                case SoundKind.Deploy:  return Resources.Load<AudioClip>("Audio/DeployDoor") ?? Clip(SoundKind.Build);
                case SoundKind.Design:  return Resources.Load<AudioClip>("Audio/CreateRacked") ?? Clip(SoundKind.Build);
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
