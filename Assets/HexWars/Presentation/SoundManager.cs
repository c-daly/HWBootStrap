using System.Collections.Generic;
using UnityEngine;

namespace HexWars.Presentation
{
    public enum SoundKind { Move, Attack, Death, EndTurn, Claim, Build, Win, Deploy, Design, Select, Place }

    /// <summary>Quiet, finite feedback. The original recordings remain available as attenuated fallbacks.</summary>
    public static class SoundManager
    {
        const int Rate = 44100;
        static SoundMixDriver _driver;
        static readonly Dictionary<SoundKind, AudioClip> _clips = new Dictionary<SoundKind, AudioClip>();
        static readonly HashSet<SoundKind> _prepared = new HashSet<SoundKind>();
        static readonly Dictionary<int, AudioClip[]> _attackVariants = new Dictionary<int, AudioClip[]>();
        static readonly Dictionary<int, bool[]> _attackPrepared = new Dictionary<int, bool[]>();
        static readonly string[] AttackFamilies = { "Light", "Mid", "Heavy" };
        static readonly int[] PreviousAttack = { -1, -1, -1 };
        static AudioClip[] _moves;
        static int _previousMove = -1;
        // Audio variation must not consume Unity's gameplay/cosmetic random stream.
        static readonly System.Random Variation = new System.Random(25419);

        /// <summary>Title-demo battle gate. User mute lives in SoundSettings.</summary>
        public static bool Muted;

        public static void Play(SoundKind kind)
        {
            if (Muted || SoundSettings.MuteAll || SoundSettings.Effects <= 0f) return;
            if (kind == SoundKind.Attack) { PlayAttack(1); return; }
            if (kind == SoundKind.Move) { PlayMove(); return; }
            Ensure();
            var clip = Clip(kind);
            float cooldown = kind == SoundKind.Select ? .09f : kind == SoundKind.EndTurn ? .3f : .075f;
            int priority = kind == SoundKind.Win ? 3 : kind == SoundKind.Death ? 2 : kind == SoundKind.Select ? 0 : 1;
            _driver.Play(kind.ToString(), clip, _prepared.Contains(kind) ? 1f : .18f, cooldown, priority);
        }

        static void PlayMove()
        {
            Ensure();
            if (_moves == null)
            {
                _moves = new AudioClip[4];
                _moves[0] = Clip(SoundKind.Move);
                for (int i = 1; i < _moves.Length; i++)
                    _moves[i] = Resources.Load<AudioClip>("Audio/Soft/Move_" + i) ?? _moves[0];
            }
            int pick = _previousMove < 0 ? Variation.Next(4) : (_previousMove + 1 + Variation.Next(3)) % 4;
            if (_driver.Play("Move", _moves[pick], _prepared.Contains(SoundKind.Move) ? 1f : .18f, .075f, 1))
                _previousMove = pick;
        }

        public static void PlayAttack(int tier)
        {
            if (Muted || SoundSettings.MuteAll || SoundSettings.Effects <= 0f) return;
            Ensure();
            int t = Mathf.Clamp(tier, 0, 2);
            if (!_attackVariants.TryGetValue(t, out var variants))
            {
                variants = new AudioClip[4]; var prepared = new bool[4];
                for (int i = 0; i < variants.Length; i++)
                {
                    string name = $"Attack{AttackFamilies[t]}_{i}";
                    variants[i] = Resources.Load<AudioClip>("Audio/Soft/" + name);
                    prepared[i] = variants[i] != null;
                    if (variants[i] == null) variants[i] = Resources.Load<AudioClip>("Audio/" + name);
                }
                _attackVariants[t] = variants; _attackPrepared[t] = prepared;
            }
            int pick = PreviousAttack[t] < 0 ? Variation.Next(4) : (PreviousAttack[t] + 1 + Variation.Next(3)) % 4;
            if (_driver.Play("Attack", variants[pick] != null ? variants[pick] : Clip(SoundKind.Attack),
                _attackPrepared[t][pick] ? 1f : .18f, .09f, 2)) PreviousAttack[t] = pick;
        }

        public static void StartTitleMusic() { Ensure(); _driver.Title(true); }
        public static void StopTitleMusic() { if (_driver != null) _driver.Title(false); }
        public static void StartAmbience() { Ensure(); _driver.Ambience(true); }
        public static void StopAmbience() { if (_driver != null) _driver.Ambience(false); }
        public static void StartDesignerHum() { Ensure(); _driver.Designer(true); }
        public static void StopDesignerHum() { if (_driver != null) _driver.Designer(false); }

        static void Ensure()
        {
            if (_driver != null) return;
            var go = new GameObject("HexWarsSound");
            if (Application.isPlaying) Object.DontDestroyOnLoad(go);
            _driver = go.AddComponent<SoundMixDriver>();
        }

        static AudioClip Clip(SoundKind kind)
        {
            if (_clips.TryGetValue(kind, out var clip)) return clip;
            clip = Resources.Load<AudioClip>("Audio/Soft/" + kind);
            if (clip != null) _prepared.Add(kind);
            else clip = Build(kind);
            _clips[kind] = clip;
            return clip;
        }

        static AudioClip Build(SoundKind kind)
        {
            switch (kind)
            {
                case SoundKind.Attack: return Whoosh("attack", .34f);
                case SoundKind.Death: return Explosion("death", .38f, .35f);
                case SoundKind.Select: return Click("select", 430f, .075f, .10f);
                case SoundKind.Move: return Click("move", 175f, .18f, .18f);
                case SoundKind.Place: return Click("place", 210f, .14f, .18f);
                case SoundKind.EndTurn: return Chime("endturn", new[] {392f, 523.25f}, .12f);
                case SoundKind.Claim: return Chime("claim", new[] {330f, 440f}, .12f);
                case SoundKind.Build: return Click("build", 150f, .2f, .22f);
                case SoundKind.Win: return Chime("complete", new[] {261.63f, 329.63f, 392f}, .16f);
                case SoundKind.Deploy: return Resources.Load<AudioClip>("Audio/DeployDoor") ?? Build(SoundKind.Build);
                case SoundKind.Design: return Resources.Load<AudioClip>("Audio/CreateRacked") ?? Build(SoundKind.Build);
                default: return Click("contact", 300f, .1f, .15f);
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
