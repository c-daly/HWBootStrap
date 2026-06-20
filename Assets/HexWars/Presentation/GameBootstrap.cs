using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Rendering;
using HexWars.Engine;

namespace HexWars.Presentation
{
    /// <summary>
    /// Wires the engine to the scene: builds a new game (config + seeded board + two players),
    /// sets up soft ambient + a soft-shadow light + the starfield skybox, and renders board + units.
    /// Map-generation parameters are configurable here. Holds the live <see cref="State"/> — the seam
    /// the input/HUD layer drives. (DemoPieces seeds a couple of visible units/generators until input
    /// drives creation.)
    /// </summary>
    [RequireComponent(typeof(BoardRenderer))]
    public sealed class GameBootstrap : MonoBehaviour
    {
        [Header("Map generation")]
        public int Seed = 7;
        public int Width = 13;
        public int Height = 9;
        public int MaxElevation = 4;
        public int ZoneDepth = 3;
        [Range(0f, 1f)] public float FlatChance = 0.6f;

        [Header("Terrain weights (relative)")]
        public int PlainsWeight = 70;
        public int ForestWeight = 15;
        public int RoughWeight = 10;
        public int WaterWeight = 5;

        [Header("Demo")]
        public bool DemoPieces = true;

        public GameState State { get; private set; }

        /// <summary>Raised after the state changes (new game or applied command) so HUD can refresh.</summary>
        public event System.Action StateChanged;

        void Start() => NewGame();

        public void NewGame()
        {
            SetupEnvironment();

            var config = GameConfig.Default();
            var genConfig = new BoardGenConfig(Width, Height, MaxElevation, ZoneDepth, FlatChance,
                                               PlainsWeight, ForestWeight, RoughWeight, WaterWeight);
            var board = new RandomBoardGenerator(genConfig).Generate(Seed);

            int nextId = 1;
            var p0 = BuildPlayer(board, PlayerId.Player0, ref nextId);
            var p1 = BuildPlayer(board, PlayerId.Player1, ref nextId);
            State = new GameState(board, config, new[] { p0, p1 }, PlayerId.Player0, 1, nextId);

            var renderer = GetComponent<BoardRenderer>();
            renderer.Render(board);
            renderer.RenderEntities(State);

            var rig = FindAnyObjectByType<CameraRig>();
            if (rig != null) rig.Frame(); // fit the camera once the board exists

            StateChanged?.Invoke();
        }

        /// <summary>Apply a command through the engine; on success update state, re-render, notify.</summary>
        public bool TryApply(Command cmd)
        {
            var result = GameEngine.Apply(State, cmd);
            if (!result.Success)
            {
                Debug.Log($"[HexWars] {cmd.GetType().Name} rejected: {result.Reason}");
                return false;
            }
            State = result.NewState;
            GetComponent<BoardRenderer>().RenderEntities(State);
            StateChanged?.Invoke();
            return true;
        }

        PlayerState BuildPlayer(Board board, PlayerId id, ref int nextId)
        {
            if (!DemoPieces)
                return new PlayerState(id, 0);

            var flatZone = board.DeploymentZone(id)
                .Where(c => board.TileAt(c).Elevation == 0)
                .OrderBy(c => c.Q).ThenBy(c => c.R)
                .ToList();

            // a few distinct builds so the role icons + size-by-points are visible
            var demos = new[]
            {
                new UnitStats(health: 7, damage: 2, defense: 2, movement: 3, verticalMovement: 2, range: 1, rangeArc: 1, vision: 2, visionArc: 1), // Brute
                new UnitStats(health: 2, damage: 6, defense: 0, movement: 3, verticalMovement: 2, range: 2, rangeArc: 1, vision: 3, visionArc: 1), // Striker
                new UnitStats(health: 2, damage: 2, defense: 0, movement: 2, verticalMovement: 2, range: 6, rangeArc: 1, vision: 4, visionArc: 1), // Sniper
                new UnitStats(health: 2, damage: 0, defense: 0, movement: 4, verticalMovement: 3, range: 0, rangeArc: 0, vision: 7, visionArc: 2), // Spotter
            };

            // start with an army (the only resource); no generators, no starting points —
            // the only way to earn points (for reinforcements) is bounty from kills
            var units = new List<Unit>();
            int placed = 0;
            for (; placed < demos.Length && placed < flatZone.Count; placed++)
                units.Add(new Unit(nextId++, id, demos[placed], flatZone[placed], 0));

            return new PlayerState(id, 0, unitsOnBoard: units);
        }

        static Cubemap _reflection;

        void SetupEnvironment()
        {
            RenderSettings.ambientMode = AmbientMode.Flat;
            RenderSettings.ambientLight = new Color(0.40f, 0.43f, 0.52f);
            RenderSettings.skybox = StarfieldSkybox();                 // dark starfield stays the visible background
            RenderSettings.defaultReflectionMode = UnityEngine.Rendering.DefaultReflectionMode.Custom;
            RenderSettings.customReflectionTexture = BrightReflection(); // ...but metal reflects an even bright env
            RenderSettings.reflectionIntensity = 1f;
            DynamicGI.UpdateEnvironment();

            // a stray fill from an earlier setup would re-add a second hotspot — remove it
            var stray = GameObject.Find("FillLight");
            if (stray != null) { if (Application.isPlaying) Destroy(stray); else DestroyImmediate(stray); }

            // one gentle light: brightness/shine comes from the even reflection, not light hotspots
            EnsureLight("KeyLight", new Color(1f, 0.98f, 0.94f), 0.9f, Quaternion.Euler(45f, -40f, 0f), LightShadows.Soft, 0.35f);
        }

        static Cubemap BrightReflection()
        {
            if (_reflection != null) return _reflection;
            const int s = 32;
            var cm = new Cubemap(s, TextureFormat.RGBA32, false);
            var tint = new Color(0.98f, 0.98f, 1f);
            for (int f = 0; f < 6; f++)
            {
                var face = (CubemapFace)f;
                // per-face base brightness gives the 6 hex facets directional variation = metallic sheen
                float b = face switch
                {
                    CubemapFace.PositiveY => 1.0f,   // up: bright
                    CubemapFace.NegativeY => 0.18f,  // down: dark
                    CubemapFace.PositiveX => 0.92f,
                    CubemapFace.NegativeX => 0.30f,
                    CubemapFace.PositiveZ => 0.70f,
                    CubemapFace.NegativeZ => 0.48f,
                    _ => 0.6f,
                };
                var cols = new Color[s * s];
                for (int y = 0; y < s; y++)
                    for (int x = 0; x < s; x++)
                    {
                        float v = Mathf.Clamp01(b * (0.8f + 0.2f * (y / (float)(s - 1))));
                        cols[y * s + x] = new Color(tint.r * v, tint.g * v, tint.b * v);
                    }
                cm.SetPixels(cols, face);
            }
            cm.Apply();
            _reflection = cm;
            return cm;
        }

        static void EnsureLight(string name, Color color, float intensity, Quaternion rot, LightShadows shadows, float shadowStrength)
        {
            var go = GameObject.Find(name);
            if (go == null) go = new GameObject(name);
            var l = go.GetComponent<Light>();
            if (l == null) l = go.AddComponent<Light>();
            l.type = LightType.Directional;
            l.color = color;
            l.intensity = intensity;
            l.shadows = shadows;
            l.shadowStrength = shadowStrength;
            go.transform.rotation = rot;
        }

        static Material StarfieldSkybox()
        {
            var tex = new Texture2D(1024, 512, TextureFormat.RGB24, false);
            var px = new Color32[1024 * 512];
            var space = new Color32(6, 7, 17, 255);
            for (int i = 0; i < px.Length; i++) px[i] = space;
            var rng = new System.Random(3);
            for (int i = 0; i < 750; i++)
            {
                int x = rng.Next(1024), y = rng.Next(512);
                byte b = (byte)rng.Next(120, 256);
                px[y * 1024 + x] = new Color32(b, b, b, 255);
            }
            tex.SetPixels32(px);
            tex.Apply();

            var sky = new Material(Shader.Find("Skybox/Panoramic"));
            if (sky.HasProperty("_MainTex")) sky.SetTexture("_MainTex", tex);
            if (sky.HasProperty("_Exposure")) sky.SetFloat("_Exposure", 1f);
            return sky;
        }
    }
}
