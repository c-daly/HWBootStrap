using UnityEngine;
using UnityEngine.Rendering;
using HexWars.Engine;

namespace HexWars.Presentation
{
    /// <summary>
    /// Wires the engine to the scene: builds a new game (config + seeded board + two players),
    /// sets up the soft ambient + soft-shadow light + starfield skybox, and renders the board.
    /// Holds the live <see cref="State"/> — the seam the input/HUD layer will drive next.
    /// </summary>
    [RequireComponent(typeof(BoardRenderer))]
    public sealed class GameBootstrap : MonoBehaviour
    {
        [Header("Map generation")]
        public int Seed = 7;
        public int Width = 9;
        public int Height = 7;
        public int MaxElevation = 4;
        public int ZoneDepth = 2;
        [Range(0f, 1f)] public float FlatChance = 0.6f;

        [Header("Terrain weights (relative)")]
        public int PlainsWeight = 70;
        public int ForestWeight = 15;
        public int RoughWeight = 10;
        public int WaterWeight = 5;

        public GameState State { get; private set; }

        void Start() => NewGame();

        public void NewGame()
        {
            SetupEnvironment();

            var config = GameConfig.Default();
            var genConfig = new BoardGenConfig(Width, Height, MaxElevation, ZoneDepth, FlatChance,
                                               PlainsWeight, ForestWeight, RoughWeight, WaterWeight);
            var board = new RandomBoardGenerator(genConfig).Generate(Seed);
            var players = new[]
            {
                new PlayerState(PlayerId.Player0, config.StartingPoints),
                new PlayerState(PlayerId.Player1, config.StartingPoints),
            };
            State = new GameState(board, config, players, PlayerId.Player0, 1, 1);

            GetComponent<BoardRenderer>().Render(board);
        }

        void SetupEnvironment()
        {
            RenderSettings.ambientMode = AmbientMode.Flat;
            RenderSettings.ambientLight = new Color(0.52f, 0.54f, 0.62f);
            RenderSettings.skybox = StarfieldSkybox();
            DynamicGI.UpdateEnvironment();

            var existing = GameObject.Find("KeyLight");
            if (existing == null)
            {
                var go = new GameObject("KeyLight");
                var l = go.AddComponent<Light>();
                l.type = LightType.Directional;
                l.intensity = 0.85f;
                l.color = new Color(1f, 0.98f, 0.92f);
                l.shadows = LightShadows.Soft;
                l.shadowStrength = 0.4f;
                go.transform.rotation = Quaternion.Euler(38f, -42f, 0f);
            }
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
