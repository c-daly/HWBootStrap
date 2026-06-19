using UnityEngine;
using UnityEngine.Rendering;

namespace HexWars.Presentation
{
    /// <summary>Self-animating stylized explosion: a flash sphere that pops, a burst of debris cubes,
    /// and a quick point-light flash. Size scales with <c>scale</c>. Destroys itself when finished.</summary>
    public sealed class ExplosionFx : MonoBehaviour
    {
        public float Duration = 0.6f;

        Color _tint = new Color(1f, 0.55f, 0.12f);
        float _scale = 1f;
        bool _withDebris = true;

        float _t;
        Transform _flash;
        Material _flashMat;
        Light _light;
        Transform[] _debris;
        Vector3[] _vel;

        public static void Spawn(Vector3 pos, Color tint, float scale = 1f, bool debris = true)
        {
            var go = new GameObject("ExplosionFx");
            go.transform.position = pos;
            var fx = go.AddComponent<ExplosionFx>();
            fx._tint = tint;
            fx._scale = scale;
            fx._withDebris = debris;
        }

        void Start()
        {
            var unlit = Shader.Find("Universal Render Pipeline/Unlit");
            if (unlit == null) unlit = Shader.Find("Unlit/Color");

            var s = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            DestroyImmediate(s.GetComponent<Collider>());
            s.transform.SetParent(transform, false);
            s.transform.localScale = Vector3.one * 0.3f * _scale;
            _flash = s.transform;
            _flashMat = new Material(unlit);
            SetColor(_flashMat, new Color(1f, 0.9f, 0.5f));
            var smr = s.GetComponent<MeshRenderer>();
            smr.sharedMaterial = _flashMat;
            smr.shadowCastingMode = ShadowCastingMode.Off;

            var lgo = new GameObject("Flash");
            lgo.transform.SetParent(transform, false);
            _light = lgo.AddComponent<Light>();
            _light.type = LightType.Point;
            _light.color = new Color(1f, 0.6f, 0.25f);
            _light.range = 7f * _scale;
            _light.intensity = 9f * _scale;

            int n = _withDebris ? 8 : 0;
            _debris = new Transform[n];
            _vel = new Vector3[n];
            if (n > 0)
            {
                var debrisMat = new Material(unlit);
                SetColor(debrisMat, _tint);
                for (int i = 0; i < n; i++)
                {
                    var d = GameObject.CreatePrimitive(PrimitiveType.Cube);
                    DestroyImmediate(d.GetComponent<Collider>());
                    d.transform.SetParent(transform, false);
                    d.transform.localScale = Vector3.one * Random.Range(0.12f, 0.24f) * _scale;
                    var mr = d.GetComponent<MeshRenderer>();
                    mr.sharedMaterial = debrisMat;
                    mr.shadowCastingMode = ShadowCastingMode.Off;
                    _debris[i] = d.transform;
                    float ang = i / (float)n * Mathf.PI * 2f;
                    _vel[i] = new Vector3(Mathf.Cos(ang), Random.Range(1.2f, 2.2f), Mathf.Sin(ang)) * Random.Range(2.5f, 4.5f) * _scale;
                }
            }
        }

        void Update()
        {
            _t += Time.deltaTime;
            float p = Mathf.Clamp01(_t / Duration);
            if (p >= 1f) { Destroy(gameObject); return; }

            float peak = 2.4f * _scale;
            float s = p < 0.35f ? Mathf.Lerp(0.3f * _scale, peak, p / 0.35f) : Mathf.Lerp(peak, 0f, (p - 0.35f) / 0.65f);
            _flash.localScale = Vector3.one * s;
            SetColor(_flashMat, Color.Lerp(new Color(1f, 0.95f, 0.6f), _tint, p));
            _light.intensity = Mathf.Lerp(9f * _scale, 0f, p);

            float dt = Time.deltaTime;
            for (int i = 0; i < _debris.Length; i++)
            {
                _vel[i] += Vector3.down * 6f * dt;
                _debris[i].localPosition += _vel[i] * dt;
                _debris[i].Rotate(180f * dt, 140f * dt, 0f);
                _debris[i].localScale = Vector3.one * Mathf.Lerp(0.2f * _scale, 0f, p);
            }
        }

        static void SetColor(Material m, Color c)
        {
            if (m.HasProperty("_BaseColor")) m.SetColor("_BaseColor", c);
            m.color = c;
        }
    }
}
