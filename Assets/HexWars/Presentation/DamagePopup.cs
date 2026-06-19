using UnityEngine;

namespace HexWars.Presentation
{
    /// <summary>A floating combat number that rises, faces the camera, and fades out, then destroys
    /// itself. Used for damage dealt on a hit.</summary>
    public sealed class DamagePopup : MonoBehaviour
    {
        public float Lifetime = 1.0f;
        float _t;
        TextMesh _tm;

        public static void Spawn(Vector3 pos, string text, Color color)
        {
            var go = new GameObject("DamagePopup");
            go.transform.position = pos;
            var tm = go.AddComponent<TextMesh>();
            tm.text = text;
            tm.fontSize = 90;
            tm.characterSize = 0.12f;
            tm.anchor = TextAnchor.MiddleCenter;
            tm.alignment = TextAlignment.Center;
            tm.fontStyle = FontStyle.Bold;
            tm.color = color;
            go.AddComponent<DamagePopup>();
        }

        void Awake() => _tm = GetComponent<TextMesh>();

        void Update()
        {
            _t += Time.deltaTime;
            float p = _t / Lifetime;
            if (p >= 1f) { Destroy(gameObject); return; }

            transform.position += Vector3.up * (1.6f * Time.deltaTime);
            var cam = Camera.main;
            if (cam != null)
                transform.rotation = Quaternion.LookRotation(transform.position - cam.transform.position);

            var c = _tm.color;
            c.a = 1f - p;
            _tm.color = c;
        }
    }
}
