using UnityEngine;
using UnityEngine.InputSystem;

namespace HexWars.Presentation
{
    /// <summary>
    /// Angled camera over the board. Frames the whole board on start, then supports pan (WASD),
    /// orbit (Q/E), and zoom (scroll) via the new Input System. Put on the Main Camera.
    /// </summary>
    [RequireComponent(typeof(Camera))]
    public sealed class CameraRig : MonoBehaviour
    {
        public Transform Target;             // BoardRenderer transform; auto-found if null
        public float Pitch = 45f;
        public float Yaw = -35f;
        public float Distance = 20f;
        public float PanSpeed = 10f;
        public float RotateSpeed = 70f;
        public float ZoomSpeed = 40f;

        Vector3 _focus = Vector3.zero;

        void Start() => Frame();

        /// <summary>Centre and fit the camera to the board's current bounds.</summary>
        public void Frame()
        {
            var t = Target != null ? Target : ResolveTarget();
            if (t != null)
            {
                var b = ComputeBounds(t);
                _focus = b.center;
                Distance = Mathf.Max(b.extents.magnitude * 2.2f, 5f);
            }
            Apply();
        }

        void Update()
        {
            float dt = Time.unscaledDeltaTime;

            var kb = Keyboard.current;
            if (kb != null)
            {
                Vector3 fwd = Vector3.ProjectOnPlane(transform.forward, Vector3.up).normalized;
                Vector3 right = transform.right;
                if (kb.wKey.isPressed || kb.upArrowKey.isPressed) _focus += fwd * PanSpeed * dt;
                if (kb.sKey.isPressed || kb.downArrowKey.isPressed) _focus -= fwd * PanSpeed * dt;
                if (kb.dKey.isPressed || kb.rightArrowKey.isPressed) _focus += right * PanSpeed * dt;
                if (kb.aKey.isPressed || kb.leftArrowKey.isPressed) _focus -= right * PanSpeed * dt;
                if (kb.qKey.isPressed) Yaw -= RotateSpeed * dt;
                if (kb.eKey.isPressed) Yaw += RotateSpeed * dt;
            }

            var mouse = Mouse.current;
            if (mouse != null)
            {
                float scroll = mouse.scroll.ReadValue().y;
                if (Mathf.Abs(scroll) > 0.01f)
                    Distance = Mathf.Clamp(Distance - scroll * ZoomSpeed * dt, 4f, 90f);
            }

            Apply();
        }

        void Apply()
        {
            var rot = Quaternion.Euler(Pitch, Yaw, 0f);
            transform.rotation = rot;
            transform.position = _focus - rot * Vector3.forward * Distance;
        }

        Transform ResolveTarget()
        {
            var br = FindAnyObjectByType<BoardRenderer>();
            return br != null ? br.transform : null;
        }

        static Bounds ComputeBounds(Transform t)
        {
            var renderers = t.GetComponentsInChildren<Renderer>();
            if (renderers.Length == 0) return new Bounds(t.position, Vector3.one);
            var b = renderers[0].bounds;
            for (int i = 1; i < renderers.Length; i++) b.Encapsulate(renderers[i].bounds);
            return b;
        }
    }
}
