using UnityEngine;
using UnityEngine.InputSystem;

namespace HexWars.Presentation
{
    /// <summary>
    /// Mouse interaction for units: hovering a unit token shows its capability tooltip; left-clicking
    /// selects it (a ring marks the selection and the tooltip stays pinned). Hex/move/deploy targeting
    /// comes with the create-unit/barracks HUD.
    /// </summary>
    [RequireComponent(typeof(UnitTooltip))]
    public sealed class UnitInputController : MonoBehaviour
    {
        UnitTooltip _tooltip;
        UnitView _selected;
        GameObject _ring;

        void Awake()
        {
            _tooltip = GetComponent<UnitTooltip>();
            BuildRing();
        }

        void Update()
        {
            var mouse = Mouse.current;
            var cam = Camera.main;
            if (mouse == null || cam == null) return;

            Vector2 mp = mouse.position.ReadValue();
            UnitView hovered = null;
            if (Physics.Raycast(cam.ScreenPointToRay(mp), out var hit, 1000f))
                hovered = hit.collider.GetComponentInParent<UnitView>();

            if (hovered != null) _tooltip.Show(hovered.Unit, mp);
            else if (_selected != null) _tooltip.Show(_selected.Unit, mp);
            else _tooltip.Hide();

            if (mouse.leftButton.wasPressedThisFrame)
            {
                _selected = hovered; // click empty space to deselect
                UpdateRing();
            }
        }

        void BuildRing()
        {
            _ring = new GameObject("SelectionRing");
            _ring.transform.SetParent(transform, false);
            _ring.AddComponent<MeshFilter>().sharedMesh = HexMesh.Ring(0.98f, 0.82f);
            var mr = _ring.AddComponent<MeshRenderer>();
            var unlit = Shader.Find("Universal Render Pipeline/Unlit");
            if (unlit == null) unlit = Shader.Find("Unlit/Color");
            var m = new Material(unlit);
            var yellow = new Color(1f, 0.9f, 0.2f);
            if (m.HasProperty("_BaseColor")) m.SetColor("_BaseColor", yellow);
            m.color = yellow;
            mr.sharedMaterial = m;
            mr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            _ring.SetActive(false);
        }

        void UpdateRing()
        {
            if (_selected == null) { _ring.SetActive(false); return; }
            var p = _selected.transform.position;
            _ring.transform.position = new Vector3(p.x, p.y - 0.06f, p.z);
            _ring.SetActive(true);
        }
    }
}
