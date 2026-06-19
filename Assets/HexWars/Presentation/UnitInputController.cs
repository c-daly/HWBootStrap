using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.Rendering;

namespace HexWars.Presentation
{
    /// <summary>
    /// Mouse interaction for units: hovering a unit token shows its capability tooltip; left-clicking
    /// selects it (a bright marker floats above the selection and the tooltip stays pinned). Hex/move/
    /// deploy targeting comes with the create-unit/barracks HUD.
    /// </summary>
    [RequireComponent(typeof(UnitTooltip))]
    public sealed class UnitInputController : MonoBehaviour
    {
        UnitTooltip _tooltip;
        UnitView _selected;
        GameObject _marker;

        void Awake()
        {
            _tooltip = GetComponent<UnitTooltip>();
            BuildMarker();
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

            if (mouse.leftButton.wasPressedThisFrame && !IsPointerOverUi())
            {
                _selected = hovered; // click empty space to deselect
                UpdateMarker();
            }

            // gentle bob so the marker is obvious
            if (_marker.activeSelf && _selected != null)
            {
                var p = _selected.transform.position;
                float bob = Mathf.Sin(Time.time * 4f) * 0.08f;
                _marker.transform.position = new Vector3(p.x, p.y + 0.85f + bob, p.z);
            }
        }

        static bool IsPointerOverUi()
        {
            var es = UnityEngine.EventSystems.EventSystem.current;
            return es != null && es.IsPointerOverGameObject();
        }

        void BuildMarker()
        {
            _marker = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            _marker.name = "SelectionMarker";
            var col = _marker.GetComponent<Collider>();
            if (col != null) Destroy(col); // must not block unit raycasts
            _marker.transform.SetParent(transform, false);
            _marker.transform.localScale = Vector3.one * 0.42f;

            var mr = _marker.GetComponent<MeshRenderer>();
            var unlit = Shader.Find("Universal Render Pipeline/Unlit");
            if (unlit == null) unlit = Shader.Find("Unlit/Color");
            var m = new Material(unlit);
            var yellow = new Color(1f, 0.92f, 0.15f);
            if (m.HasProperty("_BaseColor")) m.SetColor("_BaseColor", yellow);
            m.color = yellow;
            mr.sharedMaterial = m;
            mr.shadowCastingMode = ShadowCastingMode.Off;

            _marker.SetActive(false);
        }

        void UpdateMarker()
        {
            if (_selected == null) { _marker.SetActive(false); return; }
            var p = _selected.transform.position;
            _marker.transform.position = new Vector3(p.x, p.y + 0.85f, p.z);
            _marker.SetActive(true);
        }
    }
}
