using HexWars.Engine;
using UnityEngine;
using UnityEngine.UI;

namespace HexWars.Presentation
{
    /// <summary>Defer portrait work until its UI is visible; hidden designer panels cost no startup renders.</summary>
    [RequireComponent(typeof(RawImage))]
    public sealed class UnitPortrait : MonoBehaviour
    {
        RawImage _image;
        int _index;
        bool _detailed;
        public PlayerId Owner { get; private set; }

        public void SetArt(int index, bool detailed, PlayerId owner = PlayerId.Player0)
        {
            _image = GetComponent<RawImage>();
            _index = Mathf.Clamp(index, 0, 7);
            _detailed = detailed;
            Owner = owner;
            _image.texture = null;
        }

        void LateUpdate()
        {
            // Panel visibility is applied in Update before any portrait can request GPU work.
            if (_image == null || !_image.isActiveAndEnabled || _image.texture != null) return;
            if (GraphitePieces.TryGetPortrait(_index, _detailed, out var portrait, Owner))
                _image.texture = portrait;
        }
    }
}
