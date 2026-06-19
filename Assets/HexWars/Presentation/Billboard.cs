using UnityEngine;

namespace HexWars.Presentation
{
    /// <summary>Keeps an object turned to face the camera (used for HP bars so they read upright
    /// from the isometric angle instead of lying flat on the board).</summary>
    [ExecuteAlways]
    public sealed class Billboard : MonoBehaviour
    {
        void LateUpdate()
        {
            var cam = Camera.main;
            if (cam == null) cam = FindAnyObjectByType<Camera>();
            if (cam != null) transform.rotation = cam.transform.rotation;
        }
    }
}
