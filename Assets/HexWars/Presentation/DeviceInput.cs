using System;
using UnityEngine;

namespace HexWars.Presentation
{
    /// <summary>Gates every direct <c>Keyboard.current</c>/<c>Mouse.current</c> poll in Presentation
    /// behind application focus. Unity's Input System keeps updating device state (isPressed,
    /// wasPressedThisFrame, deltas) regardless of OS focus, so without this gate the game keeps
    /// reacting to keystrokes/clicks aimed at another, unfocused window/app.
    /// <see cref="FocusProbe"/> is the test seam: production code never reassigns it, and it defaults
    /// to the real <see cref="Application.isFocused"/>; EditMode tests substitute a fake to drive the
    /// unfocused branch without needing an actual OS focus change.</summary>
    public static class DeviceInput
    {
        public static Func<bool> FocusProbe = () => Application.isFocused;

        /// <summary>True when device input should be read this frame/poll.</summary>
        public static bool Allowed => FocusProbe();
    }
}
