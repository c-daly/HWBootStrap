using UnityEngine;

namespace HexWars.Presentation
{
    /// <summary>Placeholder until the title-screen task lands: Reopen falls back to the Host form.</summary>
    public sealed class TitleScreen : MonoBehaviour
    {
        /// <summary>Always (re)opens the Host form. Callers invoke this right after the current
        /// SetupForm's own <c>Close()</c> (Back / Cancel), so a same-frame existence guard here would
        /// race Unity's deferred <c>Destroy</c> — <c>GetComponent&lt;SetupForm&gt;()</c> still returns
        /// the dying instance as non-null until the frame ends, so the guard would never fire and no
        /// replacement form would ever open. <see cref="SetupForm.Open"/> already closes any existing
        /// form itself before adding the new one, so no guard is needed here.</summary>
        public static void Reopen(GameBootstrap game) => SetupForm.Open(game, SetupForm.SetupMode.Host);
    }
}
