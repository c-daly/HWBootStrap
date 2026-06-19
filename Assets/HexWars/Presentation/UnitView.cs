using UnityEngine;
using HexWars.Engine;

namespace HexWars.Presentation
{
    /// <summary>Carries the engine <see cref="Unit"/> on its token so hover/click UI can read its
    /// stats. Set by <see cref="BoardRenderer"/> when it builds the token.</summary>
    public sealed class UnitView : MonoBehaviour
    {
        public Unit Unit;
    }
}
