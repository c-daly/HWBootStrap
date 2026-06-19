using UnityEngine;
using HexWars.Engine;

namespace HexWars.Presentation
{
    /// <summary>Carries a tile's <see cref="HexCoord"/> on its column so click-targeting (deploy,
    /// later move) can resolve which hex was clicked.</summary>
    public sealed class TileView : MonoBehaviour
    {
        public HexCoord Coord;
    }
}
