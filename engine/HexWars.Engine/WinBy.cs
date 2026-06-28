using System;

namespace HexWars.Engine
{
    /// <summary>Multiply-selectable win conditions. Annihilation and Economy are instant (checked every
    /// command); Score resolves at the round cap. Any subset may be combined.</summary>
    [Flags]
    public enum WinBy
    {
        None = 0,
        Annihilation = 1,
        Economy = 2,
        Score = 4,
    }
}
