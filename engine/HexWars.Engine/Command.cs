namespace HexWars.Engine
{
    /// <summary>
    /// A player's intended action — the ONLY input to <see cref="GameEngine.Apply"/>, and the future
    /// network wire-format / AI action output. Concrete commands are added as their handlers are built.
    /// </summary>
    public abstract record Command(PlayerId Issuer);

    /// <summary>Design and pay for a unit; it goes to the issuer's reserve (off-board).</summary>
    public sealed record CreateUnit(PlayerId Issuer, UnitStats Stats) : Command(Issuer);

    /// <summary>Pay for and place an income generator on a hex in the issuer's deployment zone.</summary>
    public sealed record DeployGenerator(PlayerId Issuer, HexCoord Cell) : Command(Issuer);

    /// <summary>Deploy a clone of a barracks template (by index) onto a hex in the issuer's deployment
    /// zone, paying the deploy cost. The template is reusable — it is NOT consumed.</summary>
    public sealed record DeployUnit(PlayerId Issuer, int TemplateIndex, HexCoord Cell) : Command(Issuer);

    /// <summary>Move one of the issuer's units to a reachable destination (once per turn).</summary>
    public sealed record MoveUnit(PlayerId Issuer, int UnitId, HexCoord Dest) : Command(Issuer);

    /// <summary>Attack an enemy unit or generator (by id) with one of the issuer's units (once per turn).</summary>
    public sealed record AttackUnit(PlayerId Issuer, int AttackerId, int TargetId) : Command(Issuer);

    /// <summary>Take control of the hex the issuer's unit stands on, paying the capture cost. Control is
    /// persistent (kept after the unit leaves) until an enemy unit recaptures it.</summary>
    public sealed record CaptureHex(PlayerId Issuer, HexCoord Cell) : Command(Issuer);

    /// <summary>End the issuer's turn: pass to the opponent, who is credited generator income.</summary>
    public sealed record EndTurn(PlayerId Issuer) : Command(Issuer);
}
