namespace HexWars.Engine
{
    /// <summary>Why <see cref="GameEngine.Apply"/> refused a command (Success = false).</summary>
    public enum RejectionReason
    {
        None = 0,
        NotYourTurn,
        GameAlreadyOver,
        IllegalForPolicy,
        InsufficientPoints,
        InvalidStats,
        TemplateNotFound,
        UnitNotFound,
        GeneratorNotFound,
        TileNotFound,
        TileOccupied,
        TileImpassable,
        OutsideDeploymentZone,
        OutOfMovementRange,
        UnitAlreadyMoved,
        UnitAlreadyAttacked,
        TargetNotEnemy,
        TargetNotInRange,
        TargetNotVisible,
        LineOfSightBlocked,
        NoUnitOnHex,
        AlreadyControlled,
        HexNotControlled,
        MustClaimFirst,
    }
}
