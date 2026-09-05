namespace HexWars.NetServer.Configuration
{
    /// <summary>
    /// The wire protocols this build of the match host can actually speak.
    ///
    /// <c>MATCH_PROTOCOL_VERSION</c> used to be any positive integer, which made it a way to configure a
    /// server that could not work: the number is written into every match row and then compared against the
    /// number a later host is carrying, so a typo does not fail at startup - it fails months later, as every
    /// stored match becoming unrecoverable on the host that wrote them. A number this build has no code for
    /// is a deployment mistake, and the only honest moment to say so is before the process serves anything.
    ///
    /// It is a set rather than a single number because a protocol change that stays backwards compatible
    /// wants both numbers accepted for as long as older rows exist. Add the new one here, keep the old one
    /// until the rows are gone.
    /// </summary>
    public static class ProtocolContract
    {
        /// <summary>The version a fresh match is written with.</summary>
        public const int Version = 2;

        /// <summary>Every version this build will host, the current one included.</summary>
        public static readonly IReadOnlySet<int> SupportedVersions = new HashSet<int> { Version };

        /// <summary>The supported set as it appears in an error an operator has to act on.</summary>
        public static string SupportedList => string.Join(", ", SupportedVersions.Order());
    }
}
