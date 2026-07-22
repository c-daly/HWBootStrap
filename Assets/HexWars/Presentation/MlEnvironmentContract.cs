namespace HexWars.Presentation
{
    /// <summary>The explicit observation/action contract used by ML training and arena playback.</summary>
    public enum MlEnvironmentContract
    {
        TacticalV1,
        AdaptiveV1,
    }

    public static class MlEnvironmentContracts
    {
        public static string CliValue(MlEnvironmentContract environment) =>
            environment == MlEnvironmentContract.AdaptiveV1 ? "adaptive-v1" : "tactical-v1";

        public static string ContractVersion(MlEnvironmentContract environment) => CliValue(environment);
    }

    public readonly struct ModelDuelContractIdentity
    {
        public ModelDuelContractIdentity(string environment, string version, string encodingHash)
        {
            Environment = environment;
            Version = version;
            EncodingHash = encodingHash;
        }

        public string Environment { get; }
        public string Version { get; }
        public string EncodingHash { get; }
    }
}
