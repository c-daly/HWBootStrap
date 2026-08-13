namespace HexWars.Presentation
{
    /// <summary>The explicit observation/action contract used by ML training and arena playback.</summary>
    public enum MlEnvironmentContract
    {
        TacticalV1,
        AdaptiveV1,
        TacticalV2,
        TacticalV3,
    }

    public static class MlEnvironmentContracts
    {
        public static string CliValue(MlEnvironmentContract environment)
        {
            switch (environment)
            {
                case MlEnvironmentContract.AdaptiveV1: return "adaptive-v1";
                case MlEnvironmentContract.TacticalV2: return "tactical-v2";
                case MlEnvironmentContract.TacticalV3: return "tactical-v3";
                default: return "tactical-v1";
            }
        }

        public static MlEnvironmentContract Parse(string contractVersion)
        {
            switch (contractVersion)
            {
                case "tactical-v1":
                    return MlEnvironmentContract.TacticalV1;
                case "adaptive-v1":
                    return MlEnvironmentContract.AdaptiveV1;
                case "tactical-v2":
                    return MlEnvironmentContract.TacticalV2;
                case "tactical-v3":
                    return MlEnvironmentContract.TacticalV3;
                default:
                    throw new System.ArgumentException(
                        "Unknown ML environment contract: " + contractVersion,
                        nameof(contractVersion));
            }
        }

        public static string ContractVersion(MlEnvironmentContract environment) => CliValue(environment);
    }

    public readonly struct ModelDuelContractIdentity
    {
        public ModelDuelContractIdentity(
            string environment, string version, string encodingHash,
            string capacityHash = null)
        {
            Environment = environment;
            Version = version;
            EncodingHash = encodingHash;
            CapacityHash = capacityHash;
        }

        public string Environment { get; }
        public string Version { get; }
        public string EncodingHash { get; }
        public string CapacityHash { get; }
    }
}
