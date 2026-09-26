namespace HexWars.NetServer.Configuration
{
    /// <summary>Steam partner credentials and endpoint. Bound from the flat STEAM_* environment keys;
    /// never logged and never serialised into the environment report.</summary>
    public sealed class SteamOptions
    {
        public const string DefaultWebApiBaseUrl = "https://partner.steam-api.com";

        /// <summary>SteamID of the application. 0 means "not configured".</summary>
        public uint AppId { get; set; }

        /// <summary>Publisher Web API key. A secret: it must never reach a log line or an HTTP response.</summary>
        public string PublisherWebApiKey { get; set; } = string.Empty;

        public Uri WebApiBaseUrl { get; set; } = new Uri(DefaultWebApiBaseUrl);

        public int RequestTimeoutSeconds { get; set; } = 5;
    }
}
