namespace HexWars.Engine
{
    /// <summary>A parsed wire message: a TYPE and its (possibly multi-line, possibly empty) payload.</summary>
    public readonly struct NetMessage
    {
        public readonly string Type;
        public readonly string Payload;
        public NetMessage(string type, string payload) { Type = type; Payload = payload; }
    }

    /// <summary>
    /// The tiny envelope both ends speak over the WebSocket: "TYPE payload", one WebSocket message each
    /// (so a payload may be multi-line — e.g. the START start-state dump). Typed builders + a single
    /// <see cref="Parse"/> keep server and client in lockstep, the same way <see cref="CommandWire"/>
    /// does for commands.
    /// </summary>
    public static class NetProtocol
    {
        // ---- server -> client ----
        /// <summary>Tell a client which seat it holds.</summary>
        public static string Seat(PlayerId seat) => "SEAT " + (int)seat;
        /// <summary>The room is full; the connection is a spectator/turned away.</summary>
        public const string SeatFull = "SEAT FULL";
        /// <summary>A CMD payload that failed to parse (CommandWire.TryRead returned false) — sent only
        /// to the issuer, never broadcast.</summary>
        public const string Malformed = "REJECT Malformed";
        /// <summary>The authoritative start state (a <see cref="ReplayFile"/> dump), sent once both seats are present.</summary>
        public static string Start(string startStateText) => "START " + startStateText;
        /// <summary>Ask a waiting client for its process-lifetime barracks catalog.</summary>
        public const string CatalogRequest = "CATALOG?";
        /// <summary>A validated command for every client to apply locally.</summary>
        public static string Apply(Command c) => "APPLY " + CommandWire.Write(c);
        /// <summary>Tell the issuer their command was rejected, and why.</summary>
        public static string Reject(RejectionReason reason) => "REJECT " + reason;

        // ---- client -> server ----
        /// <summary>A client's attempted command.</summary>
        public static string Cmd(Command c) => "CMD " + CommandWire.Write(c);
        /// <summary>A client's normalized starting barracks catalog.</summary>
        public static string Catalog(string payload) => "CATALOG " + payload;

        // ---- shared ----
        /// <summary>Split a raw message into TYPE + payload on the first space (payload defaults to "").</summary>
        public static NetMessage Parse(string raw)
        {
            int sp = raw.IndexOf(' ');
            return sp < 0
                ? new NetMessage(raw, "")
                : new NetMessage(raw.Substring(0, sp), raw.Substring(sp + 1));
        }
    }
}
