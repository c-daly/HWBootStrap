using System.Globalization;
using System.Net;
using System.Net.Sockets;

namespace HexWars.NetServer.Hosting
{
    /// <summary>
    /// The one answer to "who is asking", shared by everything on this server that counts per caller.
    ///
    /// It has to be shared, and it has to bucket. Every per-address control here - the request rate
    /// limiter, the auth-failure throttle, the per-address socket cap and the open-match quota - is only
    /// as strong as the key it partitions on, so a control that keys on the raw address while another
    /// keys on the prefix is a control with a documented number and no floor underneath it.
    ///
    /// Two normalisations, both of which close a way around every one of those counters at once:
    ///
    /// - An IPv4 address that arrived over IPv6 is written ::ffff:a.b.c.d and would otherwise be a second,
    ///   free bucket for the same client.
    /// - An IPv6 client is not one address. A routed /64 is the smallest unit an ISP hands out, and a
    ///   client holding one can walk through eighteen quintillion addresses inside it, one per request, and
    ///   never meet a per-address cap at all. So the /64 is the unit these counters count.
    ///
    /// The cost of the second one is that clients genuinely sharing a /64 share a budget. That is the same
    /// trade every one of these controls already makes for IPv4 behind NAT, and the budgets are sized for
    /// it.
    /// </summary>
    public static class CallerKey
    {
        /// <summary>The key a connection with no address falls in - under a test server, or behind a proxy
        /// that did not forward one. Named rather than empty so it reads in a log.</summary>
        public const string Unknown = "unknown";

        /// <summary>The IPv6 prefix these counters count. See the type remarks.</summary>
        public const int IPv6PrefixBits = 64;

        /// <summary>The bucket this request falls in. Read after the forwarded-headers middleware, so behind
        /// a trusted proxy this is the client rather than the proxy.</summary>
        public static string From(HttpContext context)
        {
            ArgumentNullException.ThrowIfNull(context);

            return From(context.Connection.RemoteIpAddress);
        }

        public static string From(IPAddress? address)
        {
            if (address is null) return Unknown;

            if (address.IsIPv4MappedToIPv6) address = address.MapToIPv4();

            if (address.AddressFamily != AddressFamily.InterNetworkV6) return address.ToString();

            byte[] bytes = address.GetAddressBytes();
            for (var i = IPv6PrefixBits / 8; i < bytes.Length; i++) bytes[i] = 0;

            return new IPAddress(bytes).ToString()
                + "/" + IPv6PrefixBits.ToString(CultureInfo.InvariantCulture);
        }
    }
}
