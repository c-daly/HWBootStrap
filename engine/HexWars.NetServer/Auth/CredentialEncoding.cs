using System.Text.RegularExpressions;

namespace HexWars.NetServer.Auth
{
    /// <summary>
    /// The one place that knows how a join credential is spelled on the wire: 32 random bytes rendered as
    /// unpadded base64url, which is 43 characters that survive a JSON string, a URL and a websocket frame
    /// without escaping.
    ///
    /// Decoding is strict and canonical: a string is a credential only if re-encoding the bytes it decoded
    /// to gives back exactly that string. Base64 leaves two unused bits in the last character of a
    /// 43-character encoding, so a lenient decoder would accept four different strings for every
    /// credential. That is not a way in, the secret is still 256 bits either way, but it would mean the
    /// value the client holds is not the only value that opens the seat, and a bearer token with aliases
    /// is a bad thing to hand to anything that later wants to count, revoke or rate-limit by credential.
    /// </summary>
    public static class CredentialEncoding
    {
        /// <summary>Bytes of entropy in a credential.</summary>
        public const int CredentialBytes = 32;

        /// <summary>Characters in the encoded form of those bytes, with no padding.</summary>
        public const int CredentialCharacters = 43;

        static readonly Regex Base64UrlAlphabet = new(
            "^[A-Za-z0-9_-]+$", RegexOptions.CultureInvariant | RegexOptions.Compiled);

        /// <summary>
        /// Unpadded base64url. Any length is encodable; the credential shape is enforced only on the way
        /// back in, which is the direction facing the network.
        /// </summary>
        public static string ToBase64Url(byte[] raw)
        {
            ArgumentNullException.ThrowIfNull(raw);
            return Convert.ToBase64String(raw)
                .Replace("=", string.Empty)
                .Replace("+", "-")
                .Replace("/", "_");
        }

        /// <summary>
        /// True when the text is the canonical encoding of exactly <see cref="CredentialBytes"/> bytes,
        /// which it then hands back. On false the bytes are empty rather than null, so a caller that
        /// forgets to check the result gets a harmless value instead of a null reference.
        /// </summary>
        public static bool TryFromBase64Url(string text, out byte[] raw)
        {
            raw = Array.Empty<byte>();

            if (text is null || text.Length != CredentialCharacters) return false;
            if (!Base64UrlAlphabet.IsMatch(text)) return false;

            // 43 characters is one short of a base64 quantum, hence exactly one padding character.
            string standard = text.Replace("-", "+").Replace("_", "/") + "=";

            byte[] decoded;
            try
            {
                decoded = Convert.FromBase64String(standard);
            }
            catch (FormatException)
            {
                return false;
            }

            if (decoded.Length != CredentialBytes) return false;
            if (!string.Equals(ToBase64Url(decoded), text, StringComparison.Ordinal)) return false;

            raw = decoded;
            return true;
        }
    }
}
