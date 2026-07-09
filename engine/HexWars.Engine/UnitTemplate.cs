using System.Text;

namespace HexWars.Engine
{
    /// <summary>
    /// A reusable barracks blueprint: a name (shown in the tooltip/UI) plus the purchased stat line.
    /// Deploying a template clones <see cref="Stats"/> onto a new <see cref="Unit"/> and copies
    /// <see cref="Name"/> onto it — <see cref="Unit.DisplayName"/> falls back to the dominant role
    /// when Name is empty. Immutable.
    /// </summary>
    public readonly struct UnitTemplate
    {
        public readonly string Name;
        public readonly UnitStats Stats;

        public UnitTemplate(string name, UnitStats stats)
        {
            Name = name;
            Stats = stats;
        }

        /// <summary>Sanitize a raw (possibly null, possibly attacker-supplied) name at the engine
        /// boundary: trim, keep only <c>[A-Za-z0-9 -']</c>, cap at 20 characters, then trim again.
        /// Null/empty/fully stripped input becomes "" — callers fall back to the dominant-role label
        /// for display. The second trim matters because filtering can UNCOVER edge whitespace the
        /// first trim couldn't see yet — e.g. "♥ foo" has no leading/trailing whitespace before
        /// filtering, but stripping the disallowed '♥' leaves " foo" with a leading space. Underscore
        /// is deliberately excluded: the wire encoding maps spaces↔underscores
        /// (CommandWire.EncodeName/DecodeName), so allowing a literal '_' in a name would corrupt
        /// it on round-trip ("A_B" would come back as "A B").</summary>
        public static string Sanitize(string? raw)
        {
            if (string.IsNullOrEmpty(raw)) return "";
            var sb = new StringBuilder(20);
            foreach (char ch in raw.Trim())
            {
                if (sb.Length == 20) break;
                bool allowed = (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9')
                               || ch == ' ' || ch == '-' || ch == '\'';
                if (allowed) sb.Append(ch);
            }
            return sb.ToString().Trim();
        }
    }
}
