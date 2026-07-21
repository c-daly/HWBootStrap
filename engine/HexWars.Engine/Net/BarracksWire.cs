using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace HexWars.Engine
{
    /// <summary>
    /// Dependency-free wire format for a player's starting barracks. The first line is the format
    /// version; each following line contains a UTF-8 Base64 name and all nine invariant integer stats.
    /// </summary>
    public static class BarracksWire
    {
        public const int MaximumPayloadBytes = 32 * 1024;

        private const string Version = "V1";
        private static readonly UTF8Encoding StrictUtf8 = new UTF8Encoding(false, true);

        public static string Write(IReadOnlyList<UnitTemplate> templates)
        {
            var normalized = BarracksCatalog.Normalize(templates ?? Array.Empty<UnitTemplate>());
            var result = new StringBuilder(Version);
            foreach (var template in normalized)
            {
                result.Append('\n');
                result.Append(Convert.ToBase64String(Encoding.UTF8.GetBytes(template.Name)));
                AppendStat(result, template.Stats.Health);
                AppendStat(result, template.Stats.Damage);
                AppendStat(result, template.Stats.Defense);
                AppendStat(result, template.Stats.Movement);
                AppendStat(result, template.Stats.VerticalMovement);
                AppendStat(result, template.Stats.Range);
                AppendStat(result, template.Stats.RangeArc);
                AppendStat(result, template.Stats.Vision);
                AppendStat(result, template.Stats.VisionArc);
            }

            string payload = result.ToString();
            if (Encoding.UTF8.GetByteCount(payload) > MaximumPayloadBytes)
                throw new FormatException("barracks payload exceeds 32 KiB");
            return payload;
        }

        public static IReadOnlyList<UnitTemplate> Read(string payload)
        {
            if (payload == null || Encoding.UTF8.GetByteCount(payload) > MaximumPayloadBytes)
                throw new FormatException("barracks payload exceeds 32 KiB");

            string[] lines = payload.Split(new[] { '\n' }, StringSplitOptions.None);
            if (lines.Length == 0 || lines[0] != Version)
                throw new FormatException("unsupported barracks payload version");

            var parsed = new List<UnitTemplate>(Math.Min(lines.Length - 1, BarracksCatalog.ProtocolMaximumTemplates));
            for (int lineIndex = 1; lineIndex < lines.Length; lineIndex++)
            {
                string[] fields = lines[lineIndex].Split('|');
                if (fields.Length != 10)
                    throw new FormatException("malformed barracks record");

                string name;
                try
                {
                    name = StrictUtf8.GetString(Convert.FromBase64String(fields[0]));
                }
                catch (Exception ex) when (ex is FormatException || ex is DecoderFallbackException)
                {
                    throw new FormatException("malformed barracks template name", ex);
                }

                var stats = new int[9];
                for (int statIndex = 0; statIndex < stats.Length; statIndex++)
                {
                    if (!int.TryParse(fields[statIndex + 1], NumberStyles.Integer, CultureInfo.InvariantCulture, out stats[statIndex]))
                        throw new FormatException("malformed barracks template stat");
                }

                parsed.Add(new UnitTemplate(name, new UnitStats(
                    stats[0], stats[1], stats[2], stats[3], stats[4],
                    stats[5], stats[6], stats[7], stats[8])));
            }

            return BarracksCatalog.Normalize(parsed);
        }

        private static void AppendStat(StringBuilder target, int value)
        {
            target.Append('|');
            target.Append(value.ToString(CultureInfo.InvariantCulture));
        }
    }
}
