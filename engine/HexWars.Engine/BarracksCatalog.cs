using System;
using System.Collections.Generic;

namespace HexWars.Engine
{
    /// <summary>Canonical starter designs and the normalization shared by session barracks inputs.</summary>
    public static class BarracksCatalog
    {
        public const int ProtocolMaximumTemplates = 64;

        public static readonly IReadOnlyList<UnitTemplate> DefaultTemplates = Array.AsReadOnly(new[]
        {
            new UnitTemplate("Brute",     new UnitStats(7, 2, 2, 3, 2, 1, 1, 2, 1)),
            new UnitTemplate("Striker",   new UnitStats(2, 6, 0, 3, 2, 2, 1, 3, 1)),
            new UnitTemplate("Sniper",    new UnitStats(2, 2, 0, 2, 2, 6, 1, 4, 1)),
            new UnitTemplate("Artillery", new UnitStats(3, 6, 0, 0, 0, 5, 2, 2, 1)),
            new UnitTemplate("Scout",     new UnitStats(2, 0, 0, 4, 3, 0, 0, 7, 2)),
        });

        /// <summary>Sanitizes, validates, deduplicates, and caps a session barracks catalog.</summary>
        public static IReadOnlyList<UnitTemplate> Normalize(IEnumerable<UnitTemplate> source, int maxTemplates = ProtocolMaximumTemplates)
        {
            int limit = Math.Min(Math.Max(maxTemplates, 0), ProtocolMaximumTemplates);
            var result = new List<UnitTemplate>();
            foreach (var raw in source ?? Array.Empty<UnitTemplate>())
            {
                var item = new UnitTemplate(UnitTemplate.Sanitize(raw.Name), raw.Stats);
                if (item.Stats.Health < 1 || result.Exists(x => Same(x, item))) continue;
                if (result.Count == limit) break;
                result.Add(item);
            }
            return result;
        }

        /// <summary>Templates are equal only when their sanitized names and all nine stats match.</summary>
        public static bool Same(UnitTemplate left, UnitTemplate right)
        {
            var a = left.Stats;
            var b = right.Stats;
            return UnitTemplate.Sanitize(left.Name) == UnitTemplate.Sanitize(right.Name)
                && a.Health == b.Health
                && a.Damage == b.Damage
                && a.Defense == b.Defense
                && a.Movement == b.Movement
                && a.VerticalMovement == b.VerticalMovement
                && a.Range == b.Range
                && a.RangeArc == b.RangeArc
                && a.Vision == b.Vision
                && a.VisionArc == b.VisionArc;
        }
    }
}
