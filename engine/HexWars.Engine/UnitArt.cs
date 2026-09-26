using System;
using System.Collections.Generic;

namespace HexWars.Engine
{
    /// <summary>Version 1 cosmetic catalog. Empty means role-matched; unknown IDs safely use that default.</summary>
    public static class UnitArt
    {
        public static readonly IReadOnlyList<string> Ids = Array.AsReadOnly(new[]
            { "relay-01", "atlas-01", "edge-01", "bastion-01", "glide-01", "crux-01", "lance-01", "halo-01" });
        public static readonly IReadOnlyList<string> Names = Array.AsReadOnly(new[]
            { "Relay", "Atlas", "Edge", "Bastion", "Glide", "Crux", "Lance", "Halo" });
        public static string Normalize(string? id)
        {
            foreach (var known in Ids) if (known == id) return known;
            return "";
        }
        public static string Resolve(string? id, UnitStats stats)
        {
            string chosen = Normalize(id);
            return chosen.Length > 0 ? chosen : Ids[(int)Roles.Dominant(stats)];
        }
        public static int Index(string id)
        {
            for (int i = 0; i < Ids.Count; i++) if (Ids[i] == id) return i;
            return 0;
        }
    }
}
