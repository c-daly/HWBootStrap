using System;
using System.Collections.Generic;
using HexWars.Engine;

namespace HexWars.Presentation
{
    /// <summary>
    /// Process-lifetime barracks catalogs for the two humans who can share this client. The cache is
    /// deliberately not persisted: reloading the app/browser restores the canonical starter set.
    /// </summary>
    public sealed class SessionBarracksCache
    {
        static readonly SessionBarracksCache[] Caches = new SessionBarracksCache[2];

        readonly List<UnitTemplate> _templates;

        SessionBarracksCache()
        {
            _templates = new List<UnitTemplate>(BarracksCatalog.Normalize(BarracksCatalog.DefaultTemplates));
        }

        public static SessionBarracksCache ForLocalPlayer(int localPlayer)
        {
            if (localPlayer < 0 || localPlayer >= Caches.Length)
                throw new ArgumentOutOfRangeException(nameof(localPlayer));
            return Caches[localPlayer] ?? (Caches[localPlayer] = new SessionBarracksCache());
        }

        public bool Add(UnitTemplate template)
        {
            if (_templates.Count >= BarracksCatalog.ProtocolMaximumTemplates)
                return false;

            var normalized = BarracksCatalog.Normalize(new[] { template });
            if (normalized.Count == 0)
                return false;

            var item = normalized[0];
            foreach (var existing in _templates)
                if (BarracksCatalog.Same(existing, item))
                    return false;

            _templates.Add(item);
            return true;
        }

        public bool RemoveAt(int index)
        {
            if (index < 0 || index >= _templates.Count)
                return false;
            _templates.RemoveAt(index);
            return true;
        }

        public IReadOnlyList<UnitTemplate> Snapshot() => new List<UnitTemplate>(_templates);

        public static void ResetForTests()
        {
            for (int i = 0; i < Caches.Length; i++)
                Caches[i] = null;
        }
    }
}
