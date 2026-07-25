using System;
using System.Collections.Generic;
using System.Linq;
using HexWars.Engine;
using HexWars.Engine.Rl;
using HexWars.Presentation;

namespace HexWars.Presentation.EditorTools.MlLab
{
    /// <summary>Builds the tactical-v2 unit-template roster a local player would bring into a training
    /// scenario: the canonical barracks defaults plus that seat's saved custom designs. Reads
    /// <see cref="SessionBarracksCache"/> but never mutates or persists it — every snapshot is a
    /// detached copy, safe to attach to a scenario without keeping the cache alive.</summary>
    public static class MlTacticalRosterSource
    {
        public static IReadOnlyList<int> AvailablePlayers { get; } =
            Array.AsReadOnly(new[] { 0, 1 });

        public static IReadOnlyList<MlTrainingUnitTemplate> Snapshot(int localPlayer)
        {
            if (!AvailablePlayers.Contains(localPlayer))
                throw new ArgumentOutOfRangeException(nameof(localPlayer));

            var combined = new List<UnitTemplate>(BarracksCatalog.DefaultTemplates);
            IReadOnlyList<UnitTemplate> saved = SessionBarracksCache.ForLocalPlayer(localPlayer).Snapshot();
            foreach (UnitTemplate candidate in saved)
            {
                bool matchesDefault = BarracksCatalog.DefaultTemplates
                    .Any(item => BarracksCatalog.Same(item, candidate));
                if (!matchesDefault) combined.Add(candidate);
            }

            IReadOnlyList<UnitTemplate> normalized = BarracksCatalog.Normalize(combined);
            return normalized.Select(ToTemplate).ToList().AsReadOnly();
        }

        static MlTrainingUnitTemplate ToTemplate(UnitTemplate template) => new MlTrainingUnitTemplate
        {
            Id = TacticalV2TemplateIds.From(template),
            Name = template.Name,
            Stats = new MlTrainingUnitStats
            {
                Health = template.Stats.Health,
                Damage = template.Stats.Damage,
                Defense = template.Stats.Defense,
                Movement = template.Stats.Movement,
                VerticalMovement = template.Stats.VerticalMovement,
                Range = template.Stats.Range,
                RangeArc = template.Stats.RangeArc,
                Vision = template.Stats.Vision,
                VisionArc = template.Stats.VisionArc,
            },
        };
    }
}
