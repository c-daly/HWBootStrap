using System;
using System.Collections.Generic;
using System.Linq;
using HexWars.Engine;
using HexWars.Engine.Rl;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class TacticalV2EnvTests
    {
        [Test]
        public void Reset_IsSymmetricAndReproducible()
        {
            TacticalV2Config config = TacticalV2Config.Default();
            config.StartingUnitCount = config.MaxControllableUnits = 8;
            var first = new TacticalV2Env(seed => new GreedyAgent(seed), PlayerId.Player0, config);
            var second = new TacticalV2Env(seed => new GreedyAgent(seed), PlayerId.Player0, config);

            Assert.That(first.Reset(71), Is.EqualTo(second.Reset(71)));
            Assert.That(Signature(first.State, PlayerId.Player0),
                Is.EqualTo(Signature(first.State, PlayerId.Player1)));
        }

        /// <summary>Regression: a scripted opponent (Greedy) decides purely from raw engine legality —
        /// board cells and points, never the RL registry's synthetic per-seat capacity — so nothing
        /// stops it from proposing a DeployUnit once every registry slot already holds a living unit.
        /// Before the fix, TryApply forwarded that command straight to
        /// <see cref="TacticalV2UnitRegistry.RegisterDeployment"/>, which throws when the registry is
        /// full, crashing the whole GymServer process mid-training. A full episode against Greedy, for
        /// every starting seed in this range, must never throw — an over-capacity deploy attempt has to
        /// be treated the same as any other illegal move (silently rejected, turn continues/unsticks).</summary>
        [Test]
        public void LongGameAgainstGreedyOpponent_NeverOverflowsRegistryCapacity()
        {
            TacticalV2Config config = TacticalV2Config.Default();

            for (int seed = 0; seed < 40; seed++)
            {
                var env = new TacticalV2Env(s => new GreedyAgent(s), PlayerId.Player0, config);
                env.Reset(seed);
                for (int step = 0; step < config.MaxSteps; step++)
                {
                    StepResult result = env.Step(0); // learner always ends its own turn immediately
                    if (result.Terminated || result.Truncated) break;
                }
            }
        }

        [Test]
        public void DeployAfterDeath_ReusesReleasedSlotWithChosenTemplateIdentity()
        {
            TacticalV2EnvFixture fixture = TacticalV2EnvFixture.WithReleasedSlot();
            fixture.Environment.Step(fixture.DeployAction(template: 4, fixture.FreeCell));

            Assert.That(fixture.Environment.Slots0.TemplateIndexAt(fixture.ReleasedSlot),
                Is.EqualTo(4));
        }

        /// <summary>A seat's living-unit stat lines, sorted into a canonical order so the comparison
        /// proves the two seats hold equal MULTISETS of army composition — a mirrored start — rather
        /// than merely happening to agree on unit insertion order.</summary>
        private static List<UnitStats> Signature(GameState state, PlayerId seat) =>
            state.Player(seat).UnitsOnBoard
                .Where(u => u.IsAlive)
                .Select(u => u.Stats)
                .OrderBy(s => s.Health).ThenBy(s => s.Damage).ThenBy(s => s.Defense)
                .ThenBy(s => s.Movement).ThenBy(s => s.VerticalMovement)
                .ThenBy(s => s.Range).ThenBy(s => s.RangeArc)
                .ThenBy(s => s.Vision).ThenBy(s => s.VisionArc)
                .ToList();

        /// <summary>Builds a legal tactical-v2 position with one releasable Slots0 slot, an affordable
        /// catalog template at index 4, and a legal deployment cell — without touching env internals.
        /// The catalog has four identical "killer" templates (huge, board-covering range/vision on a
        /// perfectly flat board, so no movement or line-of-sight risk) at indices 0-3, and one distinct,
        /// minimal "cheap" template at index 4. The starting seed is chosen so the sampled two-unit
        /// roster never draws the cheap template (it must stay unused and affordable, not fielded).
        /// Round one then plays out deterministically: Player0's slot-0 unit kills Player1's slot-0 unit
        /// for a bounty (funding the later deploy of the cheap template), and the scripted opponent's
        /// surviving unit kills Player0's slot-0 unit in reply, releasing that slot.</summary>
        private sealed class TacticalV2EnvFixture
        {
            private const int CheapTemplateIndex = 4;

            public TacticalV2Env Environment { get; }
            public int ReleasedSlot { get; }
            public HexCoord FreeCell { get; }

            private TacticalV2EnvFixture(TacticalV2Env environment, int releasedSlot, HexCoord freeCell)
            {
                Environment = environment;
                ReleasedSlot = releasedSlot;
                FreeCell = freeCell;
            }

            /// <summary>The DeployUnit action index for <paramref name="template"/> at <paramref name="cell"/>,
            /// computed through the layout's own offsets — never a hardcoded action integer.</summary>
            public int DeployAction(int template, HexCoord cell) =>
                Environment.Layout.DeployOffset
                + template * Environment.Layout.CellCount
                + Environment.Layout.CellIndex[cell];

            public static TacticalV2EnvFixture WithReleasedSlot()
            {
                TacticalV2Config config = BuildConfig();
                int seed = FindSeedAvoidingTemplate(config, CheapTemplateIndex);

                var env = new TacticalV2Env(seed => new KillerOpponent(), PlayerId.Player0, config);
                env.Reset(seed);

                // Round 1, Player0's turn: attack Player1's slot-0 unit with Player0's slot-0 unit. The
                // killer templates' huge range reaches across the whole board with no movement needed,
                // and the kill's bounty funds the cheap template's deploy cost below.
                HexCoord enemySlot0Cell = env.State.Player(PlayerId.Player1).UnitsOnBoard[0].Cell;
                int attackAction = env.Layout.AttackOffset + 0 * env.Layout.CellCount
                    + env.Layout.CellIndex[enemySlot0Cell];
                env.Step(attackAction);

                // End Player0's turn. The scripted KillerOpponent replies by killing Player0's slot-0
                // unit with its own surviving unit (releasing that slot), then ends its own turn —
                // handing control back to Player0 with a released slot and bounty points banked.
                env.Step(0); // EndTurn

                int releasedSlot = FindReleasedSlot(env.Slots0);
                HexCoord freeCell = FindFreeDeploymentCell(env.State);
                return new TacticalV2EnvFixture(env, releasedSlot, freeCell);
            }

            private static TacticalV2Config BuildConfig()
            {
                var killerStats = new UnitStats(health: 1, damage: 5, defense: 0, movement: 0,
                    verticalMovement: 0, range: 40, rangeArc: 0, vision: 40, visionArc: 0);
                var cheapStats = new UnitStats(health: 1, damage: 0, defense: 0, movement: 0,
                    verticalMovement: 0, range: 0, rangeArc: 0, vision: 0, visionArc: 0);

                var templates = new[]
                {
                    new TacticalV2Template("killer-0", new UnitTemplate("Killer0", killerStats)),
                    new TacticalV2Template("killer-1", new UnitTemplate("Killer1", killerStats)),
                    new TacticalV2Template("killer-2", new UnitTemplate("Killer2", killerStats)),
                    new TacticalV2Template("killer-3", new UnitTemplate("Killer3", killerStats)),
                    new TacticalV2Template("cheap-4", new UnitTemplate("Cheap4", cheapStats)),
                };

                return new TacticalV2Config
                {
                    // flatChance=1 + maxElevation=0 guarantee every tile is elevation 0, so the killer
                    // templates' huge range/vision always has a clear line of sight, board-size-independent.
                    BoardGen = new BoardGenConfig(maxElevation: 0, flatChance: 1.0),
                    Game = GameConfig.Default(biomesEnabled: false),
                    Templates = templates,
                    StartingUnitCount = 2,
                    MaxControllableUnits = 2,
                    MaxSteps = 600,
                    ShapeScale = 0.01f,
                    StepPenalty = 0.005f,
                    ClosingWeight = 0.02f,
                    DrawCreditWeight = 0.25f,
                    PointsWeight = 0.5f,
                    PlacementPolicy = "symmetric-random-v1",
                };
            }

            /// <summary>Finds a seed whose sampled two-unit starting composition never draws
            /// <paramref name="avoidIndex"/>, so the cheap template stays out of the starting army and
            /// keeps its distinct, minimal (non-combat) stat line meaningful for the affordability check.</summary>
            private static int FindSeedAvoidingTemplate(TacticalV2Config config, int avoidIndex)
            {
                string avoidId = config.Templates[avoidIndex].Id;
                for (int seed = 0; seed < 10_000; seed++)
                {
                    bool avoided = true;
                    foreach (TacticalV2Template template in config.SampleStartingArmy(seed))
                        if (template.Id == avoidId) { avoided = false; break; }
                    if (avoided) return seed;
                }
                throw new InvalidOperationException(
                    "no seed found whose starting composition avoids the cheap template");
            }

            private static int FindReleasedSlot(TacticalV2UnitRegistry registry)
            {
                for (int slot = 0; slot < registry.Capacity; slot++)
                    if (registry.UnitIdAt(slot) < 0) return slot;
                throw new InvalidOperationException("fixture setup did not release a Slots0 slot");
            }

            private static HexCoord FindFreeDeploymentCell(GameState state)
            {
                var occupied = new HashSet<HexCoord>();
                foreach (var unit in state.Player(PlayerId.Player0).UnitsOnBoard) occupied.Add(unit.Cell);
                foreach (HexCoord cell in state.Board.DeploymentZone(PlayerId.Player0))
                    if (!occupied.Contains(cell)) return cell;
                throw new InvalidOperationException("fixture setup found no free deployment cell");
            }

            /// <summary>Always attacks with its first not-yet-attacked living unit against the first
            /// living enemy unit, else ends the turn — enough to guarantee exactly one kill per side in
            /// this fixture's small, symmetric, one-shot-kill roster.</summary>
            private sealed class KillerOpponent : IAgent
            {
                public Command Decide(GameState state)
                {
                    PlayerId me = state.ActivePlayer;
                    foreach (var unit in state.Player(me).UnitsOnBoard)
                    {
                        if (!unit.IsAlive || state.AttackedUnitIds.Contains(unit.Id)) continue;
                        foreach (var target in state.Opponent(me).UnitsOnBoard)
                            if (target.IsAlive) return new AttackUnit(me, unit.Id, target.Id);
                    }
                    return new EndTurn(me);
                }
            }
        }
    }
}
