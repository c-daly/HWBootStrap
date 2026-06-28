using System;
using System.Globalization;

namespace HexWars.Engine
{
    /// <summary>
    /// Dependency-free, drift-free wire-format for a single <see cref="Command"/>: one line of
    /// space-separated tokens. The SAME code serializes on the authoritative server and deserializes on
    /// each client, so a relayed move reconstructs identically everywhere. Covers every command type
    /// (the format <see cref="ReplayFile"/> pioneered for replays, completed with the territory commands).
    /// </summary>
    public static class CommandWire
    {
        public static string Write(Command c)
        {
            switch (c)
            {
                case MoveUnit m:        return $"M {(int)m.Issuer} {m.UnitId} {m.Dest.Q} {m.Dest.R}";
                case AttackUnit a:      return $"A {(int)a.Issuer} {a.AttackerId} {a.TargetId}";
                case EndTurn e:         return $"E {(int)e.Issuer}";
                case CreateUnit cu:     return $"C {(int)cu.Issuer} {WriteStats(cu.Stats)}";
                case DeployUnit d:      return $"D {(int)d.Issuer} {d.TemplateIndex} {d.Cell.Q} {d.Cell.R}";
                case DeployGenerator g: return $"N {(int)g.Issuer} {g.Cell.Q} {g.Cell.R}";
                case CaptureHex h:      return $"H {(int)h.Issuer} {h.Cell.Q} {h.Cell.R}";
                case BuildGenerator b:  return $"B {(int)b.Issuer} {b.Cell.Q} {b.Cell.R}";
                default: throw new FormatException("unknown command " + c.GetType().Name);
            }
        }

        public static Command Read(string line)
        {
            var p = line.Split(' ');
            var issuer = (PlayerId)I(p[1]);
            switch (p[0])
            {
                case "M": return new MoveUnit(issuer, I(p[2]), new HexCoord(I(p[3]), I(p[4])));
                case "A": return new AttackUnit(issuer, I(p[2]), I(p[3]));
                case "E": return new EndTurn(issuer);
                case "C": return new CreateUnit(issuer, ReadStats(p, 2));
                case "D": return new DeployUnit(issuer, I(p[2]), new HexCoord(I(p[3]), I(p[4])));
                case "N": return new DeployGenerator(issuer, new HexCoord(I(p[2]), I(p[3])));
                case "H": return new CaptureHex(issuer, new HexCoord(I(p[2]), I(p[3])));
                case "B": return new BuildGenerator(issuer, new HexCoord(I(p[2]), I(p[3])));
                default: throw new FormatException("unknown command token " + p[0]);
            }
        }

        internal static string WriteStats(UnitStats s) =>
            $"{s.Health} {s.Damage} {s.Defense} {s.Movement} {s.VerticalMovement} {s.Range} {s.RangeArc} {s.Vision} {s.VisionArc}";

        internal static UnitStats ReadStats(string[] p, int o) =>
            new UnitStats(I(p[o]), I(p[o + 1]), I(p[o + 2]), I(p[o + 3]), I(p[o + 4]),
                          I(p[o + 5]), I(p[o + 6]), I(p[o + 7]), I(p[o + 8]));

        private static int I(string s) => int.Parse(s, CultureInfo.InvariantCulture);
    }
}
