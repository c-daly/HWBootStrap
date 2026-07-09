namespace HexWars.Presentation
{
    /// <summary>
    /// The nine stat descriptions (design spec §6) — each one mechanic + judgment. Always available
    /// regardless of the Tips toggle (spec: "Reference, not hand-holding"). <c>Caption</c> is a distilled
    /// one-liner used for the Designer's always-visible row captions when Tips is on (Task 12);
    /// <c>Full</c> is the spec's verbatim copy shown in the tap-to-see bubble (this task, via
    /// <see cref="TipBubble"/>). Order matches <see cref="DesignPanel"/>'s stat row order exactly.
    /// </summary>
    public static class StatInfo
    {
        public static readonly (string Key, string Caption, string Full)[] All =
        {
            ("Health",
             "Absorbs damage before dying — buy it to hold ground.",
             "How much damage it absorbs before dying. Buy it for units that must hold ground under fire; a 2-health unit dies to one mistake."),

            ("Damage",
             "Kill speed — enough Damage beats Defense stacking.",
             "Subtracted by the target's Defense; a landed hit always deals at least 1. This is kill speed — enough Damage makes Defense stacking pointless."),

            ("Defense",
             "Cuts incoming damage — strong vs swarms, weak vs one gun.",
             "Subtracted from every hit you take. Against a swarm of weak attackers it multiplies your effective health; against one big gun it's nearly worthless. Read the enemy's army first."),

            ("Movement",
             "Horizontal steps per turn — reach, escape, tempo.",
             "Horizontal steps per turn. Reach, escape, and tempo. Zero is a choice: an emplacement that never moves — position it like it matters, because it will never matter again."),

            ("Vertical Move",
             "Levels climbed per turn; down/level moves are free.",
             "How many levels it can climb per turn (descending and level moves are free). High ground adds damage and reach, so climbers take the positions that win fights."),

            ("Range",
             "How far it shoots; 0 means melee only.",
             "How far it shoots (0 = melee only). Outranging the enemy's answer is free damage; high ground extends it further."),

            ("Range Arc",
             "Levels it can fire upward; can lob over terrain.",
             "How many levels up it can fire — and anything above 0 can lob over blocking terrain (indirect fire). Your army still needs eyes on the target: batteries want spotters."),

            ("Vision",
             "How far it sees; you can only shoot what's seen.",
             "How far it sees. Sight is shared by your whole army, and you can only shoot what somebody sees. Under fog, information is the game — a cheap pair of eyes makes every gun longer."),

            ("Vision Arc",
             "How many levels up it can see.",
             "How many levels up it can see. Cliffs hide things; someone has to look over the edge."),
        };
    }
}
