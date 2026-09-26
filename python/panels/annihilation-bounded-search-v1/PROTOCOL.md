# Annihilation bounded-search v1 protocol

This is an evaluation-only diagnostic ceiling, not a trainable controller and not
a production-AI promotion. The panel was locked after the curriculum gate failed
and after one development-seed reciprocal smoke, but before any 9,000,000-series
planner outcome was observed.

Both search variants use exact `LegalMoves.For` commands and immutable
`GameEngine.Apply` transitions, depth four commands, and at most 512 expansions
per decision. `bounded-search` adds a bounded health-sensitive persistent-pursuit
value. `bounded-search-terminal-only` is the same search with every nonterminal
state valued at zero.

Each controller plays Random on the existing nine-profile locked conversion bank:
ten maps per profile, both candidate seats, 180 games total. All planner traces,
replays, decision expansions, wall time, target identities, selected branches,
and top-three alternatives are retained. Existing Greedy and mixed-PPO results
are paired by profile, map seed, and candidate seat; their locked files are not
rewritten.

The 10,000,000-series confirmation namespace remains unassigned and unconsumed.
No gate or controller parameter may be changed after locked evaluation starts.
