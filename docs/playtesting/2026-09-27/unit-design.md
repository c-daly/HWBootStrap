# Unit-design and role playtest — K=0 whole-army mode — 2026-09-27

**Scope correction:** all 42 matches below use the selectable K=0 whole-army pace. The browser SetupForm instead initializes K=3 (`Assets/HexWars/Presentation/SetupForm.cs:35`) and randomizes its starting seed. These original results do not establish balance at the browser default pace. The separate `unit-design-k3` follow-up repeats the same 42 seed/design cases at K=3. The 12 manual actions here are explicitly constructed K=0 encounters. Planned biome and generator modes were not assessed and this report does not recommend removing those systems.

Release source: `/mnt/c/users/cddal/hexwars/.worktrees/player-polish-20260924`, main `1cd5f69`. Used the parent's fresh Release engine DLL, SHA 256 `459f67ad83cd61576d034ca1f9fc25d3dc3974fa4c25f96d6257fe6d53817312`. No repository edits. This is an engine playtest; I did not operate the browser or evaluate visual/audio presentation.

## Most useful finding

**The 2-to-3 HP breakpoint is a promising first balance experiment.** The default Sniper is 20 points with HP 2/range 6. Moving one point from range to health creates an equally priced HP 3/range 5 Sniper. In 12 synthetic, equal-60-point games over six public board seeds with both army seats, the HP 3 variant won **11/12**:5/6 while in seat 0 and 6/6 while in seat 1, with the same six seeds tested in each seat. This is a strong candidate for human testing, not a proven universally dominant design: it used three copies of each Sniper, no other roles or reserves, and one bounded tactical controller. Range 6 can matter with spotters; high-ground damage can erase the HP 3 advantage.

The causal explanation is exact: on level ground a Sniper's damage 2 kills HP 2 in one hit, but HP 3 takes two. The same applies to Brute damage 2 against these unarmored units. Buying the third health point doubles the number of such shots survived. Solo Sniper vision is only 4, so range 6 is not fully usable without another unit spotting. This makes the health trade especially attractive in unsupported or loosely coordinated play.

## Frozen suite and results

Criteria were written to `criteria.md` before execution. Seeds were `[1,2,3,7,11,19]`; all maps were public GameFactory 9 x 7, elevations 0–2, biomes off, damage floor 1, full-army turns, fog off. All attempted match commands passed `GameEngine.Apply`. Units can legally move and then fire in the same turn; they cannot fire and then move. Capped at 40 rounds; all 42 games ended by annihilation before the cap.

| Test | Army value | Result | Interpretation |
|---|---:|---|---|
| Ordinary default starters, six seeds | 61 vs 61 initially | Seat 0 won 3, Seat 1 won 3; rounds 4–6 | Too small and controller-dependent for a seat-balance claim |
| 3 stock Snipers vs 3 HP 3/range 5 Snipers, both seats | 60 vs 60 | HP 3 variant 11 wins, stock 1 | Strong bounded design candidate |
| 3 Strikers vs 3 balanced fighters, both seats | 60 vs 60 | Balanced 8 wins, Striker 4 | Worth exploring; not a decisive general result |
| 3 Strikers vs 6 cheap fighters, both seats | 60 vs 60 | Striker 11 wins, cheap 1 | This cheap swarm was ineffective; no broad swarm-dominance claim |

Balanced fighter: `HP3,damage3,defense0,move3,climb2,range3,arc1,vision4,visionArc1` (20). Cheap fighter: `HP1,damage2,defense0,move2,climb1,range1,arc1,vision1,visionArc1` (10). Synthetic armies were directly placed into mirrored public starting slots; they are **equal-budget controlled encounters**, not an available default starting-roster menu. Synthetic reserves were empty and bounty points were unused.

Ordinary starter games kept the real default barracks and reinforcements. The controller spent earned bounties at the start of its following turn, preferring mobile combatants and then longer range among affordable units. These reinforcements therefore must not be conflated with the no-reserve synthetic tests.

The controller orders longer-range units first, tries legal reachable firing positions, prioritizes kills and proportional damage, mildly penalizes immediate exposure, and otherwise approaches enemies with a small elevation preference. It has **no opponent move-and-shoot lookahead, coordinated screening, placement optimization or learned skill**. This especially disadvantages the short-range fragile Striker. Code and every move/attack are retained.

## Starter role concern, with the controller limitation made explicit

Across the ordinary six games, **9 of 12 starting Strikers died before their first attack**. Starting Strikers fired only 3 times and dealt 11 total effective damage, compared with 20 starting Sniper shots/35 damage and 35 Brute shots/51 damage. These totals exclude purchased reinforcements. This feels like a useful warning about easy-to-misuse frontline roles, not evidence that expert Striker play cannot work: the controller advances exposed short-range pieces too eagerly.

The damage 6 Striker overkills either HP 2 light unit, while damage 2 already kills them. Against the HP 7/defense 2 Brute, damage 6 becomes 4 and still needs two attacks. Therefore its intended advantage requires coordinated focus fire, good positioning or high-ground interactions; its health gives it very little chance to recover from a first placement mistake. Do not simply reduce damage to buy health without checking the anti-Brute breakpoint: damage 5 versus defense 2 becomes 3, requiring **three** shots against HP 7.

## A representative normal game, inspected action by action

`normal-starters-7`, full trace in `traces.json`:

1. Seat 0 advances Sniper 3 to(2, 2), Striker 2 to(3, 0), Brute 1 to(3, -1). This exposes Striker 2 before it is in range to attack; a human counterproposal is to hold it behind the Brute or use a less exposed approach.
2. Seat 1 moves Sniper 6 to(6, 0) and immediately kills Striker 2. Spending damage 6 on the Striker did not help it survive this ordinary damage 2 shot.
3. On round 2, Seat 0's Sniper 3 moves to(1, 2) and kills Seat 1's Striker 5. Both high-damage pieces disappear with no exchanged Striker fire.
4. Seat 0's Brute 1 moves to(5, 0) and kills Sniper 6. This is healthy role counterplay: closing distance with a durable piece answers a fragile ranged piece.
5. Seat 1's Brute 4 closes and starts the armored duel, doing just 1 damage. Seat 0 now has 20 bounty points from two light-unit kills.
6. Round 3: Seat 0 buys another Sniper at(0, 0), moves it to(1, 0), and fires in the same turn. The original Sniper and Brute also attack, reducing enemy Brute from 7 to 4 HP. (Bounty and immediate-deploy action balance are the economy agent's territory.)
7. Seat 1 Brute kills the old Sniper, later purchases its own Sniper, and counters the reinforcement. This creates meaningful tactical reversals rather than a straight first-kill victory.
8. After further exchanges, Seat 0's Brute survives and kills the enemy Brute on round 6. The final hit does 2 due to elevation; high ground changes the damage floor, so terrain height remains mechanically important even with biomes disabled.

This sequence diagnoses both role frustration and existing counterplay. The two Striker losses follow exposed movement choices; they are not unavoidable demonstrations of a broken role.

## Manual legal encounters: deliberate choices and counters

`manual-encounters.json` contains 12 tested actions:10 accepted and 2 intentional rule rejections, with reasons and after-state HP/points.

**Brute versus unsupported Sniper, flat ground:** Sniper at(6, 2) sees Brute at(2, 2), fires, and deals 1 through defense 2. Trying to retreat after firing is rejected as `MovementEndedByAttack`. Brute then spends movement 3 to(5, 2), attacks at distance 1 and kills the HP 2 Sniper. The Brute is a real close-range counter; shoot-then-retreat kiting is already prevented. A Sniper must reposition before committing to fire and coordinate a spotter/screen. This is a deliberately constructed encounter, not a whole normal match.

**Shared-vision specialist and its counter:** A 14-point stationary, visionless gun (`HP1,damage6,range6,arc1`) plus a 6-point spotter (`HP1,vision4,visionArc1`) can legally hit a Brute 6 hexes from the gun. It deals 4, matching the Striker's raw damage against that target. The opponent deliberately shoots the spotter with a nearby Striker. Next turn the gun's identical shot is rejected as `TargetNotVisible` although it retains range. Shared vision permits efficient specialization but also supplies clear target-selection counterplay. No claim that this immobile pair dominates a normal match.

**Minimum body:** Creating `HP1` with all other stats 0 costs no design fee. Deploying it is legal and costs 1 point. It occupies one deployment cell but cannot move, see or attack. A blocker/stall concern is mechanically plausible; this check does not demonstrate a winning blocker strategy. The six-mobile-fighter test actually argues against casually assuming cheap-unit spam is dominant.

## Ranked recommendations for another pass

1. **First experimental balance branch:** test the HP 3 breakpoint for fragile units. The equal-cost HP 3/range 5 Sniper is a concrete candidate with bounded evidence. Separately test giving the Striker HP 3 at cost 21, preserving damage 6, against its current version. Do not silently ship a full roster rework from this sample.
2. **Teach the intended role through gameplay cues:** threat overlays should include enemy move-and-fire reach and expected lethal damage, not merely current range. Striker users need a clear indication that a destination exposes them to a lethal Sniper reply. This may solve part of the frustration without changing stats.
3. **Make spotting explicit:** show which ally enables a long-range shot, and whether losing that spotter removes it. The game has meaningful shared-vision tactics, but the nominal range 6 can feel misleading when the unit only sees 4.
4. **Keep flat pricing under review, focus on breakpoints first:** linear stat cost does not imply linear combat value. Before introducing complex escalating prices, test a small health adjustment with humans and a more protective controller. Shared-vision specialization appears interesting and counterable in the explicit encounter; I would not remove it.
5. **No cheap-unit nerf on this evidence:** the tested 10-point melee fighter lost 11/12 to equal-cost Strikers. Other designs, blocker-based delays and post-kill reinforcements remain separate questions.

## Evidence / reproduction

- `Program.cs`, `Probe.csproj`: deterministic bounded match controller; `dotnet run --project Probe.csproj -c Release`.
- `manual/Program.cs`, `manual/Manual.csproj`: explicit action-by-action encounters; `dotnet run --project manual/Manual.csproj -c Release`.
- `matches.json`:42 outcomes and role damage/kill counts.
- `compact-results.json` and `representative-replays.json`: compact report bundle, including seed 7 normal and both seed 7 Sniper-variant seats.
- `traces.json`: all accepted match commands, initial stats, seeded board geometry and hit results.
- `manual-encounters.json`:12 manually chosen actions and exact rejection reasons.
- `breakpoints.json`:45 attacker/target/elevation damage calculations; arithmetic evidence distinct from played outcomes.
- `summary.json`, `designs.json`, `run.log`, `manual.log`.

Source anchors, relative to release checkout: `engine/HexWars.Engine/BarracksCatalog.cs:13` (starter stats); `UnitStats.cs:49` (sum pricing); `CombatResolver.cs:13` (damage/high-ground/floor); `TargetingService.cs:27` (range) and`:41` (shared sight); `GameEngine.cs:237` (attack ends movement); `Net/GameSetup.cs:45` and`:126` (public defaults and biomes/floor).
