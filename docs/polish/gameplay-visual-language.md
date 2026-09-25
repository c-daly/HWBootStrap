# Tactical interface and recognizable machines

**Implementation update — 2026-09-25.** The approved direction is now implemented in the Unity
player. The browser [design study](concept/tactical.html) remains available as the original review
artifact; its small illustrative simulation is independent of the game.

**Cleanup update.** The default panel now shows the decision and its consequence. Unit stats and
combat arithmetic open under **Unit details**. Squad cards are smaller, repeated labels are removed,
and help/tips live inside Menu. The board gains vertical space.

## Review map

| Addition | In the game |
|---|---|
| A focused battlefield | A selected-unit panel and scrollable squad replace the competing always-open army tools. **Design army** opens the existing designer and barracks. |
| Deliberate actions | Click a destination or opponent to preview; confirm in the panel to commit. A second board click never spends an action. Escape cancels a preview. |
| Movement evidence | Mint cell outlines, horizontal costs, a route line, and separate movement/climb costs and remaining budgets come from `MovementService.Routes`. |
| Attack evidence | Copper target brackets, a shot line, health remaining/lost, and the exact weapon/height/armor/terrain breakdown come from engine-validated attacks. |
| Availability | The squad retains per-unit move and attack symbols. Spent attacks remain visible. End turn asks for confirmation and shows how many units can still act. |
| Tangible pieces | Relay's tool rover, Atlas's armored crawler, Edge's twin-gun carrier, Bastion's shield tank, Glide's buggy, Crux's walker, Lance's long gun and Halo's radar rover replace the abstract forms. |
| Terrain detail | Low forest forms, water lines and rock seams complement the stepped elevation columns. These details have no colliders and do not add terrain rules. |
| Quieter presentation | Dark panels, the H-in-hex mark, explicit labels, and a persisted **Menu → Reduced motion** option. This removes camera shake/glides, projectiles, unit travel/drop animations and explosion flashes while retaining state changes and sounds. |

![Implemented attack forecast](evidence/tactical-native-attack.png)

## Try it

Open the [Windows player](../../Build/TacticalPreview/HexWars.exe), or serve the committed WebGL
files using the [browser instructions](webgl-preview.md). Start a local match.

1. Select a unit on the board or in the squad. Choose **Move [M]**, then a destination. Check both
   movement and climb budgets before confirming.
2. Choose **Attack [F]** and an opponent. The panel shows damage and health after the shot. Confirm
   to fire. **Attacking ends movement for that unit**, matching the existing engine rule.
3. Switch units and return: used actions stay used. Select End turn, then Keep playing to cancel.
4. Open **Design army**. Choose one of the eight appearances, change stats/name, save the template,
   and deploy it from the barracks. Appearance stays cosmetic and uses the existing catalog,
   deployment, command and replay IDs. Existing designs retain their selected form.
5. Open Menu for sound and Reduced motion. Use WASD/arrows to pan, Q/E to orbit, and the wheel to
   zoom. The preview arrows offer an alternative to choosing a destination/target on the board.

![Movement route and budgets](evidence/tactical-native-move.png)
![The eight machine forms](evidence/tactical-native-machines.png)

The collection describes forms, not fixed classes or new abilities. The selected unit's actual
role and stats remain separate. Player 1 uses a solid mint rim and Player 2 a broken copper rim.
Small screens stack the decision panel below the board; the panel scrolls while confirmation and
End turn remain visible. This pass targets desktop. The 390px layout is a smoke check, and still needs
larger touch text/controls and a separate pass on the secondary army tools.

## Rule and online boundaries

Previews never mutate the match. An authoritative state change invalidates a locked preview.
Commands online show a pending state until the existing server echo/rejection arrives; sending a
command is not presented as a confirmed hit. Hidden opponents are excluded from inspection,
target lists and damage forecasts. Zero-damage units and configured terrain bonuses use the
engine's actual formula. The default match has terrain bonuses disabled, which Unit details
states explicitly. Movement and climb are separate budgets, and attacking closes movement.

No engine rules, network formats, Steam requirements or multiplayer backend were changed by this
presentation pass. Browser multiplayer still uses the existing Legacy WebSocket server. Native
Steam lobbies remain a separate launch path. This work does not deploy or merge the branch.

## Validation

The implementation receipt and build hashes are recorded in
[evidence/tactical-implementation-checks.json](evidence/tactical-implementation-checks.json).
Native screenshots come from ordinary seeded matches using the actual Unity renderer and legal
engine commands. The opt-in `-tactical-capture <folder>` path is Windows-only; it does not add a
browser debug route or alter an ordinary launch.

The first PlayMode run caught a missing CanvasRenderer on the action symbols. The first native
render caught uncleared pixels outside a partial camera viewport and framing polluted by transient
board cues. Those failures were retained in `Library/TacticalImplementation` and fixed before the
final captures/build. The older HTML study's 14 browser interaction groups are separate evidence.

Final regression suites: **671 EditMode + 13 PlayMode passed**. The browser-discovered stale demo
selection reference is fixed by resolving views through the current TokenStore, with a regression
that rebuilds a match and selects before the old objects are destroyed.

A later WebGL build hit a UnityLinker `AccessViolationException` while reading assembly metadata.
The failed log is retained as `Library/TacticalImplementation/webgl-linker-crash.log`; the build
was retried without weakening validation. Browser labels use ordinary text where its runtime font
omitted a punctuation glyph.
