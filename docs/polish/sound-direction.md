# A quieter HexWars

Sound should give actions a little physical weight and make the board pleasant to spend time with.
The first pass uses soft mechanical contacts, brief weapon tails and warm, restrained turn notes.
The existing title music stays; its level is lower and transitions fade.

## Try it

With the [local WebGL preview](webgl-preview.md) running, open
[the sound study](http://localhost:8196/sound-study/). Start with the 20-second sequence, then use
**Current / Proposed** to compare each cue. Both versions use the same playback gain; the proposed
samples include their quieter mastering. Nothing autoplays, only one preview plays at a time,
and **Escape** stops it. The page's playback level does not change game settings.

The game uses the same effect WAVs. [A captured move and attack](evidence/sound-game-combat.webm)
records the rebuilt browser game with ambience turned down. In a match, open **Menu** for **Master**, **Effects**,
**Ambience** and **Music** controls. They save independently; existing master/mute preferences
remain intact. Ambience also controls the workshop hum. The native preview is
`Build/TacticalPreview/HexWars.exe` in this feature checkout.

## What changed

- Selection is a very quiet contact; movement and starting placement use slightly firmer contacts.
- The twelve existing weapon variations have shorter tails, rounded high frequencies and quieter
  peaks. The tiers retain different weight; a private audio random stream avoids immediate repeats.
- Deployment and design confirmation are brief mechanical cues. Turn changes use two gentle notes.
- One action gets one resolution cue: conclusion, destruction, or automatic turn handover.
  Fast-forwarding a queue does not play every skipped result at once.
- Six effect voices and short per-cue cooldowns limit chatter. A quiet selection never steals
  an important combat voice. Important effects briefly lower board ambience.
- Title music, board ambience and workshop hum fade between states. Brief endpoint tapers smooth
  the title and hum loop seams. Background focus loss silences audio; returning fades the beds in.

No new soundtrack or constant UI-hover sounds were added. All 17 supplied recordings remain
unchanged. Derived effects live in `Assets/HexWars/Resources/Audio/Soft`; their reproducible
NumPy-only preparation script is `scripts/audio/prepare_subtle_audio.py`. Run it from the repo
root, then use **HexWars > Audio > Prepare Quiet Effects** in Unity to apply PCM import settings.
The audition's `palette.json` records sample hashes, peak/RMS levels and sequence timings.

## Validation and scope

The [sound validation receipt](evidence/sound-polish-checks.json) records Unity tests, build hashes
and browser checks. Audio checks measure this page/game's own Web Audio signal; they do not capture
system audio. Focus-loss behavior passes its Unity PlayMode check; Chrome headless kept the
original tab focused during the tab-switch attempt, so browser focus loss remains unverified.
These checks establish playback, headroom and control behavior. The audition is for
judging the subjective balance on your speakers or headphones.

The sound branch is stacked on the gameplay-polish PR. It includes refreshed native and staged
WebGL builds locally; the staged browser bundle is committed. It has not been deployed to production.
