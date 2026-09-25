# HexWars audio sources

`prepare_tabletop_audio.py` renders the current tabletop effects and the original title arrangement.
Install `requirements.txt`, then run the script from any directory. It uses only the checked-in
source samples. Unity imports the generated effects as PCM and the stereo music as compressed
in-memory audio; no streaming or external synthesizer is required at runtime.

The 16-bar title score is in `title_theme()`: a 62 BPM upright-piano melody with open voicings,
small deterministic timing differences, and a dark room tail folded around the loop boundary.
The script renders all 25 effects, four movement takes, the music, the audition sequence and its
hash manifest. Existing sound settings and background recordings are retained.

## Sources and licensing

- Wood and soft impacts: [Kenney Impact Sounds](https://kenney.nl/assets/impact-sounds), CC0 1.0.
- Chip contacts: [Kenney Casino Audio](https://kenney.nl/assets/casino-audio), CC0 1.0.
- Quiet piano samples: [FreePats Upright Piano KW](https://freepats.zenvoid.org/Piano/acoustic-grand-piano.html#UprightKW),
  version 2022-02-21, recorded by Gonzalo and Roberto, CC0 1.0.

Publisher license/readme files accompany each source directory. `sources/provenance.json` records
publisher links and unmodified sample hashes. The arrangement and effect layering are specific to
HexWars; the sources are individual notes and contacts, not a downloaded composition.

`sources/previous` contains the actual effects from integration commit `cf4adbe` for the audition's
Previous buttons. All 17 originally supplied recordings are preserved under the existing Unity
Audio directory. `prepare_subtle_audio.py` reproduces the earlier sound pass and overwrites the
same effect output paths; use the tabletop script to restore the current palette afterwards.
