# Main and web release integration — 2026-09-26

The previously merged gameplay and audio PRs were stacked into feature/integration branches,
so their merged status did not mean they had reached main. This integration brings the
Steam hosting branch, the latest tabletop audio/movement branch, the checkpoint audit PR
and the Render file-watcher fix into main with their ancestry preserved.

Main retains the complete source and Steam/Postgres deployment blueprint. `render.web.yaml`
defines the separate single-instance browser deployment; web-main uses it as root
`render.yaml`. It runs Legacy multiplayer without Steam credentials or a database.
No Render service or database was provisioned, and no deployment was manually triggered.

## Integration corrections

The new mirrored backline makes an old default-board combat opener illegal. Recovery
fixtures now use an explicit 5x5 seed-7 game that still moves and damages a unit before
reconnecting. Terminal recovery assertions follow the actual final command issuer instead
of assuming player 0 wins. Journaling, replay, completion and notification assertions remain.
A focused test verifies that the compact opener damages a unit without ending the game.

## Validation

- Full engine suite: 1,219 passed, using freshly built Debug binaries. The initial Release
  run had 10 failures because process tests launched stale Debug GymServer binaries; its
  receipt is retained separately.
- Unity EditMode: 712 passed. Unity PlayMode: 48 passed. Coplay reported no compile errors.
- Focused server suite: 127 passed across request limits, durable coordinator, shutdown,
  live matches, recovery service and the compact replay fixture. Initial stale-fixture
  failures and a corrected qualification compile error are retained. This is not a full
  PostgreSQL integration rerun.
- Real .NET 8.0.28 Production Legacy server: two WebSocket clients verified private-room
  join, move/undo broadcasts, restored position and budgets, forged-seat rejection,
  reconnect retaining undo, and rejection of undo after turn handoff. Native published
  server legacy selftest passed, including damage replay on reconnect.
- Python checkpoint audit: 151 tests passed in the first complete run; three artifact
  lookup tests failed because Windows Git could not resolve the WSL worktree pointer.
  Those three passed in a separate corrected, read-only invocation. This is not a single
  uninterrupted 154-test pass.
- WebGL: Unity 6000.5.0f1 build succeeded with zero errors in 10m41s. Staged all four
  payloads and the generated index into the server's actual wwwroot. Cache key: `b214de56`.
  The build used cached worktree commit `c481ceb2`; its Assets, Packages, ProjectSettings
  and engine source are identical to the integrated candidate. Later corrections affect
  server verification fixtures, Python and deployment metadata only.
- Fresh WebGL browser smoke: Chrome loaded cache key `b214de56`, rendered the title screen
  with the H logo, and reported no browser errors or failed requests. This check covers
  boot/rendering; the transport proof above exercises multiplayer actions separately.
- Parsed the web configuration and checked the single Legacy service, web-main branch,
  manual deployment setting, included WebGL and absence of Steam/database requirements.

## Retained local receipts and recovery

Unity/engine logs and XML/TRX receipts are in the cached player worktree's
`Library/MainIntegration`. Server publish/transport, Python and browser receipts are in
the integration worktree's `Library/MainIntegration`; focused server receipts and PR
audit are in `/tmp/hexwars-main-integration`.

A second full Python invocation was interrupted after inherited Git path variables let
a temporary-repository test touch the real local integration metadata. Its accidental
fixture commit is preserved only on local `recovery/checkpoint-audit-fixture-20260926`.
The integration branch/index and shared Git configuration were restored while preserving
source changes; the original dirty research checkout remained unchanged. Git connectivity
verification passed. No remote refs were changed by that incident. The final three-test
rerun was restricted to read-only artifact checks.
