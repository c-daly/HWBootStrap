# Multiplayer integration and local validation — 2026-09-24

The hosting and security work is combined on `codex/steam-multiplayer-integration-20260924`.
Local multiplayer verification accepts development placeholders. A registered Steam App ID is not
required for the scripted workflow. Live Steam and Render promotion remain outstanding.

## Source and preservation

- Remote `steam-hosting/integration` was verified at `0f0047cc8941534c49668e33ca4e94ff0d1cd0ff`.
- Hosting source: `steam-hosting/06-render-ops`, committed tip `524bb10`, plus seven dirty files.
- Security source: `steam-hosting/07-security-ops`, committed tip `405952e`, plus twelve dirty files
  and the untracked `CallerKey.cs` and `SafeLogging.cs` implementations.
- Both working states were snapshotted before merging. Their source worktrees remain unchanged;
  patch and untracked-file hashes were checked against the snapshots.
- The root checkout remains on its existing `codex/tactical-v3-safe-stop-retry` branch with its
  pre-existing local work. AI training was not resumed.
- The isolated, durable worktree is `.worktrees/steam-multiplayer-integration-20260924` under the
  main repository. The first local integration commit is `5fb5337`; subsequent Unity compatibility
  and validation documentation changes belong to the same branch.

## Added in this integration — review map

1. **Hosting/security composition:** merged shutdown barriers, readiness, metrics, credential
   redaction, per-caller quotas, retention and recovery. Conflict resolution keeps metrics-aware
   rejection and redacted exception logging together, registers retention alongside health checks,
   and keeps the legacy health route's CORS policy.
2. **Health-check logging repair:** ASP.NET logs `HealthCheckResult.Exception` independently of
   the response writer. The combined code initially leaked synthetic database credentials through
   that path. Two new integration tests failed before the fix and passed afterward. Health checks
   now supply sanitized exception descriptions without retaining the raw inner exception;
   shutdown failures also use the redacted logger.
3. **Real process recovery proof:** `selftest-durable` now starts distinct child server processes.
   The previous implementation replaced in-process hosts while describing that as a process restart.
   Added a forced-termination variant, full-game completion, durable winner/status checks, terminal
   reconnection for both players, bounded socket operations, and child cleanup. Legacy self-testing
   now uses the actual server composition required by the security middleware.
4. **Placeholder workflow:** `scripts/verify-multiplayer.sh` builds Release and runs legacy play,
   graceful recovery and forced-termination recovery. Its runtime invocation pins .NET 8 rather
   than silently rolling forward to a different runtime major. The existing synthetic App ID
   `480000`, two synthetic accounts and scripted Steam API require no publisher key or registration.
   The shipped client endpoint remains the explicit `OWNER-INPUT` placeholder.
5. **Unity validation repairs:** qualified `UnityEngine.Object`, replaced unsupported NUnit
   constraints, and made menu construction take an explicit platform value so legacy input tests
   remain meaningful in a Steam-enabled editor. Added assertions for both menus. PlayMode tests
   inject their fake socket transport for every lobby handoff and no longer suppress all error logs.
   Retained the `STEAMWORKS_NET` Standalone build symbol installed by the pinned Steamworks package.

## Verification

| Check | Result | Runtime / scope |
|---|---|---|
| Complete NetServer suite | 953 passed, 0 failed/skipped | .NET 10.0.12, disposable PostgreSQL 16.15; existing test-project target |
| Complete engine and linked client suite | 1,170 passed, 0 failed/skipped | .NET 8.0.31 |
| Updated linked Steam client assertions | 84 passed, 0 failed/skipped | .NET 8.0.31; follow-up after NUnit compatibility edits |
| Legacy transport verifier | Passed | Two real WebSocket clients, play and reconnect, Release server on .NET 8.0.31 |
| Graceful process recovery | Passed | Three distinct server processes; full 12-command game; Player0 winner |
| Forced-termination recovery | Passed | First child killed; new process recovers and completes the same 12-command game |
| Terminal reconnects | Passed in both recovery modes | Both seats receive the full identical terminal state after another process restart |
| Journal accounting | Passed in both recovery modes | Every acknowledged command appears exactly once, in order; persisted winner/status/timestamp checked |
| Health-check leak regression | 2 failures reproduced, then 2 passes | Synthetic credentials checked in response bodies and framework logs |
| Unity EditMode | 662 passed, 0 failed/skipped | Unity 6000.5.0f1; both title-menu variants and Steam client assertions |
| Unity PlayMode | 5 passed, 0 failed/skipped | Placeholder Steam lobby/API/socket fixtures; real MonoBehaviour execution |
| Steamworks.NET package resolution | Passed | Pinned revision resolved in Unity; no package-lock changes |
| Engine plugin identity | Passed | Windows Release output and Unity plugin SHA-256 both `5e51c0b1295edfb4a435f9b8db91f9177f14019d2a6e562ad2d43865e074fff2` |
| Verification script with no database configured | Exit 3 | Stops before building or opening a database |

The process verifier uses real HTTP/WebSockets, the production server pipeline, the deterministic
engine and real PostgreSQL. The Steam partner API is scripted. Unity smoke tests use fake Steam,
match allocation and socket transports; these are separate proofs, not a real Steam session.

## Evidence and reproducibility

See [local multiplayer validation](../../operations/local-multiplayer-validation.md) for commands.
Local raw evidence is retained under `Library/MultiplayerValidation-20260924/`, including source
snapshots, compressed TRX reports, red/green security evidence, build logs and both process-verifier
runs. Unity XML reports and import/test logs are retained under `Library/multiplayer-*`.

PostgreSQL 16.15 was built under `/tmp` from the
[official source release](https://www.postgresql.org/ftp/source/v16.15/), with its published SHA-256
verified, and bound only to `127.0.0.1:55432`. The ASP.NET Core 8.0.31 runtime was downloaded from
Microsoft's release metadata and SHA-512 verified. No system database service was installed.
The temporary database was stopped after validation. Unrelated editor-generated settings changes
were archived in the evidence directory and restored before the final commit.

Earlier failures remain in the evidence: a sandboxed build failed without compiler diagnostics;
the first native build identified a missing metrics argument; an engine-test attempt lacked the
.NET 8 runtime; initial Unity runs exposed the compatibility and test-isolation issues repaired
above. An initial server verifier run rolled forward to .NET 10.0.12 and exited 134 with an internal
CLR error. Its cause is unproven; the deployment-runtime .NET 8 checks subsequently passed, and the
script now requests that runtime explicitly.

## Remaining promotion gates

- Register the Steam base-game and Playtest App IDs, configure publisher credentials through the
  secret store, and verify actual two-account authentication, invitations and lobby membership.
- Apply the Render configuration and verify the container, staging endpoint, TLS, deployment
  interruption and rollback. Docker was unavailable during this local validation.
- Run an actual two-player Unity/Steam session against staging and visually inspect gameplay,
  interruption, recovery and player-facing error states. Passing scripted PlayMode tests is not
  that live session.
- Perform the planned load and operational drills before selecting a public invite cap. Keep the
  single-instance topology until room ownership/routing is implemented and failure-tested.
- Address existing dependency-maintenance warnings: the test-only Testcontainers dependency brings
  SSH.NET 2023.0.0 advisories. Those warnings were retained, not suppressed.

This is local implementation and validation evidence. It does not establish a deployed or
release-ready Steam Playtest.
