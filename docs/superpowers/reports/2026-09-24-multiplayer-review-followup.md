# Multiplayer review follow-up

This follow-up addresses all six findings in PR #21 on top of `7048f42`.

| Finding | Result |
| --- | --- |
| Slow chunked bodies occupy public connections | Undeclared bodies on routes without explicitly marked consumers are rejected before reading, including POSTs to GET-only, unknown and bodyless endpoints. Matchmaking upload buffering runs after the endpoint rate limiter and has a 74-second maximum: the 16 KiB cap at Kestrel's 240 bytes/second minimum plus its five-second grace period. Real Kestrel TCP tests cover immediate GET rejection, unfinished POST expiry and a maximum-size upload at 240 bytes/second plus completion just before the independent deadline. |
| A database outage resembles a corrupt journal | Failed reads and journals disappearing after enumeration return exit 2 with `INCOMPLETE`, never a corrupt-journal summary. Exception details are redacted. |
| Terminal history blocks promotion | Output states its scope. Runbooks use `verify-journals --open-only` for readiness and keep a separate audit of retained historical matches. |
| An interrupted drain reports zero | The shutdown summary reports an unknown interrupted result unless a completed drain supplied a count. |
| Reconciliation reads have no deadline | Each reconciliation pass receives a five-second cancellation deadline and resumes after its previous attempt, preventing a slow first read from starving later cached matches. Successfully inspected terminal matches are evicted without waiting for busy gates; those matches are retried later while other reaped matches still close in the current pass. |
| A local process proof is called staging verification | The promotion checklist now separately requires the disposable local proof and two packaged clients playing and reconnecting against staging. |

First-round validation: **959 server tests passed, zero failed or skipped**, including PostgreSQL integration and real loopback Kestrel requests. The affected subset previously passed 37 tests. The production .NET 8 process checks also passed: legacy WebSocket play/reconnect, graceful process restart, and forced termination, including a complete 12-command game and both terminal reconnects. Steam identities were scripted against the owned disposable PostgreSQL 16 instance.

Receipts are retained under `Library/PrFollowup-20260924/`. The first focused run had 14 setup failures because the temporary PostgreSQL instance started on its default loopback port 5432 while the fixture expected 55432; the listener was corrected and the original failure receipt retained. No production or staging database was contacted.

No migration, game rule, Steam identity contract, deployment, or live credential changed. Live Steam/Render validation still requires the actual App ID, accounts, and deployment. The pre-existing test-only SSH.NET dependency advisories remain visible in build output.


The second review identified that the original five-second upload deadline was too aggressive for progressing uploads, and waiting for one reconciliation gate could consume its entire budget. Both were corrected without removing the overall body/reconciliation bounds. Final validation passed **961 server tests, zero failed or skipped**, followed by another successful .NET 8 legacy/graceful/forced-termination process-verification run on the final code. All earlier receipts remain alongside the final receipt.


The subsequent full run exposed a concurrent shutdown race: the second host stop call could return before the first call finished its drain and summary. Shutdown now shares one completion task across all callers. The shutdown integration test invokes concurrent stops and requires one completed summary. The earlier failed full-run receipt remains retained; it is not treated as a pass.


The first controlled-clock upload test advanced Kestrel's pre-existing timer before the request-body timer was installed. The test now records the startup timer count and waits for the additional request deadline before advancing time; the corrected real-socket suite passed all eight cases. That initial test failure is also retained.


## PR #22 comment follow-up

The next review correctly found that checking the HTTP method alone still buffered POSTs to GET-only routes. Body buffering now requires explicit `WithHexWarsRequestBody` metadata on the matched endpoint. Only the create/join matchmaking handlers opt in; rejected methods, unknown routes and bodyless POST handlers are refused immediately with the connection closed. Named rate limits still run before buffering.

The original six-second test did not prove the upload boundary. It now completes the full 16 KiB body at 73.5 seconds on the controlled application clock, while the existing incomplete-body case expires at 74 seconds. A separate real-clock, real-Kestrel socket test sends all 16 KiB in 240-byte-per-second chunks over 68 seconds and verifies the entire body reaches the handler.

The first real-clock test incorrectly treated Kestrel's five-second grace period as time excluded from the average. Kestrel correctly closed that below-minimum-rate upload; the failed receipt is preserved. The corrected test sends at the minimum rate from the beginning, matching Kestrel's actual rate calculation without changing the production minimum data rate or upload deadline.

The maximum-size tests also reproduced a production boundary defect: Kestrel counts chunk framing against its transport limit, so exactly 16 KiB of payload received a 413. Explicit body consumers now have a separate bounded wire allowance of 96 KiB plus the five-byte terminator, enough for one-byte chunks. The decoded-content probe remains capped at 16 KiB plus one byte. Tests accept exact-size bodies with one-byte and 240-byte chunks and reject oversized chunk extensions. The first follow-up assertions also exposed a test-only response assumption (the echo response was itself chunked); the fixture now returns explicit byte content so the complete echoed payload can be checked directly. These failure receipts remain retained alongside the corrected focused run.

Current comment-follow-up validation: **969 server tests passed, zero failed or skipped**, including all 16 real-Kestrel cases and the complete PostgreSQL suite. The focused subset of 15 cases passed separately before the full run. Receipts for this round are retained under `Library/Pr22Comments-20260924/`; the complete raw TRX is also retained under `/tmp/hexwars-pr22-comments-20260924/`.

The final .NET 8.0.31 Release process verifier also passed all three modes on the comment-follow-up code: legacy WebSocket play/reconnect, graceful restart, and forced termination with a complete 12-command game and both terminal reconnects. The owned disposable database was stopped afterward.
