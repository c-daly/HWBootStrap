# Multiplayer review follow-up

This follow-up addresses all six findings in PR #21 on top of `7048f42`.

| Finding | Result |
| --- | --- |
| Slow chunked bodies occupy public connections | Undeclared bodies on methods without consumers are rejected before reading. POST upload buffering runs after the endpoint rate limiter and has a 74-second maximum: the 16 KiB cap at Kestrel's 240 bytes/second minimum plus its five-second grace period. Real Kestrel TCP tests cover immediate GET rejection, unfinished POST expiry and a progressing upload that completes after five seconds. |
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
