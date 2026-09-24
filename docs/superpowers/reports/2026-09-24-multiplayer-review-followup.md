# Multiplayer review follow-up

This follow-up addresses all six findings in PR #21 on top of `7048f42`.

| Finding | Result |
| --- | --- |
| Slow chunked bodies occupy public connections | Undeclared request-body reads have an independent five-second deadline and return 408. Real Kestrel TCP tests cover unfinished GET and POST bodies that stay within the transport minimum rate. |
| A database outage resembles a corrupt journal | Failed reads and journals disappearing after enumeration return exit 2 with `INCOMPLETE`, never a corrupt-journal summary. Exception details are redacted. |
| Terminal history blocks promotion | Output states its scope. Runbooks use `verify-journals --open-only` for readiness and keep a separate audit of retained historical matches. |
| An interrupted drain reports zero | The shutdown summary reports an unknown interrupted result unless a completed drain supplied a count. |
| Reconciliation reads have no deadline | Each reconciliation pass receives a five-second cancellation deadline and resumes after its previous attempt, preventing a slow first read from starving later cached matches. Successfully inspected terminal matches are evicted immediately. |
| A local process proof is called staging verification | The promotion checklist now separately requires the disposable local proof and two packaged clients playing and reconnecting against staging. |

Validation: **959 server tests passed, zero failed or skipped**, including PostgreSQL integration and real loopback Kestrel requests. The affected subset previously passed 37 tests. The production .NET 8 process checks also passed: legacy WebSocket play/reconnect, graceful process restart, and forced termination, including a complete 12-command game and both terminal reconnects. Steam identities were scripted against the owned disposable PostgreSQL 16 instance.

Receipts are retained under `Library/PrFollowup-20260924/`. The first focused run had 14 setup failures because the temporary PostgreSQL instance started on its default loopback port 5432 while the fixture expected 55432; the listener was corrected and the original failure receipt retained. No production or staging database was contacted.

No migration, game rule, Steam identity contract, deployment, or live credential changed. Live Steam/Render validation still requires the actual App ID, accounts, and deployment. The pre-existing test-only SSH.NET dependency advisories remain visible in build output.
