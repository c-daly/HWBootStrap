# Render Steam Playtest runbook

How the HexWars match service is deployed, released, rolled back, watched and sized on Render. This is the
procedure document; the environment contract is
[Steam / Render environment authority](steam-render-environments.md), the probe and shutdown behaviour is
[Health, metrics and graceful shutdown](health-and-shutdown.md), and the blueprint that provisions all of
it is `render.yaml` at the repository root. **Status: Written, not yet applied.** **Date: 2026-09-05.**

Nothing here has been run against a Render account. Applying the blueprint creates paid resources and is an
owner decision.

## 1. Topology

```
         Steam client (Unity)                     browser (WebGL, legacy)
                |                                          |
   HTTPS  POST /api/v1/steam/matches                 HTTPS  GET /
   WSS    /ws/v2                                     WSS    /ws
                |                                          |
                v                                          v
      +---------------------+                  +------------------------+
      |    Render router    |  TLS terminated  |     Render router      |
      +----------+----------+                  +-----------+------------+
                 |  X-Forwarded-For, -Proto                |
                 v                                         v
      +---------------------+                  +------------------------+
      |    hexwars-match    |  1 instance      |  the web-main service  |
      |     0.5c-512mb      |  INCLUDE_WEBGL   |  (WebGL, unchanged)    |
      |    Docker, main     |  = false         |                        |
      +----------+----------+                  +------------------------+
                 |  private network, same region
                 v
      +---------------------+
      |  hexwars-match-db   |  Postgres 16, ipAllowList: []
      +---------------------+

      hexwars-match-staging and hexwars-match-staging-db are the same pair,
      built from steam-hosting/integration, with secrets of their own.
```

The match service and its database sit in the same region and talk over the private network. The database
has an empty IP allow list, so nothing on the public internet can open a connection to it at all.

## 2. Prerequisites

Every one of these is an owner input, and none can be derived from the repository.

| Needed | For | Where it goes |
|---|---|---|
| Steamworks base App ID and Playtest App ID | Identifying tickets and lobbies | `STEAM_APP_ID`, per service |
| Publisher Web API key | Server-side ticket and lobby calls | `STEAM_PUBLISHER_WEB_API_KEY`, per service |
| A domain for the match service | The websocket URL clients are handed | `MATCH_PUBLIC_BASE_URL`, custom domain |
| Playtest region | Blueprint, unchangeable after creation | `region:` in `render.yaml` |
| Postgres plan decision | Blueprint, cost | `plan:` under `databases:` |
| Log retention window | Render log settings, cost | Dashboard, see section 8 |

The publisher key is a **secret**. It is typed into the Render Dashboard, and never committed, never logged
and never pasted into a ticket.

## 3. First deploy

1. **Apply the blueprint.** In the Render Dashboard, create a Blueprint from this repository. Review what it
   proposes: two web services and two Postgres instances. This is the point at which billing starts.
2. **Fill the secrets.** For each of `hexwars-match` and `hexwars-match-staging` set `STEAM_APP_ID`,
   `STEAM_PUBLISHER_WEB_API_KEY`, `MATCH_PUBLIC_BASE_URL`, `ALLOWED_WEB_ORIGINS` and
   `MATCH_COMPATIBLE_CLIENT_BUILDS`. Staging gets values of its own: a staging service holding the
   production publisher key is a production secret kept in a second place.
3. **Confirm each database is private.** Open it and check that the IP allow list is empty and that the
   service is using the internal connection string. `DATABASE_URL` is wired by the blueprint and is never
   typed in by hand.
4. **Add the custom domain and let TLS issue.** Add the domain to `hexwars-match`, point DNS at Render and
   wait for the certificate. Until it is live every client is refused: in Production the server accepts a v2
   socket only over https.
5. **Set `MATCH_PUBLIC_BASE_URL` to `https://<domain>`.** It has to be the domain clients actually reach.
   The websocket URL in every create and join response is derived from it, so a stale value sends every
   client to the wrong host.
6. **Check readiness.**

   ```
   curl -sS https://<domain>/health/ready
   ```

   It must answer 200. If it answers 503, read the body: it names the check that failed.

7. **Check the socket.** A real handshake needs a real credential, so the honest check is the client: run
   the Unity client against the deployed host. To confirm only that the upgrade is accepted:

   ```
   websocat -v wss://<domain>/ws/v2
   ```

   The socket should open and then close with 1008 after the auth deadline, having been sent nothing. A
   refusal at the upgrade itself means the origin policy, the transport check or the per-IP cap turned it
   away, and none of those are the credential.

8. **Record what was deployed.** `GET /health/live` reports the build id, which is `RENDER_GIT_COMMIT`, so
   it names the exact commit that is serving.

## 4. Deploy procedure

Auto-deploy is off on both services. Every release is deliberate.

1. **Staging first.** Trigger a manual deploy of `hexwars-match-staging` from the integration branch.
2. **Wait for readiness.** Staging `GET /health/ready` must be 200 with all four checks Healthy. A
   `Degraded` recovery check means staging is carrying a match this build refuses; see the
   [match recovery runbook](match-recovery-runbook.md) before going further.
3. **Run the multiplayer verification.** First run `scripts/verify-multiplayer.sh` against a disposable
   local database, as described in [local multiplayer validation](local-multiplayer-validation.md).
   It completes games through real process restarts with a scripted Steam API. Then run two real Steam
   clients against staging through a full game and reconnect. The local script does not target staging
   or prove real Steam authentication; readiness alone does not establish either result.
4. **Deploy production by hand.** Trigger a manual deploy of `hexwars-match`.
5. **Watch the handover.** Render sends SIGTERM to the old instance. It flips readiness to false, drains
   in-flight commits for up to 10 s, sends every seated socket `SERVER RESTART`, then closes them with
   1012. Clients reconnect, re-authenticate and are dealt `START` with the whole log, so a game in progress
   survives a deploy as a pause rather than a loss. `maxShutdownDelaySeconds` is 60, comfortably more than
   the 25 s the process allows itself.
6. **Confirm.** `GET /health/live` reports the new commit and `GET /health/ready` is 200.

A deploy during a busy period interrupts every live game for a few seconds. Prefer a quiet window.

## 5. Rollback

1. In the Render Dashboard, open the service, find the previous successful deploy and **Redeploy** it.
2. That is the whole procedure. **Never run a down migration.** Migrations here are additive-only, so the
   schema a newer build applied is still valid for the build before it: an added column or table is simply
   unused. Rolling the schema back would delete data the newer build already wrote.
3. If a new migration really is incompatible with the previous app version, **roll forward** - ship a fix on
   top rather than undoing the schema. The additive-only rule exists so this case does not arise, and a
   migration that breaks it should not have merged.
4. After a rollback, check readiness, and read the `schema` check the right way round. It compares the
   migrations THIS BUILD carries against the ledger in the database and reports the ones the ledger does
   not have. So `pending` always means the database is BEHIND the build - which is not the rollback case
   and is normally impossible, because the startup migration applies them before the host serves anything.
   Seeing it after a rollback means the deploy you rolled back to could not migrate; fix that, do not roll
   further back.
5. A build that is older than the schema is the expected rollback outcome, and it reports **Healthy**. The
   ledger holds migration ids this build has never heard of, and the runner ignores them: it only ever asks
   what of its own is missing, never what else is there. That is what additive-only buys - the older build
   simply does not use the newer column or table. If you want to confirm it, the extra ids are visible in
   `SELECT version FROM schema_migrations`.

## 6. Scaling

**Exactly one instance, until the horizontal-scaling gate is done.**

A match lives in the memory of the process hosting it. A second instance would accept handshakes for
matches it is not holding, load its own projection of the same game from the journal, and the two copies
would diverge the moment both accepted a command. Nothing in the current design prevents that: there is no
sticky routing by match id and no explicit ownership of a match.

So scale **up**, never out. If the instance runs short of CPU or memory, move the plan from `0.5c-512mb` to
`1c-2g`. Raising `numInstances` above 1 is a correctness bug, not a capacity decision.

What would lift the gate: route every request and socket for a match id to one instance, and make match
ownership explicit and transferable. Until then the blueprint pins `numInstances: 1`, and this section is
why.

## 7. Alerting

Render has no built-in alert on an endpoint, so the uptime check is external.

| Alert | Condition | Why it matters |
|---|---|---|
| Service down | External uptime check on `GET /health/ready` every 60 s, alerting after 2 consecutive failures | One failed probe is a deploy; two is an outage |
| Unrecoverable match | Log match `Match .* cannot be recovered` | A journal that needs a human; follow the [match recovery runbook](match-recovery-runbook.md) |
| Steam refusing us | Log match `Steam publisher key rejected` | A revoked or wrong publisher key stops every create and join |
| Database failures | `databaseFailures` above 0 in the `Metrics` log line | Durable writes are failing and players are seeing `REJECT TemporaryFailure` |
| Restart loop | More than 2 Render restart events in 10 minutes | The process is crashing rather than reporting unready |
| Reconnect spike | `reconnects` rising by more than 20 between two consecutive `Metrics` lines | Sockets are being dropped somewhere between the client and here |

The `Metrics` line is written once a minute at Information and carries the whole snapshot, so the last three
alerts are log rules over that one line. `GET /api/v1/metrics` serves the same snapshot to anything that
would rather poll; it needs the `X-Metrics-Token` header, whose value Render generated and holds in the
Dashboard.

## 8. Cost and retention

- **Spend alert.** Set a workspace spend alert before the first apply. Four paid resources at the plans in
  the blueprint is the floor, and the database plan is the line most likely to need raising.
- **Plan sizing.** `0.5c-512mb` for the service and `0.1c-256mb` for the database are Playtest sizes, not
  launch sizes. The service is CPU-light and bounded by live projections in memory; the database is bounded
  by connections, and one instance opens at most 20 of them. Raise the database first.
- **Log retention.** OWNER-INPUT. Render retains logs for a window that depends on the workspace plan.
  Decide it before the Playtest opens: the procedures in the recovery runbook depend on being able to read
  back what a failed match did.

## 9. Promotion checklist

Before promoting a staging build to production:

- [ ] Staging `GET /health/ready` is 200 with all four checks Healthy
- [ ] `scripts/verify-multiplayer.sh` passed against staging
- [ ] `verify-journals` passed against a restored copy of the staging database. NOT `selftest-durable`:
      it drops the schema of whatever it is given, so it belongs on a throwaway database only
- [ ] No new migration breaks the additive-only rule
- [ ] The client build in the Playtest depot is listed in `MATCH_COMPATIBLE_CLIENT_BUILDS`, or that list is empty
- [ ] A quiet deploy window is chosen, or the interruption is acceptable
- [ ] The previous production deploy is identified, so the rollback target is known before it is needed

## 10. Related documents

- [Steam / Render environment authority](steam-render-environments.md)
- [Health, metrics and graceful shutdown](health-and-shutdown.md)
- [Match recovery runbook](match-recovery-runbook.md)
- [Protocol v2](protocol-v2.md)
