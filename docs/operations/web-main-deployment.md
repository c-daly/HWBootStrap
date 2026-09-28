# Web deployment branch

`main` carries the complete Unity, engine, multiplayer, research and web source.
`web-main` follows reviewed main releases and carries the same rebuilt WebGL client.
Its root `render.yaml` is the browser-only `render.web.yaml` maintained on main.
The Steam/Postgres blueprint stays on main; merging web-main back into main would
replace that deployment choice and is not the promotion workflow.

The web configuration uses one `hwbootstrap` Docker service with `INCLUDE_WEBGL=true`,
`LOBBY_PROVIDER=Legacy` and `ASPNETCORE_ENVIRONMENT=Production`. It needs no Steam App ID,
publisher key or database. It includes invite links, browser multiplayer and local play.
Legacy matches remain in memory and end when the service restarts; this is not asynchronous
email play or durable Steam matchmaking.

The gameplay AI is selected through the shared
[project/deployment model catalog](ai-model-configuration.md). Its Python CPU
runtime and immutable checkpoint are shipped in the server image; the browser
uses the server's current catalog and inference endpoint. Validate resident
memory and decision latency on the selected hosting plan before release.

For an existing Render Docker service, select `web-main` and use the repository-root
`Dockerfile`. Match the settings in `render.web.yaml`. Remove any old `DATABASE_URL` or
Steam-provider configuration that was copied from the separate Steam service. Change
`ALLOWED_WEB_ORIGINS` and `MATCH_PUBLIC_BASE_URL` if using a different public domain.
The Blueprint disables automatic deploys, so games are interrupted only when a deploy is
requested. An existing Dashboard service does not adopt these settings merely from a Git push.
No Blueprint, database or Render service is created by synchronizing the Git branches.

For the next release: integrate feature PRs into main, rebuild/stage WebGL from that source,
then merge main into web-main while retaining web-main's root `render.yaml` (copied from
main's `render.web.yaml`). Verify that `engine/HexWars.NetServer/wwwroot/Build` and the cache
key in `index.html` match the intended build before deploying. Reload browser tabs afterward.

The deployment schema and branch/automatic-deploy behavior are documented in the
[Render Blueprint reference](https://render.com/docs/blueprint-spec).
