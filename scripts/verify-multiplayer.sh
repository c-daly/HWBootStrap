#!/usr/bin/env bash
set -euo pipefail

# Local proof only: both durable modes DROP the disposable database's public schema.
# No Steam registration, publisher key, or live account is required.
if [[ -z "${HEXWARS_TEST_DATABASE_URL:-}" ]]; then
    echo 'Set HEXWARS_TEST_DATABASE_URL to a disposable PostgreSQL database named with the word test.' >&2
    exit 3
fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$repo_root"
sdk="${HEXWARS_DOTNET_SDK:-dotnet}"
runtime="${HEXWARS_DOTNET_RUNTIME:-dotnet}"
export DOTNET_HOSTBUILDER__RELOADCONFIGONCHANGE=false

"$sdk" build engine/HexWars.NetServer/HexWars.NetServer.csproj -c Release --nologo --verbosity minimal
server=engine/HexWars.NetServer/bin/Release/net8.0/HexWars.NetServer.dll
"$runtime" exec --roll-forward LatestPatch "$server" selftest
"$runtime" exec --roll-forward LatestPatch "$server" selftest-durable
"$runtime" exec --roll-forward LatestPatch "$server" selftest-durable-crash
