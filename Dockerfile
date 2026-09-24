# ---- build the .NET server (which also serves the WebGL client from wwwroot) ----
FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src

# restore first (better layer caching)
COPY engine/HexWars.Engine/HexWars.Engine.csproj       engine/HexWars.Engine/
COPY engine/HexWars.NetServer/HexWars.NetServer.csproj engine/HexWars.NetServer/
RUN dotnet restore engine/HexWars.NetServer/HexWars.NetServer.csproj

# copy sources (engine/HexWars.NetServer/wwwroot holds the compiled WebGL client) and publish
COPY engine/HexWars.Engine/    engine/HexWars.Engine/
COPY engine/HexWars.NetServer/ engine/HexWars.NetServer/
RUN dotnet publish engine/HexWars.NetServer/HexWars.NetServer.csproj -c Release -o /app /p:UseAppHost=false

# One repository, two deployments. The legacy WebGL service serves the browser client out of wwwroot and
# keeps the default; the Steam match service is built with --build-arg INCLUDE_WEBGL=false and drops it,
# because a Unity bundle it will never serve is tens of megabytes in every image, every pull and every
# rollback. Dropped after publish rather than before restore, so both images share every layer above.
#
# On Render this arrives as a service environment variable: Render translates the environment variables
# of a Docker service into build arguments, so INCLUDE_WEBGL=false in render.yaml reaches this ARG.
ARG INCLUDE_WEBGL=true
RUN if [ "$INCLUDE_WEBGL" != "true" ]; then rm -rf /app/wwwroot; fi

# ---- runtime ----
FROM mcr.microsoft.com/dotnet/aspnet:8.0
WORKDIR /app
COPY --from=build /app ./
ENV PORT=8080
# The deployed image is immutable, so live appsettings reload is unnecessary. Disabling it also
# prevents ASP.NET Core default configuration providers from allocating inotify watchers on hosts
# with a low per-user inotify limit (such as Render).
ENV DOTNET_HOSTBUILDER__RELOADCONFIGONCHANGE=false
# Fail closed by default: a container started with no environment at all is a production container, and
# the configuration validation refuses to serve one that is missing its secrets rather than coming up
# half configured. A development run overrides this explicitly.
ENV ASPNETCORE_ENVIRONMENT=Production
# Workstation GC on purpose. Server GC sizes its heaps per core and is the right default for a machine
# that owns its CPUs; this runs one small instance on a shared plan, where per-core heaps cost resident
# memory the match host would rather spend on projections.
ENV DOTNET_gcServer=0
EXPOSE 8080
# The aspnet image ships a non-root user named app. Nothing here writes to the filesystem and 8080 is
# unprivileged, so there is no reason for this process to be able to modify its own image.
USER app
# CMD rather than ENTRYPOINT, so the whole command can be replaced rather than appended to. The
# subcommands an operator runs against this image - describe-environment, selftest, selftest-durable -
# are then written the way they read, and the Render Docker Command field, which replaces CMD,
# behaves as its documentation says rather than passing arguments to a fixed dotnet invocation.
CMD ["dotnet", "HexWars.NetServer.dll"]
