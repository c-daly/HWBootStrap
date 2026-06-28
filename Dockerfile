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

# ---- runtime ----
FROM mcr.microsoft.com/dotnet/aspnet:8.0
WORKDIR /app
COPY --from=build /app ./
ENV PORT=8080
EXPOSE 8080
ENTRYPOINT ["dotnet", "HexWars.NetServer.dll"]
