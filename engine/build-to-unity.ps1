# Builds HexWars.Engine (Release) and copies the DLL into the Unity project's Plugins folder.
# Run this after changing engine code so Unity picks up the new build.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot               # repo root (this script lives in engine/)
$dotnet = "C:\Program Files\dotnet\dotnet.exe"

& $dotnet build -c Release "$root\engine\HexWars.Engine\HexWars.Engine.csproj"

$dll = "$root\engine\HexWars.Engine\bin\Release\netstandard2.1\HexWars.Engine.dll"
$dest = "$root\Assets\HexWars\Plugins"
New-Item -ItemType Directory -Force -Path $dest | Out-Null
Copy-Item $dll $dest -Force
Write-Host "Copied HexWars.Engine.dll -> $dest"
