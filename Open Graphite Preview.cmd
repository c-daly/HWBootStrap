@echo off
cd /d "%~dp0"
if not exist "Build\GraphitePreview\HexWars.exe" (
 echo The Windows preview has not been built. See docs\polish\windows-preview.md.
 pause
 exit /b 1
)
start "HexWars Graphite" "Build\GraphitePreview\HexWars.exe" -graphite-workshop -screen-fullscreen 0 -screen-width 1600 -screen-height 900
