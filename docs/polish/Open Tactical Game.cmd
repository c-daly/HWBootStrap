@echo off
set "TACTICAL_EXE=%~dp0..\..\Build\TacticalPreview\HexWars.exe"
if not exist "%TACTICAL_EXE%" (
  echo Build the Windows player using HexWars - Build Tactical Windows Preview in Unity.
  pause
  exit /b 1
)
start "HexWars" "%TACTICAL_EXE%"
