# HexWars — Unity wiring (engine → Unity)

Prep done remotely so the runway is clear for the presentation layer. **Not yet verified in the
Unity editor** (do the checks below when you're at the machine).

## What was set up

- **Engine as a precompiled plugin.** `HexWars.Engine.dll` (Release, netstandard2.1) is built from
  `engine/` and copied to `Assets/HexWars/Plugins/`. It's a **build artifact** (git-ignored);
  regenerate after engine changes with:
  ```
  powershell -File engine/build-to-unity.ps1
  ```
- **`Assets/HexWars/Presentation/HexWars.Presentation.asmdef`** — the presentation assembly.
  `overrideReferences:false` + `autoReferenced:true`, so it auto-references the Unity engine, the
  auto-referenced plugin DLL (the engine), and package assemblies (Input System, UGUI).
- **`EngineLink.cs`** — a tiny smoke test (`HexLayout.ToWorld` via the engine). If it compiles in
  Unity, the engine reference resolves. Delete it once real renderer code references the engine.
- **`HexLayout`** (in the engine) — pure flat-top axial→world (x,z), unit-tested.

## Verify in the editor (first thing)

1. Open the project in Unity 6; let it import.
2. Confirm **no compile errors** (the `EngineLink` smoke test proving the engine DLL is linked).
   - If the DLL isn't picked up: select `Assets/HexWars/Plugins/HexWars.Engine.dll`, ensure
     **Auto Reference** is on and Editor/Standalone platforms are enabled.
3. (Optional) Decide integration style: keep the **DLL** (current) or switch to **engine source as a
   Unity asmdef** (no rebuild step, but source must live under `Assets/`). DLL is simplest for now.

## Next: build the presentation (with you driving)

`BoardRenderer` (stacked hex prefab columns via `HexLayout` + elevation, terrain materials, starfield
skybox) → `UnitView`/`GeneratorView` (cyan vs red, black outlines) → `CameraRig` → `InputController`
(select → issue `Command`s) → HUD + the self-documenting create-unit point-allocation panel →
`GamePresenter` (owns `GameState`, pushes commands, renders events) → hotseat turn flow.
The aesthetic target is the mockup rendered this session.
