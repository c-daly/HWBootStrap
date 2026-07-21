# Inline Input and Setup QoL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace every browser prompt with prefilled in-game input fields, raise board dimensions to 64, and expose the existing turn-action limit as a free numeric value.

**Architecture:** A focused `InlineFieldRules` class owns parsing and validation independently of Unity UI. `UiKit` creates styled uGUI `InputField` controls, while each screen owns its committed value and inline error label. Engine sanitization remains the authoritative boundary.

**Tech Stack:** Unity 6000.5, uGUI `InputField`, C#, NUnit, WebGL.

## Global Constraints

- No browser `window.prompt` or modal validation alert.
- Blank becomes `0` only for fields whose rule explicitly allows zero.
- Board width and height are `5..64`.
- Turn actions retain the current `KActionsPolicy` semantics; `0` means unlimited and positive values accept the full non-negative `Int32` range.
- Existing defaults remain prefilled, including turn actions `3`.
- After every C# edit, run Unity `check_compile_errors`.

---

### Task 1: Pure inline-field validation

**Files:**
- Create: `Assets/HexWars/Presentation/InlineFieldRules.cs`
- Create: `Assets/HexWars/Presentation/InlineFieldRules.cs.meta`
- Create: `Assets/HexWars/Tests/Editor/InlineFieldRulesTests.cs`
- Create: `Assets/HexWars/Tests/Editor/InlineFieldRulesTests.cs.meta`

**Interfaces:**
- Produces: `InlineFieldRules.TryInt(string text, int min, int max, bool blankMeansZero, out int value, out string error)`.

- [ ] **Step 1: Write failing EditMode tests.** Cover valid endpoints, whitespace, blank-with-zero, blank-without-zero, negative input, overflow, and values outside the supplied range.

```csharp
[TestCase("", 0, int.MaxValue, true, true, 0)]
[TestCase("", 5, 64, false, false, 0)]
[TestCase("64", 5, 64, false, true, 64)]
[TestCase("65", 5, 64, false, false, 0)]
public void TryInt_EnforcesFieldRule(string text, int min, int max, bool blankZero,
                                     bool expectedOk, int expected)
{
    bool ok = InlineFieldRules.TryInt(text, min, max, blankZero, out int value, out _);
    Assert.That(ok, Is.EqualTo(expectedOk));
    if (ok) Assert.That(value, Is.EqualTo(expected));
}
```

- [ ] **Step 2: Run the focused EditMode test and confirm it fails because `InlineFieldRules` does not exist.**

Run the Unity EditMode suite filtered to `InlineFieldRulesTests`; expected result: compile failure or failed test before implementation.

- [ ] **Step 3: Implement the pure parser.** Use invariant integer parsing, never clamp user input in this layer, and return concise errors such as `Required`, `Enter a whole number`, and `Use 5–64`.

```csharp
public static bool TryInt(string text, int min, int max, bool blankMeansZero,
                          out int value, out string error)
{
    text = (text ?? "").Trim();
    if (text.Length == 0 && blankMeansZero) text = "0";
    if (text.Length == 0) { value = 0; error = "Required"; return false; }
    if (!int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out value))
    { error = "Enter a whole number"; return false; }
    if (value < min || value > max)
    { error = max == int.MaxValue ? $"Use {min} or more" : $"Use {min}–{max}"; return false; }
    error = "";
    return true;
}
```

- [ ] **Step 4: Run `check_compile_errors` and the focused EditMode tests; expect all parser cases to pass.**

- [ ] **Step 5: Commit.**

```powershell
git add Assets/HexWars/Presentation/InlineFieldRules.cs Assets/HexWars/Presentation/InlineFieldRules.cs.meta Assets/HexWars/Tests/Editor/InlineFieldRulesTests.cs Assets/HexWars/Tests/Editor/InlineFieldRulesTests.cs.meta
git commit -m "feat(ui): add inline field validation rules"
```

### Task 2: Styled uGUI input factory

**Files:**
- Modify: `Assets/HexWars/Presentation/UiKit.cs:229`
- Create: `Assets/HexWars/Tests/Editor/UiKitInputFieldTests.cs`
- Create: `Assets/HexWars/Tests/Editor/UiKitInputFieldTests.cs.meta`

**Interfaces:**
- Consumes: `InlineFieldRules.TryInt(...)`.
- Produces: `UiKit.InputField(...)`, `UiKit.IntField(...)`, and an `InlineIntBinding` with `Commit()` and `Restore()`.

- [ ] **Step 1: Write EditMode tests that build a temporary canvas and assert text is prefilled, numeric fields use integer content type, invalid commits retain typed text plus an error, valid commits invoke the setter, and Escape restores committed text.**

- [ ] **Step 2: Run the focused tests and confirm they fail because the factory methods do not exist.**

- [ ] **Step 3: Add the uGUI factories and binding.** Use the existing `InputBg`, fonts, colors, and `UiKit.SetRect`; wire `onEndEdit` to `Commit`, and expose `Restore` so screen-level keyboard handling can invoke it.

```csharp
public sealed class InlineIntBinding
{
    public InputField Field { get; }
    public Text Error { get; }
    public bool Commit();
    public void Restore();
}
```

- [ ] **Step 4: Keep `PromptInt`, `PromptText`, and `ValueBox` temporarily so the project continues compiling while later tasks migrate their callers. Delete all three only in Task 4 after `rg` confirms no call sites remain.**

- [ ] **Step 5: Run `check_compile_errors` and `UiKitInputFieldTests`; expect all tests to pass.**

- [ ] **Step 6: Commit.**

```powershell
git add Assets/HexWars/Presentation/UiKit.cs Assets/HexWars/Tests/Editor/UiKitInputFieldTests.cs Assets/HexWars/Tests/Editor/UiKitInputFieldTests.cs.meta
git commit -m "feat(ui): add styled inline input fields"
```

### Task 3: Setup form and authoritative bounds

**Files:**
- Modify: `Assets/HexWars/Presentation/SetupForm.cs:90`
- Modify: `engine/HexWars.Engine/Net/GameSetup.cs:50`
- Modify: `engine/HexWars.Engine.Tests/GameSetupSanitizeTests.cs:12`
- Create: `Assets/HexWars/Tests/Editor/SetupFormInputTests.cs`
- Create: `Assets/HexWars/Tests/Editor/SetupFormInputTests.cs.meta`

**Interfaces:**
- Consumes: `UiKit.IntField(...)`.
- Preserves: `GameSetup.TurnActions` and `KActionsPolicy` behavior.

- [ ] **Step 1: Change engine tests first.** Assert width/height `64` pass unchanged, oversized dimensions clamp to `64`, negative turn actions clamp to `0`, and `int.MaxValue` remains unchanged.

```csharp
Assert.That(new GameSetup(GameMode.Annihilation, 64, 64, 0, 7, turnActions: int.MaxValue)
    .Sanitized().TurnActions, Is.EqualTo(int.MaxValue));
```

- [ ] **Step 2: Run the focused engine test; expect failures at the old `24` and `8` caps.**

- [ ] **Step 3: Update `GameSetup.Sanitized()`.** Clamp dimensions to `5..64` and turn actions with `Math.Max(0, TurnActions)`; leave every other safety bound unchanged.

- [ ] **Step 4: Write SetupForm EditMode tests.** Assert all numeric values are prefilled, width/height reject blank, turn actions commit blank as zero, Start is blocked while a field is invalid, and no plus/minus or pace-preset buttons exist.

- [ ] **Step 5: Replace `ValueBox`, `NumberRowIn`, and pace preset cycling with `UiKit.IntField` bindings.** Keep the current defaults and helper copy `0 = whole team / unlimited`. Start calls `Commit()` on every binding and returns early when any binding is invalid.

- [ ] **Step 6: Run the full engine suite, `check_compile_errors`, and SetupForm EditMode tests. Expected: engine tests pass and no browser prompt is reached from setup.**

- [ ] **Step 7: Rebuild the Release engine DLL and copy it to `Assets/HexWars/Plugins/HexWars.Engine.dll`; run `check_compile_errors` again.**

- [ ] **Step 8: Commit.**

```powershell
git add Assets/HexWars/Presentation/SetupForm.cs Assets/HexWars/Tests/Editor/SetupFormInputTests.cs Assets/HexWars/Tests/Editor/SetupFormInputTests.cs.meta engine/HexWars.Engine/Net/GameSetup.cs engine/HexWars.Engine.Tests/GameSetupSanitizeTests.cs
git commit -m "feat(setup): use inline numeric fields and allow 64x64 boards"
```

### Task 4: Inline room code and template name

**Files:**
- Modify: `Assets/HexWars/Presentation/TitleScreen.cs:90`
- Modify: `Assets/HexWars/Presentation/DesignPanel.cs:150`
- Modify: `Assets/HexWars/Presentation/UiKit.cs:240`
- Delete: `Assets/HexWars/Plugins/HexWarsPrompt.jslib`
- Delete: `Assets/HexWars/Plugins/HexWarsPrompt.jslib.meta`
- Create: `Assets/HexWars/Tests/Editor/InlineTextFlowsTests.cs`
- Create: `Assets/HexWars/Tests/Editor/InlineTextFlowsTests.cs.meta`

**Interfaces:**
- Consumes: `UiKit.InputField(...)`.
- Preserves: `UnitTemplate.Sanitize(string)` and existing room-code normalization.

- [ ] **Step 1: Write EditMode tests.** Assert Join by Code edits inline and rejects empty/malformed codes beside the field; assert the Designer pre-fills and sanitizes a name without a modal prompt; assert Enter submits and Escape restores.

- [ ] **Step 2: Run the focused tests and confirm the current prompt-driven flows fail them.**

- [ ] **Step 3: Add a compact room-code field beside Join.** Normalize with `Trim().ToUpperInvariant()`, reuse the existing validity check, and keep errors inline.

- [ ] **Step 4: Replace the Designer's name button/prompt with a persistent text field.** Commit its text in `OnCreate`, pass the sanitized value through the existing `CreateUnit`, and retain authoritative online confirmation behavior.

- [ ] **Step 5: Delete `PromptText`, `PromptInt`, `ValueBox`, their WebGL imports, and `HexWarsPrompt.jslib` after `rg -n 'PromptText|PromptInt|ValueBox|HexWarsPrompt' Assets` returns only their definitions/plugin files.**

- [ ] **Step 6: Run `check_compile_errors`, all presentation EditMode tests, and a WebGL build compile. Manually verify desktop typing, mobile soft keyboard, Enter, Escape, and focus-loss behavior.**

- [ ] **Step 7: Commit.**

```powershell
git add Assets/HexWars/Presentation/TitleScreen.cs Assets/HexWars/Presentation/DesignPanel.cs Assets/HexWars/Presentation/UiKit.cs Assets/HexWars/Tests/Editor/InlineTextFlowsTests.cs Assets/HexWars/Tests/Editor/InlineTextFlowsTests.cs.meta Assets/HexWars/Plugins/HexWarsPrompt.jslib Assets/HexWars/Plugins/HexWarsPrompt.jslib.meta
git commit -m "feat(ui): replace browser prompts with inline text entry"
```

### Task 5: Integrated verification

**Files:**
- Modify only if verification exposes a defect in files already listed above.

- [ ] **Step 1: Run the complete engine test suite.** Expected: zero failures.
- [ ] **Step 2: Run Unity `check_compile_errors` and the complete HexWars EditMode suite.** Expected: zero compile errors and zero failed tests.
- [ ] **Step 3: In Play Mode, create a local AI game with `64x64`, edit every numeric field, use blank turn actions, reject blank width, name a unit, return to title, and join by code. Inspect Unity logs afterward.**
- [ ] **Step 4: Build WebGL and repeat room/name/number entry in a browser. Confirm no native browser prompt or alert appears.**
- [ ] **Step 5: Run `git diff --check` and commit any verification-only corrections with a narrowly scoped message.**
