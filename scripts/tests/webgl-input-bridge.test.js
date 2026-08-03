const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

function loadBridge() {
  const listeners = {};
  const messages = [];
  const input = {
    id: "",
    type: "",
    inputMode: "",
    value: "",
    style: {},
    addEventListener(kind, handler) {
      listeners[kind] = handler;
    },
    setAttribute(name, value) {
      this[name] = value;
    },
    focus() {
      this.focused = true;
    },
    blur() {
      this.focused = false;
      listeners.blur?.({});
    },
    setSelectionRange() {},
  };
  const body = {
    appendChild(element) {
      this.child = element;
    },
  };
  const document = {
    body,
    createElement(tag) {
      assert.equal(tag, "input");
      return input;
    },
    getElementById(id) {
      return body.child?.id === id ? body.child : null;
    },
  };
  const library = {};
  const context = {
    LibraryManager: { library },
    UTF8ToString: (value) => value,
    SendMessage: (...args) => messages.push(args),
    document,
    window: {},
    mergeInto(target, additions) {
      Object.assign(target, additions);
    },
  };

  const source = fs.readFileSync(
    path.join(__dirname, "..", "..", "Assets", "HexWars", "Plugins", "HexWarsWebInput.jslib"),
    "utf8",
  );
  vm.runInNewContext(source, context);
  return { library, listeners, messages, input };
}

test("native input takes browser focus and forwards ordinary characters to Unity", () => {
  const { library, listeners, messages, input } = loadBridge();

  library.HexWarsWebInputBegin(17, "Scout t", 0);
  input.value = "Scout team";
  listeners.input({});

  assert.equal(input.focused, true);
  assert.equal(input.type, "text");
  assert.deepEqual(messages, [
    ["HexWarsWebInputHub", "OnNativeInput", JSON.stringify({
      id: 17,
      kind: "change",
      value: "Scout team",
    })],
  ]);
});

test("numeric fields request a numeric keyboard and submit without leaking the key to Chrome", () => {
  const { library, listeners, messages, input } = loadBridge();

  library.HexWarsWebInputBegin(9, "24", 1);
  input.value = "6t4";
  listeners.input({});
  assert.equal(input.value, "64");

  let prevented = false;
  input.value = "64";
  listeners.keydown({
    key: "Enter",
    preventDefault() {
      prevented = true;
    },
    stopPropagation() {},
  });

  assert.equal(input.inputMode, "numeric");
  assert.equal(prevented, true);
  assert.deepEqual(messages.at(-1), [
    "HexWarsWebInputHub",
    "OnNativeInput",
    JSON.stringify({ id: 9, kind: "submit", value: "64" }),
  ]);
});

test("escape restores the value from the start of the edit", () => {
  const { library, listeners, messages, input } = loadBridge();

  library.HexWarsWebInputBegin(3, "Original", 0);
  input.value = "Discard me";
  listeners.keydown({
    key: "Escape",
    preventDefault() {},
    stopPropagation() {},
  });

  assert.deepEqual(messages.at(-1), [
    "HexWarsWebInputHub",
    "OnNativeInput",
    JSON.stringify({ id: 3, kind: "cancel", value: "Original" }),
  ]);
});

test("Unity can end native entry without committing a second time", () => {
  const { library, listeners, messages, input } = loadBridge();

  library.HexWarsWebInputBegin(5, "Working", 0);
  library.HexWarsWebInputEnd(5);
  listeners.blur({});

  assert.equal(input.focused, false);
  assert.deepEqual(messages, []);
});
