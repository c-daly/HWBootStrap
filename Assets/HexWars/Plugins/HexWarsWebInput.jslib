// A Unity canvas InputField does not become the browser's active text control. Browser extensions
// therefore see ordinary shortcut keys (for example Vimium's bare "t") instead of typed text.
// Keep a real, nearly invisible DOM input focused while a HexWars field is selected and mirror its
// value back through one Unity receiver.
mergeInto(LibraryManager.library, {
  HexWarsWebInputBegin: function (id, initialPtr, numeric) {
    var initial = UTF8ToString(initialPtr);
    var state = window.__hexWarsWebInputState;
    if (!state) {
      state = {
        active: null,
        send: function (kind, value) {
          if (!state.active) return;
          SendMessage("HexWarsWebInputHub", "OnNativeInput", JSON.stringify({
            id: state.active.id,
            kind: kind,
            value: value
          }));
        }
      };

      var input = document.createElement("input");
      input.id = "hexwars-webgl-input";
      input.type = "text";
      input.autocomplete = "off";
      input.autocapitalize = "none";
      input.spellcheck = false;
      input.setAttribute("aria-hidden", "true");
      input.style.position = "fixed";
      input.style.left = "0";
      input.style.top = "0";
      input.style.width = "1px";
      input.style.height = "1px";
      input.style.opacity = "0.01";
      input.style.pointerEvents = "none";
      input.style.zIndex = "-1";

      input.addEventListener("input", function () {
        if (state.active && state.active.numeric) {
          var sign = input.value.charAt(0) === "-" ? "-" : "";
          input.value = sign + input.value.replace(/[^0-9]/g, "");
        }
        state.send("change", input.value);
      });
      input.addEventListener("keydown", function (event) {
        if (!state.active) return;
        if (event.key === "Enter") {
          event.preventDefault();
          event.stopPropagation();
          state.send("submit", input.value);
          state.active = null;
          input.blur();
        } else if (event.key === "Escape") {
          event.preventDefault();
          event.stopPropagation();
          state.send("cancel", state.active.initial);
          state.active = null;
          input.blur();
        }
      });
      input.addEventListener("blur", function () {
        if (!state.active) return;
        state.send("blur", input.value);
        state.active = null;
      });
      document.body.appendChild(input);
      state.input = input;
      window.__hexWarsWebInputState = state;
    }

    state.active = { id: id, initial: initial, numeric: numeric !== 0 };
    state.input.inputMode = numeric ? "numeric" : "text";
    state.input.value = initial;
    try {
      state.input.focus({ preventScroll: true });
    } catch (_) {
      state.input.focus();
    }
    try {
      state.input.setSelectionRange(initial.length, initial.length);
    } catch (_) {
      // Some mobile browsers do not expose a selection range while opening their keyboard.
    }
  },

  HexWarsWebInputEnd: function (id) {
    var state = window.__hexWarsWebInputState;
    if (!state || !state.active || state.active.id !== id) return;
    state.active = null;
    state.input.blur();
  }
});
