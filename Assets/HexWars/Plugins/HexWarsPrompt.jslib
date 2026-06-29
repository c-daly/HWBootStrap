// Bridges to the browser's native prompt() so the soft keyboard works for text entry on mobile WebGL,
// where Unity's own InputField cannot summon a keyboard. Returns the typed string (or the current value
// if the user cancels). Synchronous — fine for the lobby, which isn't running a game loop yet.
mergeInto(LibraryManager.library, {
  HexWarsPrompt: function (messagePtr, currentPtr) {
    var message = UTF8ToString(messagePtr);
    var current = UTF8ToString(currentPtr);
    var result = window.prompt(message, current);
    if (result === null) result = current;
    var size = lengthBytesUTF8(result) + 1;
    var buffer = _malloc(size);
    stringToUTF8(result, buffer, size);
    return buffer;
  }
});
