using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

namespace HexWars.Presentation
{
    /// <summary>
    /// Mirrors a legacy uGUI <see cref="InputField"/> through a native browser input while selected.
    /// The browser therefore treats ordinary keys as text entry instead of extension shortcuts.
    /// </summary>
    [RequireComponent(typeof(InputField))]
    public sealed class WebGlInputBridge : MonoBehaviour, ISelectHandler, IDeselectHandler
    {
        InputField _field;
        int _bridgeId;
        string _textAtFocus = string.Empty;
        bool _nativeActive;

        public InputField Field => _field != null ? _field : (_field = GetComponent<InputField>());
        internal event Action CancelRequested;

        internal void Bind(InputField field) => _field = field;

        public void OnSelect(BaseEventData eventData)
        {
            if (Field == null || !Field.interactable || Field.readOnly) return;
            EnsureRegistered();
            WebGlInputHub.Instance.Activate(_bridgeId);
            _textAtFocus = Field.text ?? string.Empty;
            _nativeActive = true;
#if UNITY_WEBGL && !UNITY_EDITOR
            WebGLInput.captureAllKeyboardInput = false;
            HexWarsWebInputBegin(
                _bridgeId,
                _textAtFocus,
                Field.contentType == InputField.ContentType.IntegerNumber ? 1 : 0);
#endif
        }

        public void OnDeselect(BaseEventData eventData)
        {
            if (!_nativeActive) return;
            EndNativeInput();
        }

        void OnDestroy()
        {
            if (_nativeActive) EndNativeInput();
            if (_bridgeId != 0 && WebGlInputHub.TryGetExisting(out var hub))
                hub.Unregister(_bridgeId, this);
        }

        internal void Receive(string kind, string value)
        {
            if (Field == null || !_nativeActive) return;
            switch (kind)
            {
                case "change":
                    Field.text = Normalize(value);
                    break;
                case "submit":
                    Field.text = Normalize(value);
                    MarkNativeFinished();
                    Field.onSubmit.Invoke(Field.text);
                    Field.DeactivateInputField();
                    ClearSelection();
                    break;
                case "cancel":
                    MarkNativeFinished();
                    UiKit.MarkInputEscapeHandled();
                    if (CancelRequested != null)
                        CancelRequested.Invoke();
                    else
                    {
                        Field.SetTextWithoutNotify(_textAtFocus);
                        Field.DeactivateInputField();
                    }
                    ClearSelection();
                    break;
                case "blur":
                    Field.text = Normalize(value);
                    MarkNativeFinished();
                    Field.DeactivateInputField();
                    ClearSelection();
                    break;
            }
        }

        void EnsureRegistered()
        {
            if (_bridgeId == 0) _bridgeId = WebGlInputHub.Instance.Register(this);
        }

        void EndNativeInput()
        {
#if UNITY_WEBGL && !UNITY_EDITOR
            HexWarsWebInputEnd(_bridgeId);
#endif
            MarkNativeFinished();
        }

        void MarkNativeFinished()
        {
            _nativeActive = false;
            if (_bridgeId != 0 && WebGlInputHub.TryGetExisting(out var hub))
                hub.Finish(_bridgeId);
#if UNITY_WEBGL && !UNITY_EDITOR
            WebGLInput.captureAllKeyboardInput = true;
#endif
        }

        internal void Superseded()
        {
            _nativeActive = false;
        }

        string Normalize(string value)
        {
            string normalized = value ?? string.Empty;
            if (Field.contentType == InputField.ContentType.IntegerNumber)
            {
                var digits = new System.Text.StringBuilder(normalized.Length);
                for (int i = 0; i < normalized.Length; i++)
                {
                    char character = normalized[i];
                    if (character >= '0' && character <= '9') digits.Append(character);
                    else if (character == '-' && i == 0) digits.Append(character);
                }
                normalized = digits.ToString();
            }
            if (Field.characterLimit > 0 && normalized.Length > Field.characterLimit)
                normalized = normalized.Substring(0, Field.characterLimit);
            return normalized;
        }

        void ClearSelection()
        {
            var eventSystem = EventSystem.current ?? FindAnyObjectByType<EventSystem>();
            if (eventSystem != null && eventSystem.currentSelectedGameObject == gameObject)
                eventSystem.SetSelectedGameObject(null);
        }

#if UNITY_WEBGL && !UNITY_EDITOR
        [DllImport("__Internal")]
        static extern void HexWarsWebInputBegin(int id, string initial, int numeric);

        [DllImport("__Internal")]
        static extern void HexWarsWebInputEnd(int id);
#endif
    }

    /// <summary>Single named receiver used by the WebGL SendMessage bridge.</summary>
    public sealed class WebGlInputHub : MonoBehaviour
    {
        const string HubName = "HexWarsWebInputHub";
        static WebGlInputHub _instance;

        readonly Dictionary<int, WebGlInputBridge> _bridges =
            new Dictionary<int, WebGlInputBridge>();
        int _nextId = 1;
        int _activeId;

        internal static WebGlInputHub Instance
        {
            get
            {
                if (_instance != null) return _instance;
                _instance = FindAnyObjectByType<WebGlInputHub>();
                if (_instance != null) return _instance;
                var go = new GameObject(HubName);
                _instance = go.AddComponent<WebGlInputHub>();
                if (Application.isPlaying) DontDestroyOnLoad(go);
                return _instance;
            }
        }

        internal static bool TryGetExisting(out WebGlInputHub hub)
        {
            hub = _instance;
            return hub != null;
        }

        internal int Register(WebGlInputBridge bridge)
        {
            int id = _nextId++;
            _bridges[id] = bridge;
            return id;
        }

        internal void Activate(int id)
        {
            if (_activeId != 0 && _activeId != id &&
                _bridges.TryGetValue(_activeId, out var previous) && previous != null)
                previous.Superseded();
            _activeId = id;
        }

        internal void Finish(int id)
        {
            if (_activeId == id) _activeId = 0;
        }

        internal void Unregister(int id, WebGlInputBridge bridge)
        {
            if (_bridges.TryGetValue(id, out var registered) && registered == bridge)
                _bridges.Remove(id);
            Finish(id);
        }

        /// <summary>Entry point called by the native browser input through Unity SendMessage.</summary>
        public void OnNativeInput(string json)
        {
            if (string.IsNullOrEmpty(json)) return;
            NativeInputMessage message;
            try
            {
                message = JsonUtility.FromJson<NativeInputMessage>(json);
            }
            catch (ArgumentException)
            {
                return;
            }
            if (message == null || message.id <= 0 || message.id != _activeId ||
                string.IsNullOrEmpty(message.kind)) return;
            if (_bridges.TryGetValue(message.id, out var bridge) && bridge != null)
                bridge.Receive(message.kind, message.value);
        }

        void OnDestroy()
        {
            if (_instance == this) _instance = null;
        }

        [Serializable]
        sealed class NativeInputMessage
        {
            public int id;
            public string kind;
            public string value;
        }
    }
}
