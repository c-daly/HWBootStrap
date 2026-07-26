using System;
using HexWars.Presentation;
using NUnit.Framework;

namespace HexWars.Presentation.Tests
{
    /// <summary>Covers the focus-gate seam added for the batch fix: 13 direct
    /// <c>Keyboard.current</c>/<c>Mouse.current</c> polling sites across CameraRig, PauseToggle,
    /// EscapeMenu, TitleScreen, SetupForm and DesignPanel were reacting to device state regardless of
    /// application focus. <see cref="DeviceInput.Allowed"/> is a bare property wrapping
    /// <see cref="UnityEngine.Application.isFocused"/>, which itself can't be flipped from an EditMode
    /// test — <see cref="DeviceInput.FocusProbe"/> is the injected seam that makes the gate testable.</summary>
    public sealed class DeviceInputTests
    {
        Func<bool> _originalProbe;

        [SetUp]
        public void SetUp() => _originalProbe = DeviceInput.FocusProbe;

        [TearDown]
        public void TearDown() => DeviceInput.FocusProbe = _originalProbe;

        [Test]
        public void Allowed_FollowsFocusProbe_WhenUnfocused()
        {
            DeviceInput.FocusProbe = () => false;

            Assert.That(DeviceInput.Allowed, Is.False);
        }

        [Test]
        public void Allowed_FollowsFocusProbe_WhenFocused()
        {
            DeviceInput.FocusProbe = () => true;

            Assert.That(DeviceInput.Allowed, Is.True);
        }
    }
}
