using HexWars.Presentation;
using NUnit.Framework;

namespace HexWars.Presentation.Tests
{
    /// <summary>Pure snap-rule coverage for the playback speed slider (spec: comfort controls — speed
    /// past 4x). Mirrors <see cref="ModelArenaIdentityTests"/>'s idiom of testing the extracted static
    /// math directly rather than driving OnGUI.</summary>
    public sealed class PauseToggleTests
    {
        [TestCase(0.1f, 0.25f)]
        [TestCase(0.25f, 0.25f)]
        [TestCase(0.6f, 0.5f)]
        [TestCase(1.0f, 1.0f)]
        [TestCase(1.13f, 1.25f)]
        [TestCase(4.0f, 4.0f)]
        public void SnapSpeed_AtOrBelow4x_SnapsToQuarterSteps(float input, float expected)
        {
            Assert.That(PauseToggle.SnapSpeed(input), Is.EqualTo(expected).Within(0.001f));
        }

        [TestCase(4.01f, 4f)]
        [TestCase(4.4f, 4f)]
        [TestCase(4.6f, 5f)]
        [TestCase(8.5f, 8f)] // Mathf.Round is round-half-to-even: 8 is the even neighbor
        [TestCase(15.6f, 16f)]
        [TestCase(16.0f, 16f)]
        public void SnapSpeed_Above4x_SnapsToWholeNumberSteps(float input, float expected)
        {
            Assert.That(PauseToggle.SnapSpeed(input), Is.EqualTo(expected).Within(0.001f));
        }

        [Test]
        public void SnapSpeed_ClampsBelowMinimumUpToQuarterX()
        {
            Assert.That(PauseToggle.SnapSpeed(-5f), Is.EqualTo(0.25f));
            Assert.That(PauseToggle.SnapSpeed(0f), Is.EqualTo(0.25f));
        }

        [Test]
        public void SnapSpeed_ClampsAboveMaximumDownTo16x()
        {
            Assert.That(PauseToggle.SnapSpeed(100f), Is.EqualTo(16f));
            Assert.That(PauseToggle.SnapSpeed(16.4f), Is.EqualTo(16f));
        }
    }
}
