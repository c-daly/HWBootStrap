using NUnit.Framework;

namespace HexWars.Presentation.Tests
{
    public class InlineFieldRulesTests
    {
        [TestCase("5", 5, 64, false, true, 5)]
        [TestCase("64", 5, 64, false, true, 64)]
        [TestCase("  12  ", 5, 64, false, true, 12)]
        [TestCase("", 0, int.MaxValue, true, true, 0)]
        [TestCase("   ", 0, int.MaxValue, true, true, 0)]
        [TestCase("", 5, 64, false, false, 0)]
        [TestCase("-1", 0, int.MaxValue, true, false, 0)]
        [TestCase("2147483648", 0, int.MaxValue, true, false, 0)]
        [TestCase("4", 5, 64, false, false, 0)]
        [TestCase("65", 5, 64, false, false, 0)]
        public void TryInt_EnforcesFieldRule(
            string text,
            int min,
            int max,
            bool blankMeansZero,
            bool expectedOk,
            int expected)
        {
            bool ok = InlineFieldRules.TryInt(
                text,
                min,
                max,
                blankMeansZero,
                out int value,
                out _);

            Assert.That(ok, Is.EqualTo(expectedOk));
            if (ok)
                Assert.That(value, Is.EqualTo(expected));
        }

        [TestCase("", 5, 64, false, "Required")]
        [TestCase("not a number", 5, 64, false, "Enter a whole number")]
        [TestCase("2147483648", 0, int.MaxValue, true, "Enter a whole number")]
        [TestCase("4", 5, 64, false, "Use 5–64")]
        [TestCase("-1", 0, int.MaxValue, true, "Use 0 or more")]
        public void TryInt_ExplainsInvalidInput(
            string text,
            int min,
            int max,
            bool blankMeansZero,
            string expectedError)
        {
            bool ok = InlineFieldRules.TryInt(
                text,
                min,
                max,
                blankMeansZero,
                out _,
                out string error);

            Assert.That(ok, Is.False);
            Assert.That(error, Is.EqualTo(expectedError));
        }
    }
}
