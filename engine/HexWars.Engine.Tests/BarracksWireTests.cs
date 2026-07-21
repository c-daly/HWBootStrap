using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class BarracksWireTests
    {
        [Test]
        public void WriteAndRead_RoundTripSanitizedNameAndAllNineStats()
        {
            var stats = new UnitStats(1, -2, 3, -4, 5, -6, 7, -8, 9);
            var payload = BarracksWire.Write(new[]
            {
                new UnitTemplate("  O'Brien Mk-2!?  ", stats),
            });

            var result = BarracksWire.Read(payload);

            Assert.That(payload, Does.StartWith("V1\n"));
            Assert.That(payload, Does.Not.Contain("O'Brien"));
            Assert.That(result, Has.Count.EqualTo(1));
            Assert.That(result[0].Name, Is.EqualTo("O'Brien Mk-2"));
            AssertStats(result[0].Stats, 1, -2, 3, -4, 5, -6, 7, -8, 9);
        }

        [Test]
        public void WriteAndRead_EmptyCatalogRoundTrips()
        {
            var payload = BarracksWire.Write(Array.Empty<UnitTemplate>());

            Assert.That(payload, Is.EqualTo("V1"));
            Assert.That(BarracksWire.Read(payload), Is.Empty);
        }

        [Test]
        public void Write_NormalizesAndTruncatesCatalogToSixtyFourEntries()
        {
            var source = Enumerable.Range(1, 70)
                .Select(i => new UnitTemplate("Unit " + i, new UnitStats(i, 1, 2, 3, 4, 5, 6, 7, 8)))
                .Concat(new[]
                {
                    new UnitTemplate("Unit 1", new UnitStats(1, 1, 2, 3, 4, 5, 6, 7, 8)),
                    new UnitTemplate("Invalid", new UnitStats(0, 1, 2, 3, 4, 5, 6, 7, 8)),
                })
                .ToArray();

            var result = BarracksWire.Read(BarracksWire.Write(source));

            Assert.That(result, Has.Count.EqualTo(BarracksCatalog.ProtocolMaximumTemplates));
            Assert.That(result[0].Name, Is.EqualTo("Unit 1"));
            Assert.That(result[63].Name, Is.EqualTo("Unit 64"));
        }

        [Test]
        public void Read_NormalizesNamesRejectsInvalidHealthAndRemovesExactDuplicates()
        {
            string rawName = Convert.ToBase64String(Encoding.UTF8.GetBytes("  Hearts: ♥ Alpha_One!  "));
            string normalizedName = Convert.ToBase64String(Encoding.UTF8.GetBytes("Hearts  AlphaOne"));
            string payload = "V1\n"
                + rawName + "|2|1|2|3|4|5|6|7|8\n"
                + normalizedName + "|2|1|2|3|4|5|6|7|8\n"
                + rawName + "|0|1|2|3|4|5|6|7|8";

            var result = BarracksWire.Read(payload);

            Assert.That(result, Has.Count.EqualTo(1));
            Assert.That(result[0].Name, Is.EqualTo("Hearts  AlphaOne"));
        }

        private static IEnumerable<TestCaseData> MalformedPayloads()
        {
            yield return new TestCaseData("").SetName("Read_RejectsMissingVersion");
            yield return new TestCaseData("V2").SetName("Read_RejectsUnknownVersion");
            yield return new TestCaseData("V1\nnot-base64!|1|2|3|4|5|6|7|8|9").SetName("Read_RejectsInvalidBase64Name");
            yield return new TestCaseData("V1\nQQ==|1|2|3|4|5|6|7|8").SetName("Read_RejectsTruncatedRecord");
            yield return new TestCaseData("V1\nQQ==|1|2|3|4|5|6|7|8|9|10").SetName("Read_RejectsRecordWithExtraField");
            yield return new TestCaseData("V1\nQQ==|1|2|3|four|5|6|7|8|9").SetName("Read_RejectsNonNumericStat");
            yield return new TestCaseData("V1\n").SetName("Read_RejectsEmptyRecord");
        }

        [TestCaseSource(nameof(MalformedPayloads))]
        public void Read_RejectsMalformedPayload(string payload)
        {
            Assert.That(() => BarracksWire.Read(payload), Throws.TypeOf<FormatException>());
        }

        [Test]
        public void Read_RejectsPayloadLargerThanThirtyTwoKiBBeforeParsingRecords()
        {
            string payload = "V1\n" + new string('x', BarracksWire.MaximumPayloadBytes);

            Assert.That(Encoding.UTF8.GetByteCount(payload), Is.GreaterThan(BarracksWire.MaximumPayloadBytes));
            Assert.That(() => BarracksWire.Read(payload), Throws.TypeOf<FormatException>());
        }

        [Test]
        public void Read_EnforcesThirtyTwoKiBLimitInUtf8BytesNotCharacters()
        {
            string payload = "V1\n" + new string('\u00e9', BarracksWire.MaximumPayloadBytes / 2);

            Assert.That(payload.Length, Is.LessThanOrEqualTo(BarracksWire.MaximumPayloadBytes));
            Assert.That(Encoding.UTF8.GetByteCount(payload), Is.GreaterThan(BarracksWire.MaximumPayloadBytes));
            Assert.That(() => BarracksWire.Read(payload), Throws.TypeOf<FormatException>());
        }

        private static void AssertStats(UnitStats actual, params int[] expected)
        {
            Assert.That(new[]
            {
                actual.Health,
                actual.Damage,
                actual.Defense,
                actual.Movement,
                actual.VerticalMovement,
                actual.Range,
                actual.RangeArc,
                actual.Vision,
                actual.VisionArc,
            }, Is.EqualTo(expected));
        }
    }
}
