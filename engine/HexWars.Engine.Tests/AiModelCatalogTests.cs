using System;
using HexWars.Engine.AI;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class AiModelCatalogTests
    {
        static AiModelDefinition Model(string id) => new AiModelDefinition
        {
            id = id, label = id, package = id, kind = "trained", contract_version = "tactical-v3",
            checkpoint_sha256 = new string('a', 64), encoding_hash = new string('b', 64),
            capacity_hash = new string('c', 64)
        };

        static AiModelCatalog Catalog() => new AiModelCatalog
        {
            default_difficulty_id = "normal", models = new[] { Model("first"), Model("second") },
            difficulties = new[] {
                new AiDifficultyDefinition { id="normal", label="Normal", model_id="first" },
                new AiDifficultyDefinition { id="hard", label="Hard", model_id="second" }
            }
        };

        [Test]
        public void ChangedDefaultResolvesEveryUnspecifiedSelectionWithoutChangingExplicitSelection()
        {
            var catalog = Catalog();
            Assert.That(catalog.Resolve().id, Is.EqualTo("first"));
            catalog.default_difficulty_id = "hard";
            Assert.That(catalog.Resolve().id, Is.EqualTo("second"));
            Assert.That(catalog.Resolve("first").id, Is.EqualTo("first"));
            Assert.Throws<ArgumentException>(() => catalog.Resolve("missing"));
        }

        [Test]
        public void InvalidDefaultAndDuplicateIdsCannotResolvePlausibleFallback()
        {
            var catalog = Catalog();
            catalog.default_difficulty_id = "missing";
            Assert.Throws<ArgumentException>(() => catalog.Resolve("first"));
            catalog.default_difficulty_id = "normal";
            catalog.models[1].id = "first";
            Assert.Throws<ArgumentException>(() => catalog.Resolve());
        }

        [Test]
        public void DifficultyLabelsAndModelMappingsAreIndependentOfModelNames()
        {
            var catalog = Catalog();
            catalog.difficulties[0].label = "Recruit";
            catalog.difficulties[0].model_id = "second";
            Assert.That(catalog.ResolveDifficulty().label, Is.EqualTo("Recruit"));
            Assert.That(catalog.Resolve().id, Is.EqualTo("second"));
            Assert.That(catalog.ResolveDifficulty("hard").model_id, Is.EqualTo("second"));
            Assert.Throws<ArgumentException>(() => catalog.ResolveDifficulty("first"));
        }

        [Test]
        public void MissingModelMappingAndDuplicateDifficultiesFailClosed()
        {
            var catalog = Catalog();
            catalog.difficulties[0].model_id = "missing";
            Assert.Throws<ArgumentException>(() => catalog.Validate());
            catalog.difficulties[0].model_id = "first";
            catalog.difficulties[1].id = "normal";
            Assert.Throws<ArgumentException>(() => catalog.Validate());
        }

        [TestCase("../secret")]
        [TestCase("/absolute")]
        [TestCase("C:/models")]
        [TestCase("sub\\model")]
        [TestCase("sub/./model")]
        [TestCase("sub//model")]
        public void ModelPackagesCannotEscapeTheirConfiguredRoot(string package)
        {
            var catalog = Catalog();
            catalog.models[0].package = package;
            Assert.Throws<ArgumentException>(() => catalog.Validate());
        }

        [Test]
        public void ScriptedModelCannotClaimACheckpoint()
        {
            var catalog = Catalog();
            catalog.models[0].kind = "greedy";
            Assert.Throws<ArgumentException>(() => catalog.Validate());
            catalog.models[0] = new AiModelDefinition { id = "first", label = "Greedy", kind = "greedy" };
            Assert.DoesNotThrow(() => catalog.Validate());
            Assert.That(catalog.Resolve().IsTrained, Is.False);
        }

        [TestCase("a")]
        [TestCase("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")]
        [TestCase("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n")]
        public void CheckpointPinsRequireExactLowercaseDigest(string digest)
        {
            var catalog = Catalog();
            catalog.models[0].checkpoint_sha256 = digest;
            Assert.Throws<ArgumentException>(() => catalog.Validate());
        }

        [Test]
        public void RevisionBindsExactConfigurationBytes()
        {
            Assert.That(AiModelCatalog.RevisionOf("abc"), Is.EqualTo(
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"));
            Assert.That(AiModelCatalog.RevisionOf("abc\n"), Is.Not.EqualTo(AiModelCatalog.RevisionOf("abc")));
        }
    }
}
