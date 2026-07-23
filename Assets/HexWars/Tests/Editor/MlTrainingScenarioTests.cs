using System;
using System.IO;
using System.Linq;
using HexWars.Presentation;
using HexWars.Presentation.EditorTools.MlLab;
using NUnit.Framework;

namespace HexWars.Presentation.Tests
{
    public sealed class MlTrainingScenarioTests
    {
        static readonly string BuiltInLibraryPath =
            Path.Combine("python", "config", "training-game-templates.json");
        string _scratch;
        string _projectRoot;
        MlTrainingScenario _scenario;

        [SetUp]
        public void SetUp()
        {
            _scratch = Path.Combine(
                Path.GetTempPath(), "hexwars-unity-scenarios-" + Guid.NewGuid().ToString("N"));
            _projectRoot = Path.Combine(_scratch, "project");
            Directory.CreateDirectory(_projectRoot);
            _scenario = MlTrainingScenarioLibrary.Load(BuiltInLibraryPath)
                .Filter(MlEnvironmentContract.AdaptiveV1)
                .First();
        }

        [TearDown]
        public void TearDown()
        {
            if (Directory.Exists(_scratch)) Directory.Delete(_scratch, recursive: true);
        }

        [Test]
        public void Library_LoadsAndFiltersBuiltInTemplates()
        {
            var library = MlTrainingScenarioLibrary.Load(BuiltInLibraryPath);

            Assert.That(library.Filter(MlEnvironmentContract.AdaptiveV1)
                .Select(item => item.Id), Is.EqualTo(new[]
                {
                    "adaptive-standard",
                    "adaptive-long-battle",
                    "adaptive-large-battle",
                }));
        }

        [Test]
        public void Validation_RejectsInvalidBoardRulesAndEpisodeBoundaries()
        {
            _scenario.Board.Width = 0;
            _scenario.Rules.RoundCap = 0;
            _scenario.Episode.MaxSteps = 0;

            var errors = _scenario.Validate();

            Assert.That(errors, Has.Some.Contains("board width"));
            Assert.That(errors, Has.Some.Contains("round cap"));
            Assert.That(errors, Has.Some.Contains("max steps"));
        }

        [Test]
        public void Library_RejectsDuplicateTemplateIds()
        {
            string path = CopyLibrary(json =>
                json.Replace("\"id\": \"tactical-long-battle\"",
                    "\"id\": \"tactical-standard\""));

            var error = Assert.Throws<InvalidDataException>(
                () => MlTrainingScenarioLibrary.Load(path));

            Assert.That(error.Message, Does.Contain("duplicate template id"));
        }

        [Test]
        public void Library_RejectsTemplateIdThatDoesNotMatchEnvironment()
        {
            string path = CopyLibrary(json =>
                json.Replace("\"id\": \"adaptive-standard\"",
                    "\"id\": \"wrong-standard\""));

            var error = Assert.Throws<InvalidDataException>(
                () => MlTrainingScenarioLibrary.Load(path));

            Assert.That(error.Message, Does.Contain("does not match its environment"));
        }

        [Test]
        public void SessionWriter_RoundTripsResolvedScenario()
        {
            string path = MlTrainingScenarioStore.WriteSessionScenario(_projectRoot, _scenario);

            Assert.That(path, Does.StartWith(Path.Combine(_projectRoot, "Library")));
            Assert.That(
                MlTrainingScenarioFile.Load(path).Board.Width,
                Is.EqualTo(_scenario.Board.Width));
        }

        [Test]
        public void SessionWriter_AtomicallyReplacesExistingScenario()
        {
            string path = MlTrainingScenarioStore.WriteSessionScenario(_projectRoot, _scenario);
            _scenario.Board.Width = 15;

            string replacementPath =
                MlTrainingScenarioStore.WriteSessionScenario(_projectRoot, _scenario);

            Assert.That(replacementPath, Is.EqualTo(path));
            Assert.That(MlTrainingScenarioFile.Load(path).Board.Width, Is.EqualTo(15));
            Assert.That(File.Exists(path + ".tmp"), Is.False);
        }

        [Test]
        public void SessionWriter_RejectsInvalidScenarioWithoutReplacingExistingCopy()
        {
            string path = MlTrainingScenarioStore.WriteSessionScenario(_projectRoot, _scenario);
            _scenario.Board.Width = 0;

            Assert.Throws<InvalidDataException>(
                () => MlTrainingScenarioStore.WriteSessionScenario(_projectRoot, _scenario));
            Assert.That(MlTrainingScenarioFile.Load(path).Board.Width, Is.EqualTo(13));
            Assert.That(_scenario.Board.Width, Is.Zero);
        }

        [Test]
        public void SaveAsTemplate_RefusesOverwriteWithoutExplicitConfirmation()
        {
            string libraryPath = CopyLibrary(json => json);
            _scenario.Name = "Changed";

            Assert.Throws<InvalidOperationException>(() =>
                MlTrainingScenarioStore.SaveAsTemplate(libraryPath, _scenario, overwrite: false));

            var unchanged = MlTrainingScenarioLibrary.Load(libraryPath)
                .Filter(MlEnvironmentContract.AdaptiveV1)
                .Single(item => item.Id == _scenario.Id);
            Assert.That(unchanged.Name, Is.EqualTo("Standard"));
            Assert.That(_scenario.Name, Is.EqualTo("Changed"));
        }

        [Test]
        public void SaveAsTemplate_OverwritesOnlyAfterConfirmation()
        {
            string libraryPath = CopyLibrary(json => json);
            _scenario.Name = "Changed";

            MlTrainingScenarioStore.SaveAsTemplate(libraryPath, _scenario, overwrite: true);

            var saved = MlTrainingScenarioLibrary.Load(libraryPath)
                .Filter(MlEnvironmentContract.AdaptiveV1)
                .Single(item => item.Id == _scenario.Id);
            Assert.That(saved.Name, Is.EqualTo("Changed"));
            Assert.That(File.Exists(libraryPath + ".tmp"), Is.False);
        }

        string CopyLibrary(Func<string, string> transform)
        {
            string path = Path.Combine(_scratch, "training-game-templates.json");
            File.WriteAllText(path, transform(File.ReadAllText(BuiltInLibraryPath)));
            return path;
        }
    }
}
