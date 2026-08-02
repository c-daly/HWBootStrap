using System;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using HexWars.Engine.Rl;
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
        public void Library_RejectsMissingUnexpectedAndWrongTypedRootMembers()
        {
            AssertLibraryRejected(json => RemoveProperty(json, "schema_version"));
            AssertLibraryRejected(json => InsertBeforeFinalBrace(
                json, ",\n  \"unexpected\": true"));
            AssertLibraryRejected(json => json.Replace(
                "\"schema_version\": 1", "\"schema_version\": 1.0"));
        }

        [Test]
        public void File_RejectsUnexpectedMembersAtEveryObjectLevel()
        {
            AssertScenarioRejected(_scenario, json =>
                InsertBeforeFinalBrace(json, ",\n  \"unexpected\": true"));
            foreach (string section in new[]
                     {
                         "board", "rules", "episode", "reward", "adaptive",
                     })
            {
                AssertScenarioRejected(_scenario, json =>
                    InsertFirstObjectMember(json, section, "\"unexpected\": true,"));
            }
        }

        [Test]
        public void File_RejectsMissingRulesFieldsThatJsonUtilityWouldDefault()
        {
            foreach (string field in new[]
                     {
                         "actions_per_turn", "round_cap", "starting_points",
                         "fog_of_war", "biomes_enabled", "bounty_rate",
                         "deploy_cost_multiplier", "generator_cost",
                         "generator_output", "generator_health",
                     })
            {
                AssertScenarioRejected(_scenario, json => RemoveProperty(json, field));
            }
        }

        [Test]
        public void File_RejectsMissingEnvironmentSpecificFields()
        {
            AssertScenarioRejected(_scenario, json =>
                RemoveProperty(json, "deployment_completion_bonus"));
            AssertScenarioRejected(_scenario, json =>
                RemoveObjectProperty(json, "adaptive"));

            var tactical = MlTrainingScenarioLibrary.Load(BuiltInLibraryPath)
                .Filter(MlEnvironmentContract.TacticalV1).First();
            AssertScenarioRejected(tactical, json => RemoveProperty(json, "shape_scale"));
        }

        [Test]
        public void File_RejectsIrrelevantEnvironmentSpecificSectionsAndRewardFields()
        {
            var tactical = MlTrainingScenarioLibrary.Load(BuiltInLibraryPath)
                .Filter(MlEnvironmentContract.TacticalV1).First();
            AssertScenarioRejected(tactical, json => InsertBeforeFinalBrace(
                json,
                ",\n  \"adaptive\": {\"starting_unit_count\": 6," +
                "\"starting_army_budget\": 132,\"max_design_point_cost\": 24}"));
            AssertScenarioRejected(tactical, json =>
                InsertFirstObjectMember(
                    json, "reward", "\"intermediate_decision_penalty\": 0.001,"));
            AssertScenarioRejected(_scenario, json =>
                InsertFirstObjectMember(json, "reward", "\"shape_scale\": 0.01,"));
        }

        [Test]
        public void File_RejectsJsonPrimitiveTypeMismatches()
        {
            AssertScenarioRejected(_scenario, json => json.Replace(
                "\"fog_of_war\": true", "\"fog_of_war\": 1"));
            AssertScenarioRejected(_scenario, json => json.Replace(
                "\"actions_per_turn\": 0", "\"actions_per_turn\": false"));
            AssertScenarioRejected(_scenario, json => json.Replace(
                "\"width\": 13", "\"width\": 13.0"));
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

        [TestCase(MlEnvironmentContract.TacticalV1, "tactical-standard")]
        [TestCase(MlEnvironmentContract.AdaptiveV1, "adaptive-standard")]
        public void PreflightDimensions_MatchAuthoritativeEngineDefaults(
            MlEnvironmentContract environment, string templateId)
        {
            var scenario = MlTrainingScenarioLibrary.Load(BuiltInLibraryPath)
                .Filter(environment).Single(item => item.Id == templateId);

            MlTrainingScenarioPreflight preflight =
                MlTrainingScenarioPreflight.Create(scenario);
            TrainingScenario engineScenario = MlTrainingScenarioPreflight.ToEngine(scenario);
            MlContract contract = environment == MlEnvironmentContract.AdaptiveV1
                ? MlContract.CreateAdaptive(engineScenario.BuildAdaptive())
                : MlContract.Create(engineScenario.BuildTactical());

            Assert.That(preflight.ObservationSize, Is.EqualTo(contract.ObservationSize));
            Assert.That(preflight.ActionSize, Is.EqualTo(contract.ActionSize));
            Assert.That(preflight.DisplayText, Does.Contain(
                "Observation " + contract.ObservationSize));
            Assert.That(preflight.DisplayText, Does.Contain(
                "actions " + contract.ActionSize));
            Assert.That(preflight.LargeScenarioWarning, Is.False);
        }

        [Test]
        public void ScenarioConversion_PreservesEngineContractPrecision()
        {
            TrainingScenario expected = TrainingScenario.CreateStandard(
                MlContract.AdaptiveVersion, "adaptive-standard");
            TrainingScenario converted =
                MlTrainingScenarioPreflight.ToEngine(_scenario);

            Assert.That(
                converted.Board.FlatChance,
                Is.EqualTo(expected.Board.FlatChance));
            Assert.That(
                converted.Rules.BountyRate,
                Is.EqualTo(expected.Rules.BountyRate));
            Assert.That(
                converted.Rules.DeployCostMultiplier,
                Is.EqualTo(expected.Rules.DeployCostMultiplier));
            Assert.That(
                ModelDuelEnvironmentFactory.ContractIdentity(converted).EncodingHash,
                Is.EqualTo(
                    ModelDuelEnvironmentFactory.ContractIdentity(expected).EncodingHash));
        }

        [TestCase(MlEnvironmentContract.TacticalV1, "tactical-large-battle")]
        [TestCase(MlEnvironmentContract.AdaptiveV1, "adaptive-large-battle")]
        public void PreflightDimensions_MatchAuthoritative24By16Presets(
            MlEnvironmentContract environment, string templateId)
        {
            var scenario = MlTrainingScenarioLibrary.Load(BuiltInLibraryPath)
                .Filter(environment).Single(item => item.Id == templateId);

            MlTrainingScenarioPreflight preflight =
                MlTrainingScenarioPreflight.Create(scenario);
            TrainingScenario engineScenario = MlTrainingScenarioPreflight.ToEngine(scenario);
            MlContract contract = environment == MlEnvironmentContract.AdaptiveV1
                ? MlContract.CreateAdaptive(engineScenario.BuildAdaptive())
                : MlContract.Create(engineScenario.BuildTactical());

            Assert.That(scenario.Board.Width, Is.EqualTo(24));
            Assert.That(scenario.Board.Height, Is.EqualTo(16));
            Assert.That(preflight.ObservationSize, Is.EqualTo(contract.ObservationSize));
            Assert.That(preflight.ActionSize, Is.EqualTo(contract.ActionSize));
            Assert.That(preflight.LargeScenarioWarning, Is.True);
        }

        [Test]
        public void TacticalV2Scenario_RoundTripsTemplateIdentityAndCount()
        {
            MlTrainingScenario scenario = Load("tactical-v2-standard");
            scenario.TacticalV2.StartingUnitCount = 7;
            scenario.TacticalV2.MaxControllableUnits = 7;
            scenario.TacticalV2.Templates =
                MlTacticalRosterSource.Snapshot(0).ToList();

            string path = MlTrainingScenarioStore.WriteSessionScenario(_projectRoot, scenario);
            MlTrainingScenario restored = MlTrainingScenarioFile.Load(path);

            Assert.That(restored.TacticalV2.StartingUnitCount, Is.EqualTo(7));
            Assert.That(restored.TacticalV2.Templates.Select(item => item.Id),
                Is.EqualTo(scenario.TacticalV2.Templates.Select(item => item.Id)));
        }

        [Test]
        public void ProfiledTacticalV2Scenario_LoadsStrictFixture()
        {
            string path = Path.Combine(
                "Assets", "HexWars", "Tests", "Editor", "Fixtures",
                "ProfiledTacticalV2Scenario.json");

            MlTrainingScenario loaded = MlTrainingScenarioFile.Load(path);

            Assert.That(loaded.Id, Is.EqualTo("annihilation-imitation-v1"));
        }

        [Test]
        public void ProfiledTacticalV2Scenario_SerializePreservesProfileData()
        {
            string path = Path.Combine(
                "Assets", "HexWars", "Tests", "Editor", "Fixtures",
                "ProfiledTacticalV2Scenario.json");
            MlTrainingScenario loaded = MlTrainingScenarioFile.Load(path);

            string outputPath =
                MlTrainingScenarioStore.WriteSessionScenario(_projectRoot, loaded);
            string serialized = File.ReadAllText(outputPath);

            Assert.That(serialized, Does.Contain("\"start_profiles\""));
            Assert.That(serialized, Does.Contain("\"start_distribution\""));
            Assert.That(serialized, Does.Contain("\"conversion-1v1-far\""));
            Assert.That(serialized, Does.Contain("\"standard-3v3\""));
            Assert.That(serialized, Does.Contain("\"basis_points\": 7000"));
            Assert.That(
                MlTrainingScenarioFile.Load(outputPath).Id,
                Is.EqualTo("annihilation-imitation-v1"));
        }

        [Test]
        public void ProfiledTacticalV2Scenario_EngineConversionPreservesStartContract()
        {
            string path = Path.Combine(
                "Assets", "HexWars", "Tests", "Editor", "Fixtures",
                "ProfiledTacticalV2Scenario.json");
            MlTrainingScenario loaded = MlTrainingScenarioFile.Load(path);

            TrainingScenario converted = MlTrainingScenarioPreflight.ToEngine(loaded);

            Assert.That(loaded.TacticalV2.StartProfiles, Has.Count.EqualTo(10));
            Assert.That(
                loaded.TacticalV2.StartDistribution
                    .Single(item => item.ProfileId == "standard-3v3").BasisPoints,
                Is.EqualTo(7000));
            Assert.That(converted.TacticalV2.StartProfiles, Has.Count.EqualTo(10));
            Assert.That(
                converted.TacticalV2.StartProfiles
                    .Single(item => item.Id == "conversion-1v1-far").Separation,
                Is.EqualTo("far"));
            Assert.That(
                converted.TacticalV2.StartDistribution
                    .Single(item => item.ProfileId == "standard-3v3").BasisPoints,
                Is.EqualTo(7000));
        }

        [Test]
        public void SourceRunPreflight_LoadsTheResolvedRunScenario()
        {
            string run = Path.Combine(_scratch, "source-run");
            Directory.CreateDirectory(run);
            string scenarioPath = Path.Combine(run, "scenario.json");
            File.Copy(
                MlTrainingScenarioStore.WriteSessionScenario(_projectRoot, _scenario),
                scenarioPath);

            MlTrainingScenarioPreflight preflight =
                MlTrainingScenarioPreflight.LoadSourceRun(run);

            Assert.That(preflight.TemplateId, Is.EqualTo("adaptive-standard"));
            Assert.That(preflight.DisplayText, Does.Contain("adaptive-v1"));
            Assert.That(preflight.DisplayText, Does.Contain("Board 13\u00d79"));
        }

        string CopyLibrary(Func<string, string> transform)
        {
            string path = Path.Combine(_scratch, "training-game-templates.json");
            File.WriteAllText(path, transform(File.ReadAllText(BuiltInLibraryPath)));
            return path;
        }

        MlTrainingScenario Load(string templateId) =>
            MlTrainingScenarioLibrary.Load(BuiltInLibraryPath)
                .Templates.First(item => item.Id == templateId);

        void AssertLibraryRejected(Func<string, string> transform)
        {
            string path = CopyLibrary(transform);
            Assert.Throws<InvalidDataException>(() => MlTrainingScenarioLibrary.Load(path));
        }

        void AssertScenarioRejected(
            MlTrainingScenario scenario, Func<string, string> transform)
        {
            string validPath =
                MlTrainingScenarioStore.WriteSessionScenario(_projectRoot, scenario);
            string valid = File.ReadAllText(validPath);
            string changed = transform(valid);
            Assert.That(changed, Is.Not.EqualTo(valid), "Test transform must change JSON.");
            string candidate = Path.Combine(
                _scratch, "scenario-" + Guid.NewGuid().ToString("N") + ".json");
            File.WriteAllText(candidate, changed);
            Assert.Throws<InvalidDataException>(() => MlTrainingScenarioFile.Load(candidate));
        }

        static string InsertFirstObjectMember(
            string json, string property, string member)
        {
            string marker = "\"" + property + "\": {";
            return json.Replace(marker, marker + "\n    " + member);
        }

        static string InsertBeforeFinalBrace(string json, string fragment)
        {
            int index = json.LastIndexOf('}');
            return index < 0 ? json : json.Insert(index, fragment);
        }

        static string RemoveObjectProperty(string json, string property) =>
            new Regex(
                ",?\\s*\"" + Regex.Escape(property) + "\"\\s*:\\s*\\{[^{}]*\\}",
                RegexOptions.CultureInvariant).Replace(json, string.Empty, 1);

        static string RemoveProperty(string json, string property)
        {
            const string value = "(?:true|false|null|\"(?:\\\\.|[^\"])*\"|" +
                "-?\\d+(?:\\.\\d+)?(?:[eE][+-]?\\d+)?)";
            string escaped = Regex.Escape(property);
            var followedByComma = new Regex(
                "\\s*\"" + escaped + "\"\\s*:\\s*" + value + "\\s*,",
                RegexOptions.CultureInvariant);
            string changed = followedByComma.Replace(json, string.Empty, 1);
            if (!string.Equals(changed, json, StringComparison.Ordinal)) return changed;
            return new Regex(
                ",\\s*\"" + escaped + "\"\\s*:\\s*" + value,
                RegexOptions.CultureInvariant).Replace(json, string.Empty, 1);
        }
    }
}
