using HexWars.Presentation;
using NUnit.Framework;
using UnityEngine;

namespace HexWars.Presentation.Tests
{
    public sealed class ModelArenaIdentityTests
    {
        [Test]
        public void Driver_AlwaysCarriesArenaOverlays()
        {
            var go = new GameObject("arena", typeof(BoardRenderer), typeof(ModelDuelDriver));
            try
            {
                Assert.That(go.GetComponent<ModelArenaIdentityOverlay>(), Is.Not.Null);
                Assert.That(go.GetComponent<EventConsole>(), Is.Not.Null);
                var driver = go.GetComponent<ModelDuelDriver>();
                driver.P0Spec = "greedy";
                driver.P1Spec = "random";
                Assert.That(driver.IdentitySnapshot()[0].Controller, Is.EqualTo("Greedy"));
            }
            finally { Object.DestroyImmediate(go); }
        }

        [Test]
        public void OrdinaryGame_DoesNotCarryArenaEventConsole()
        {
            var go = new GameObject("ordinary game", typeof(GameBootstrap));
            try
            {
                Assert.That(go.GetComponent<EventConsole>(), Is.Null);
            }
            finally { Object.DestroyImmediate(go); }
        }

        [Test]
        public void ReplayPlayer_CarriesEventConsole()
        {
            var go = new GameObject("replay", typeof(ReplayPlayer));
            try
            {
                Assert.That(go.GetComponent<EventConsole>(), Is.Not.Null);
            }
            finally { Object.DestroyImmediate(go); }
        }

        [Test]
        public void Driver_IdentitySnapshotBeforeStart_IsSafeAndMarksNeitherSeatActive()
        {
            var go = new GameObject("arena", typeof(BoardRenderer), typeof(ModelDuelDriver));
            try
            {
                var rows = go.GetComponent<ModelDuelDriver>().IdentitySnapshot();

                Assert.That(rows[0].IsActive, Is.False);
                Assert.That(rows[1].IsActive, Is.False);
            }
            finally { Object.DestroyImmediate(go); }
        }

        [Test]
        public void Overlay_ShouldRender_RequiresAnEnabledActiveDriver()
        {
            var go = new GameObject("arena", typeof(BoardRenderer), typeof(ModelDuelDriver));
            try
            {
                var driver = go.GetComponent<ModelDuelDriver>();
                Assert.That(ModelArenaIdentityOverlay.ShouldRender(driver), Is.True);

                driver.enabled = false;
                Assert.That(ModelArenaIdentityOverlay.ShouldRender(driver), Is.False);

                driver.enabled = true;
                go.SetActive(false);
                Assert.That(ModelArenaIdentityOverlay.ShouldRender(driver), Is.False);
            }
            finally { Object.DestroyImmediate(go); }
        }

        [Test]
        public void Overlay_HidesAdaptiveIdentityBeforeAtomicReveal()
        {
            var go = new GameObject("arena", typeof(BoardRenderer), typeof(ModelDuelDriver));
            try
            {
                var driver = go.GetComponent<ModelDuelDriver>();
                driver.Environment = MlEnvironmentContract.AdaptiveV1;
                Assert.That(ModelArenaIdentityOverlay.ShouldRender(driver), Is.False);

                driver.Environment = MlEnvironmentContract.TacticalV1;
                Assert.That(ModelArenaIdentityOverlay.ShouldRender(driver), Is.True);
            }
            finally { Object.DestroyImmediate(go); }
        }

        [Test]
        public void Overlay_CharacterBudget_ShrinksForNarrowerRowsAndCapsLandscape()
        {
            Assert.That(ModelArenaIdentityOverlay.CharacterBudget(160f, true), Is.LessThan(
                ModelArenaIdentityOverlay.CharacterBudget(320f, true)));
            Assert.That(ModelArenaIdentityOverlay.CharacterBudget(4f, true), Is.GreaterThanOrEqualTo(24));
            Assert.That(ModelArenaIdentityOverlay.CharacterBudget(430f, false), Is.EqualTo(72));
        }

        [Test]
        public void Overlay_RowText_BeginsWithRequestedActiveArrow()
        {
            var row = ModelArenaIdentity.Build("greedy", "random", null, null, 0, 0, 0, 0)[0];

            Assert.That(ModelArenaIdentityOverlay.RowText(row, 72), Does.StartWith("▶"));
        }

        [Test]
        public void Build_LiveJsonSpecWithoutResolvedMetadata_UsesSafeRunLeaf()
        {
            string spec = new ModelSeatConfiguration
            {
                Kind = ModelControllerKind.LiveRun,
                Path = "C:/runs/alpha",
            }.BuildSpec();

            var row = ModelArenaIdentity.Build(spec, "greedy", null, null, -1, 0, 0, 0)[0];

            Assert.That(row.Controller, Is.EqualTo("alpha"));
            Assert.That(row.Checkpoint, Is.EqualTo("loading checkpoint"));
        }

        [TestCase("")]
        [TestCase("{malformed")]
        [TestCase("run:")]
        public void Build_BlankOrMalformedModelSpec_FallsBackSafely(string spec)
        {
            Assert.DoesNotThrow(() => ModelArenaIdentity.Build(spec, "greedy", null, null, -1, 0, 0, 0));
            Assert.That(ModelArenaIdentity.Build(spec, "greedy", null, null, -1, 0, 0, 0)[0].Controller,
                Is.EqualTo("model"));
        }

        [TestCase("run:C:/runs/alpha", "alpha")]
        [TestCase("ppo:C:/models/alpha.zip", "alpha.zip")]
        [TestCase("dqn:C:/models/beta.zip", "beta.zip")]
        public void Build_PrefixedSpecs_UseSafePathLeaf(string spec, string expected)
        {
            Assert.That(ModelArenaIdentity.Build(spec, "greedy", null, null, -1, 0, 0, 0)[0].Controller,
                Is.EqualTo(expected));
        }

        [Test]
        public void Build_ResolvedIdentity_IncludesReloadFailureStatus()
        {
            var resolved = PolicyBridge.ParseReady(
                "{\"ready\":true,\"seat_models\":[{\"seat\":0,\"path\":\"model_128_steps.zip\",\"step\":128}]}"
            ).Seats[0];

            var row = ModelArenaIdentity.Build("ppo:model.zip", "greedy", resolved, null, -1, 0, 0, 0,
                "reload failed", null)[0];

            Assert.That(row.Checkpoint, Is.EqualTo("model_128_steps.zip"));
            Assert.That(row.Status, Is.EqualTo("reload failed"));
            Assert.That(ModelArenaIdentityOverlay.RowText(row, 72), Does.Contain("reload failed"));
        }

        [Test]
        public void Overlay_Layout_StacksLandscapeRowsLeftOfEventConsole()
        {
            Rect first = ModelArenaIdentityOverlay.RowRect(0, 1280f, false);
            Rect second = ModelArenaIdentityOverlay.RowRect(1, 1280f, false);

            Assert.That(first.xMax, Is.LessThanOrEqualTo(850f));
            Assert.That(second.y, Is.GreaterThan(first.y));
            Assert.That(second.x, Is.EqualTo(first.x));
        }

        [Test]
        public void Overlay_PortraitText_UsesTwoLinesAndKeepsMetricsOffIdentityLine()
        {
            var resolved = PolicyBridge.ParseReady(
                "{\"ready\":true,\"seat_models\":[{\"seat\":0,\"path\":\"model_128_steps.zip\",\"step\":128}]}"
            ).Seats[0];
            var row = ModelArenaIdentity.Build("ppo:model.zip", "greedy", resolved, null, -1, 2, 1, 0)[0];
            float width = ModelArenaIdentityOverlay.RowWidth(390f, true);

            string[] lines = ModelArenaIdentityOverlay.PortraitLines(
                row, ModelArenaIdentityOverlay.CharacterBudget(width, true));

            Assert.That(lines, Has.Length.EqualTo(2));
            Assert.That(lines[0], Does.Not.Contain(".zip"));
            Assert.That(lines[0], Does.Not.Contain(row.Record));
            Assert.That(lines[1], Does.Contain("step 128"));
            Assert.That(lines[1], Does.Contain(row.Record));
        }

        [Test]
        public void Build_LabelsScriptedSeatsAndMarksCurrentSeat()
        {
            var rows = ModelArenaIdentity.Build("greedy", "random", null, null, 1, 0, 0, 0);

            Assert.That(rows[0].Player, Is.EqualTo("P1"));
            Assert.That(rows[0].Controller, Is.EqualTo("Greedy"));
            Assert.That(rows[0].IsActive, Is.False);
            Assert.That(rows[1].Controller, Is.EqualTo("Random"));
            Assert.That(rows[1].IsActive, Is.True);
            Assert.That(rows[0].Record, Is.EqualTo("0-0-0 (—)"));
        }

        [Test]
        public void Build_LabelsPassiveAsScriptedWithoutLoadingCheckpoint()
        {
            ModelArenaSeatIdentity row = ModelArenaIdentity.Build(
                "passive", "greedy", null, null, 0, 0, 0, 0)[0];

            Assert.That(row.Controller, Is.EqualTo("Passive"));
            Assert.That(row.Checkpoint, Is.Empty);
            Assert.That(row.Algorithm, Is.Empty);
        }

        [Test]
        public void Build_UsesResolvedCheckpointAndMirrorsRecords()
        {
            var resolved = PolicyBridge.ParseReady(
                "{\"ready\":true,\"model_seats\":[0],\"seat_models\":[{\"seat\":0,\"kind\":\"run\",\"path\":\"C:/runs/alpha/checkpoints/model_20480_steps.zip\",\"algorithm\":\"maskable_ppo\",\"step\":20480}]}"
            ).Seats[0];

            var rows = ModelArenaIdentity.Build("run:C:/runs/alpha", "greedy", resolved, null, 0, 3, 1, 1);

            Assert.That(rows[0].Controller, Is.EqualTo("alpha"));
            Assert.That(rows[0].Checkpoint, Is.EqualTo("model_20480_steps.zip"));
            Assert.That(rows[0].Algorithm, Is.EqualTo("Maskable PPO"));
            Assert.That(rows[0].Step, Is.EqualTo("step 20,480"));
            Assert.That(rows[0].Record, Is.EqualTo("3-1-1 (60%)"));
            Assert.That(rows[1].Record, Is.EqualTo("1-3-1 (20%)"));
        }

        [Test]
        public void Build_LabelsStructuredPolicyGradientAsOutcomeCandidate()
        {
            var resolved = PolicyBridge.ParseReady(
                "{\"ready\":true,\"model_seats\":[0]," +
                "\"seat_models\":[{\"seat\":0,\"kind\":\"run\"," +
                "\"path\":\"C:/runs/candidate/checkpoints/policy-update-000003.pt\"," +
                "\"algorithm\":\"structured_policy_gradient\",\"step\":3}]}"
            ).Seats[0];

            ModelArenaSeatIdentity row = ModelArenaIdentity.Build(
                "run:C:/runs/candidate",
                "greedy",
                resolved,
                null,
                0,
                0,
                0,
                0)[0];

            Assert.That(row.Controller, Is.EqualTo("candidate"));
            Assert.That(row.Algorithm, Is.EqualTo("Outcome candidate"));
            Assert.That(row.Checkpoint,
                Is.EqualTo("policy-update-000003.pt"));
            Assert.That(row.Step, Is.EqualTo("step 3"));
        }

        [Test]
        public void Build_PresentationRolesProjectLearnerCentricRecordsOntoCurrentSeats()
        {
            var rows = ModelArenaIdentity.Build(
                "greedy", "run:C:/runs/learner",
                null, null, currentSeat: 1,
                p0Wins: 4, p1Wins: 3, draws: 2,
                learnerSeat: 1, learnerWins: 5, learnerLosses: 2,
                learnerDraws: 1);

            Assert.That(rows[0].Role, Is.EqualTo("Opponent"));
            Assert.That(rows[0].Record, Is.EqualTo("2-5-1 (25%)"));
            Assert.That(rows[1].Role, Is.EqualTo("Learner"));
            Assert.That(rows[1].Record, Is.EqualTo("5-2-1 (62.5%)"));
            Assert.That(ModelArenaIdentityOverlay.RowText(rows[1], 72),
                Does.StartWith("▶ P2").And.Contain("Learner").And.Contain("learner"));
        }

        [Test]
        public void Build_PresentationOpponentUsesRecordedPoolLabelWithoutRelabelingLearner()
        {
            var rows = ModelArenaIdentity.Build(
                "run:C:/runs/learner", "ppo:C:/models/pool-b.zip",
                null, null, currentSeat: 0,
                p0Wins: 0, p1Wins: 0, draws: 0,
                learnerSeat: 0, learnerWins: 0, learnerLosses: 0,
                learnerDraws: 0, opponentLabel: "pool-b · step 20");

            Assert.That(rows[0].Role, Is.EqualTo("Learner"));
            Assert.That(rows[0].Controller, Is.EqualTo("learner"));
            Assert.That(rows[1].Role, Is.EqualTo("Opponent"));
            Assert.That(rows[1].Controller, Is.EqualTo("pool-b · step 20"));
            Assert.That(ModelArenaIdentityOverlay.RowText(rows[1], 72),
                Does.Contain("Opponent").And.Contain("pool-b · step 20"));
        }

        [Test]
        public void Build_ManualArenaLeavesRolesBlankAndKeepsSeatRecords()
        {
            var rows = ModelArenaIdentity.Build(
                "greedy", "random", null, null, 0, 3, 1, 1);

            Assert.That(rows[0].Role, Is.Empty);
            Assert.That(rows[0].Record, Is.EqualTo("3-1-1 (60%)"));
            Assert.That(rows[1].Role, Is.Empty);
            Assert.That(rows[1].Record, Is.EqualTo("1-3-1 (20%)"));
        }

        [Test]
        public void Build_HandlesIncompleteResolvedRunMetadata()
        {
            var resolved = PolicyBridge.ParseReady(
                "{\"ready\":true,\"model_seats\":[0],\"seat_models\":[{\"seat\":0,\"kind\":\"run\",\"algorithm\":\"maskable_ppo\"}]}"
            ).Seats[0];

            var row = ModelArenaIdentity.Build(null, "greedy", resolved, null, 0, 0, 0, 0)[0];

            Assert.That(row.Controller, Is.EqualTo("model"));
            Assert.That(row.Checkpoint, Is.EqualTo(string.Empty));
            Assert.That(row.Algorithm, Is.EqualTo("Maskable PPO"));
            Assert.That(row.Step, Is.EqualTo("step unknown"));
        }

        [Test]
        public void MiddleTruncate_PreservesBothEnds()
        {
            Assert.That(ModelArenaIdentity.MiddleTruncate("abcdefghijklmnop", 11), Is.EqualTo("abcd…klmnop"));
        }

        [Test]
        public void Build_IncludesPointsInIdentityRowsAndDefaultsToZero()
        {
            var withoutPoints = ModelArenaIdentity.Build("greedy", "random", null, null, 0, 0, 0, 0);

            Assert.That(withoutPoints[0].Points, Is.EqualTo(0));
            Assert.That(withoutPoints[1].Points, Is.EqualTo(0));

            var withPoints = ModelArenaIdentity.Build(
                "greedy", "random", null, null, 0, 0, 0, 0,
                p0Points: 12, p1Points: 7);

            Assert.That(withPoints[0].Points, Is.EqualTo(12));
            Assert.That(withPoints[1].Points, Is.EqualTo(7));
        }

        [Test]
        public void Overlay_MetricsText_IncludesPointsContinuously()
        {
            var scored = ModelArenaIdentity.Build(
                "greedy", "random", null, null, 0, 0, 0, 0, p0Points: 5)[0];
            var unscored = ModelArenaIdentity.Build("greedy", "random", null, null, 0, 0, 0, 0)[0];

            Assert.That(ModelArenaIdentityOverlay.MetricsText(scored), Does.Contain("5 pts"));
            Assert.That(ModelArenaIdentityOverlay.MetricsText(unscored), Does.Contain("0 pts"),
                "points must display continuously, including zero, per spec \"Player Point Totals\"");
        }

        // ---- Comfort controls (Sound / Fullscreen) row layout ----

        [Test]
        public void ComfortControlsRowIndex_ClaimsTheRowAfterFogWhenShown()
        {
            Assert.That(ModelArenaIdentityOverlay.ComfortControlsRowIndex(2, fogRowShown: false), Is.EqualTo(2));
            Assert.That(ModelArenaIdentityOverlay.ComfortControlsRowIndex(2, fogRowShown: true), Is.EqualTo(3));
        }

        [Test]
        public void ComfortControlsRowRect_StacksBelowIdentityRowsAndClaimsAnExtraRowWhenFogShown()
        {
            Rect withoutFog = ModelArenaIdentityOverlay.ComfortControlsRowRect(2, fogRowShown: false, 1280f, false);
            Rect withFog = ModelArenaIdentityOverlay.ComfortControlsRowRect(2, fogRowShown: true, 1280f, false);

            Assert.That(withFog.y, Is.GreaterThan(withoutFog.y),
                "when the fog toggle claims a row, comfort controls must stack one row further down");
            Assert.That(withoutFog.x, Is.EqualTo(withFog.x), "both share the same left corner as the identity rows");
        }

        [Test]
        public void SoundAndFullscreenToggleRects_SitSideBySideWithoutOverlappingOrOverflowingTheRow()
        {
            Rect row = ModelArenaIdentityOverlay.ComfortControlsRowRect(2, fogRowShown: true, 1280f, false);
            Rect sound = ModelArenaIdentityOverlay.SoundToggleRect(row);
            Rect fullscreen = ModelArenaIdentityOverlay.FullscreenToggleRect(row);

            Assert.That(sound.x, Is.EqualTo(row.x));
            Assert.That(sound.xMax, Is.LessThanOrEqualTo(fullscreen.x), "Sound must sit fully left of Fullscreen");
            Assert.That(fullscreen.xMax, Is.LessThanOrEqualTo(row.xMax), "Fullscreen must not overflow the row");
            Assert.That(sound.y, Is.EqualTo(row.y));
            Assert.That(fullscreen.y, Is.EqualTo(row.y));
            Assert.That(sound.height, Is.EqualTo(row.height));
            Assert.That(fullscreen.height, Is.EqualTo(row.height));
        }

        [Test]
        public void ComfortControlsRowRect_NarrowUsesPortraitRowHeightForSpacing()
        {
            Rect landscape = ModelArenaIdentityOverlay.ComfortControlsRowRect(1, fogRowShown: false, 1280f, false);
            Rect portrait = ModelArenaIdentityOverlay.ComfortControlsRowRect(1, fogRowShown: false, 390f, true);

            Assert.That(portrait.y, Is.GreaterThan(landscape.y),
                "portrait rows are taller, so the same row index sits further down the screen");
        }

        // ---- Fullscreen reflection guard (GameViewFullscreen) ----
        //
        // The reflection calls in TryResolve/IsMaximized/SetMaximized are wrapped in try/catch so a
        // failure (unresolvable type, or GetWindow/GetValue/SetValue throwing — e.g. no Game view
        // exists yet under -batchmode) latches Unavailable and lets the toggle silently disappear
        // instead of throwing on every OnGUI event. GameViewTypeName is a test seam forcing the
        // resolve path down its failure branch without needing to fake an actual editor window; the
        // deeper case (a real GetWindow/GetValue/SetValue throw after a *successful* type/property
        // resolve) isn't independently forced here — faking that would need a stand-in reflection
        // target, which is disproportionate scaffolding for this defensive guard.

        [Test]
        public void GameViewFullscreen_UnresolvableTypeName_IsMaximizedReturnsFalseAndLatchesUnavailable()
        {
            string original = GameViewFullscreen.GameViewTypeName;
            GameViewFullscreen.ResetCacheForTests();
            GameViewFullscreen.GameViewTypeName = "Totally.Bogus.NonexistentType, NoSuchAssembly";
            try
            {
                Assert.That(GameViewFullscreen.Unavailable, Is.False,
                    "must not latch before the first resolve attempt");
                Assert.That(GameViewFullscreen.IsMaximized(), Is.False);
                Assert.That(GameViewFullscreen.Unavailable, Is.True,
                    "an unresolvable type must latch Unavailable so the toggle silently stops drawing itself");
            }
            finally
            {
                GameViewFullscreen.GameViewTypeName = original;
                GameViewFullscreen.ResetCacheForTests();
            }
        }

        [Test]
        public void GameViewFullscreen_UnresolvableTypeName_SetMaximizedDoesNotThrow()
        {
            string original = GameViewFullscreen.GameViewTypeName;
            GameViewFullscreen.ResetCacheForTests();
            GameViewFullscreen.GameViewTypeName = "Totally.Bogus.NonexistentType, NoSuchAssembly";
            try
            {
                Assert.DoesNotThrow(() => GameViewFullscreen.SetMaximized(true));
                Assert.That(GameViewFullscreen.Unavailable, Is.True);
            }
            finally
            {
                GameViewFullscreen.GameViewTypeName = original;
                GameViewFullscreen.ResetCacheForTests();
            }
        }

        [Test]
        public void GameViewFullscreen_RealGameViewType_NeverThrows()
        {
            GameViewFullscreen.ResetCacheForTests();
            try
            {
                Assert.DoesNotThrow(() => GameViewFullscreen.IsMaximized());
                Assert.DoesNotThrow(() => GameViewFullscreen.SetMaximized(false));
            }
            finally
            {
                GameViewFullscreen.ResetCacheForTests();
            }
        }

        // Locks the no-create/no-focus semantics behind the fix for the per-frame OS-foreground-theft
        // bug: TryResolve must never call EditorWindow.GetWindow (which creates AND focuses a Game
        // view), so under a batch-mode test run — where no Game view instance exists — resolution
        // must fail closed (IsMaximized false) without latching Unavailable, since a missing window
        // instance is transient (a Game view can still open later) unlike an unresolvable type/property.
        [Test]
        public void GameViewFullscreen_RealTypeButNoGameViewInstance_ReturnsFalseWithoutLatching()
        {
            GameViewFullscreen.ResetCacheForTests();
            try
            {
                Assert.That(GameViewFullscreen.IsMaximized(), Is.False,
                    "no Game view instance exists under batch-mode test execution, and resolution " +
                    "must never create one via GetWindow");
                Assert.That(GameViewFullscreen.Unavailable, Is.False,
                    "a missing window instance is transient (unlike an unresolvable type) and must " +
                    "not latch Unavailable");
                Assert.DoesNotThrow(() => GameViewFullscreen.SetMaximized(true),
                    "SetMaximized must no-op rather than throw when there is no window to set");
                Assert.That(GameViewFullscreen.Unavailable, Is.False);
            }
            finally
            {
                GameViewFullscreen.ResetCacheForTests();
            }
        }
    }
}
