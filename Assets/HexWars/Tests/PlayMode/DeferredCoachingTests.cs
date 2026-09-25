using System.Collections;
using System.Linq;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
using UnityEngine.UI;

namespace HexWars.Presentation.PlayModeTests
{
    public sealed class DeferredCoachingTests
    {
        bool _enabled;
        [SetUp]
        public void SetUp()
        {
            _enabled = TipsService.Enabled;
            TipBubble.Dismiss(); TipsService.NewGame(); TipsService.Enabled = true;
        }
        [UnityTearDown]
        public IEnumerator TearDown()
        {
            GameOverBanner.Dismiss();
            TipsService.NewGame(); TipBubble.Dismiss(); TipsService.Enabled = _enabled;
            yield return null;
        }

        [UnityTest]
        public IEnumerator FirstBountyWaitsForOpenHelpAndShowsOnceWithoutAnotherKill()
        {
            TipsService.Show("first-select", "Select a destination");
            TipsService.Show("first-bounty", "Unit destroyed. +4 points for your army.");
            TipsService.Show("first-bounty", "A later bounty must not replace the first.");
            AssertText("Select a destination");
            TipBubble.Show("Explicit stat reference", new Vector2(200, 200));
            yield return null;
            AssertText("Explicit stat reference");
            TipBubble.Dismiss(); yield return null;
            Assert.That(TipBubble.IsOpen, Is.False, "Dismissing help must leave a quiet interval.");
            yield return new WaitForSecondsRealtime(1.2f);
            var root = AssertText("Unit destroyed. +4 points for your army.");
            Assert.That(root.transform.Find("Backdrop"), Is.Null, "Deferred coaching stays non-modal.");
            Assert.That(root.GetComponentInChildren<Text>().fontSize, Is.EqualTo(15));
            TipBubble.Dismiss();
            TipsService.Show("first-bounty", "Do not repeat");
            Assert.That(TipBubble.IsOpen, Is.False);
        }

        [UnityTest]
        public IEnumerator DisablingTipsCancelsPendingCoaching()
        {
            QueueBounty(); TipsService.Enabled = false;
            yield return new WaitForSecondsRealtime(1.2f);
            Assert.That(TipBubble.IsOpen, Is.False);
            TipsService.Enabled = true;
            yield return new WaitForSecondsRealtime(1.2f);
            Assert.That(TipBubble.IsOpen, Is.False, "Turning tips back on must not revive a discarded event.");
        }

        [UnityTest]
        public IEnumerator NewGameDiscardsBountyFromThePreviousMatch()
        {
            QueueBounty(); TipsService.NewGame(); TipBubble.Dismiss();
            yield return new WaitForSecondsRealtime(1.2f);
            Assert.That(TipBubble.IsOpen, Is.False);
            TipsService.Show("first-bounty", "New match bounty");
            AssertText("New match bounty");
        }

        [UnityTest]
        public IEnumerator MatchResultCancelsCoachingUntilTheNextGameEvenAfterBannerDismissal()
        {
            QueueBounty(); yield return null;
            GameOverBanner.Show("Match complete", "Annihilation", Color.gray);
            Assert.That(GameObject.Find(GameOverBanner.RootName), Is.Not.Null);
            Assert.That(TipBubble.IsOpen, Is.False, "The result must immediately dismiss current coaching.");
            yield return new WaitForSecondsRealtime(1.2f);
            Assert.That(TipBubble.IsOpen, Is.False, "A pending bounty must not appear over the result.");

            GameOverBanner.Dismiss();
            Assert.That(GameObject.Find(GameOverBanner.RootName), Is.Null);
            TipsService.Enabled = false; TipsService.Enabled = true;
            TipsService.Show("first-bounty", "Late bounty from the finished match");
            TipsService.Show("first-select", "Inspecting the final board must stay quiet");
            yield return new WaitForSecondsRealtime(1.2f);
            Assert.That(TipBubble.IsOpen, Is.False, "Dismissal and preference changes must not reopen match coaching.");

            TipsService.NewGame();
            TipsService.Show("first-bounty", "Bounty from the next match");
            AssertText("Bounty from the next match");
        }

        static void QueueBounty()
        {
            TipBubble.Show("Existing help", new Vector2(200, 200));
            TipsService.Show("first-bounty", "Old bounty");
        }
        static GameObject AssertText(string expected)
        {
            var root = GameObject.Find(TipBubble.RootName);
            Assert.That(root, Is.Not.Null);
            Assert.That(root.GetComponentsInChildren<Text>().Any(t => t.text == expected), Is.True);
            return root;
        }
    }
}
