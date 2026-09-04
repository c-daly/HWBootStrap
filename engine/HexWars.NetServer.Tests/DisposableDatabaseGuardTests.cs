using HexWars.NetServer.Tests.Fixtures;
using NUnit.Framework;

namespace HexWars.NetServer.Tests
{
    /// <summary>
    /// The check that stands between HEXWARS_TEST_DATABASE_URL and a database somebody cares about.
    ///
    /// These tests touch no database, which is the point: the rule has to be provable without running the
    /// destructive thing it guards, or the only way to test it would be to point it at something real.
    /// </summary>
    [TestFixture]
    public sealed class DisposableDatabaseGuardTests
    {
        /// <summary>A username that is nothing like any database name below, so an assertion about the
        /// database appearing in a message cannot be satisfied by the credentials instead.</summary>
        const string Username = "appuser";

        const string Password = "correct-horse-battery";

        static Func<string, string?> NoEnvironment => _ => null;

        static Func<string, string?> Confirming(string value) =>
            name => name == DisposableDatabaseGuard.ConfirmationEnvironmentVariable ? value : null;

        static string Url(string database) =>
            "postgres://" + Username + ":" + Password + "@db.example.com:5432/" + database;

        static string KeyValue(string database) =>
            "Host=db.example.com;Port=5432;Username=" + Username + ";Password=" + Password
            + ";Database=" + database;

        [TestCase("hexwars_test")]
        [TestCase("test")]
        [TestCase("HexWarsTEST")]
        [TestCase("ci_TestBed")]
        public void ADatabaseWhoseNameSaysItIsForTests_IsDisposable(string database)
        {
            Assert.That(DisposableDatabaseGuard.IsDisposable(Url(database), NoEnvironment, out string url),
                Is.True, url);
            Assert.That(DisposableDatabaseGuard.IsDisposable(KeyValue(database), NoEnvironment, out string kv),
                Is.True, kv);
        }

        [TestCase("hexwars")]
        [TestCase("production")]
        [TestCase("hexwars_prod")]
        [TestCase("postgres")]
        public void ADatabaseWhoseNameDoesNot_IsRefused(string database)
        {
            Assert.That(DisposableDatabaseGuard.IsDisposable(Url(database), NoEnvironment, out _), Is.False);
            Assert.That(DisposableDatabaseGuard.IsDisposable(KeyValue(database), NoEnvironment, out _),
                Is.False, "the key=value form is the same target and must get the same answer");
        }

        [Test]
        public void ARefusedDatabase_IsAcceptedOnceItIsConfirmedByName()
        {
            Assert.That(
                DisposableDatabaseGuard.IsDisposable(Url("hexwars"), Confirming("hexwars"), out string why),
                Is.True, why);
            Assert.That(
                DisposableDatabaseGuard.IsDisposable(KeyValue("hexwars"), Confirming("hexwars"), out _),
                Is.True);
        }

        [Test]
        public void AConfirmationNamingADifferentDatabase_DoesNotAuthoriseThisOne()
        {
            // The confirmation names a database rather than saying "yes" so that one left behind in a
            // shell cannot authorise whatever database is connected to next.
            Assert.That(DisposableDatabaseGuard.IsDisposable(Url("hexwars"), Confirming("other"), out _),
                Is.False);
            Assert.That(DisposableDatabaseGuard.IsDisposable(Url("hexwars"), Confirming("HEXWARS"), out _),
                Is.False, "a database name is case sensitive, so the confirmation has to be too");
            Assert.That(DisposableDatabaseGuard.IsDisposable(Url("hexwars"), Confirming(""), out _), Is.False);
        }

        [TestCase("")]
        [TestCase("   ")]
        [TestCase("not a connection string")]
        [TestCase("Host=db.example.com;Username=appuser")]
        public void AValueThatNamesNoDatabase_IsRefused(string target)
        {
            Assert.That(DisposableDatabaseGuard.IsDisposable(target, NoEnvironment, out _), Is.False);
            Assert.That(DisposableDatabaseGuard.DatabaseName(target), Is.Null);
        }

        [Test]
        public void TheDatabaseTheFixtureWouldConnectTo_IsTheOneThatGetsChecked()
        {
            Assert.That(DisposableDatabaseGuard.DatabaseName(Url("hexwars_test")), Is.EqualTo("hexwars_test"));
            Assert.That(DisposableDatabaseGuard.DatabaseName(KeyValue("hexwars_test")),
                Is.EqualTo("hexwars_test"));
        }

        [Test]
        public void TheContainerTheFixtureStarts_PassesTheSameRuleAsEverythingElse()
        {
            Assert.That(
                DisposableDatabaseGuard.IsDisposable(
                    KeyValue(PostgresTestDatabase.ContainerDatabase), NoEnvironment, out string why),
                Is.True,
                "the container path is not exempt from the rule, so its database has to satisfy it: " + why);
        }

        [Test]
        public void TheFixture_RefusesADatabaseThatIsNotMarkedDisposable()
        {
            var refused = Assert.Throws<InvalidOperationException>(
                () => PostgresTestDatabase.RequireDisposable(Url("hexwars"), NoEnvironment));

            Assert.That(refused!.Message, Does.Contain("hexwars"),
                "the refusal has to name the database or nobody can tell which one it means");
            Assert.That(refused.Message, Does.Not.Contain(Password), "and never the password");
            Assert.That(refused.Message, Does.Not.Contain(Username), "nor the username");
            Assert.That(refused.Message,
                Does.Contain(DisposableDatabaseGuard.ConfirmationEnvironmentVariable),
                "and it has to say how to confirm a database on purpose");
        }

        [Test]
        public void TheFixture_AcceptsATestDatabaseAndHandsBackItsConnectionString()
        {
            string connectionString =
                PostgresTestDatabase.RequireDisposable(Url("hexwars_test"), NoEnvironment);

            Assert.That(connectionString, Does.Contain("hexwars_test"));
            Assert.That(DisposableDatabaseGuard.DatabaseName(connectionString), Is.EqualTo("hexwars_test"));
        }
    }
}
