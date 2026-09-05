using HexWars.NetServer.Tests.Fixtures;
using NUnit.Framework;
using ServerGuard = HexWars.NetServer.Operations.DisposableDatabaseGuard;

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
            + ";Database=\"" + database + "\"";

        [TestCase("hexwars_test")]
        [TestCase("test")]
        [TestCase("test_hexwars")]
        [TestCase("a_test_b")]
        [TestCase("HEXWARS_TEST")]
        [TestCase("ci-test")]
        [TestCase("test-bed")]
        [TestCase("ci-test-bed")]
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
        [TestCase("contest")]
        [TestCase("latest")]
        [TestCase("protest_db")]
        [TestCase("HexWarsTEST")]
        [TestCase("ci_TestBed")]
        public void ADatabaseWhoseNameDoesNot_IsRefused(string database)
        {
            // The last five are why the rule is a word and not a substring. Every one of them carries the
            // letters t-e-s-t and none of them is a database anybody meant to hand to something that drops
            // schemas: "contest" and "latest" are ordinary English, and a "TestBed" that was never
            // delimited is indistinguishable from them to a search that only looks for the letters.
            Assert.That(DisposableDatabaseGuard.IsDisposable(Url(database), NoEnvironment, out _), Is.False);
            Assert.That(DisposableDatabaseGuard.IsDisposable(KeyValue(database), NoEnvironment, out _),
                Is.False, "the key=value form is the same target and must get the same answer");
        }

        [TestCase("hexwars_test", true)]
        [TestCase("test", true)]
        [TestCase("test_hexwars", true)]
        [TestCase("a_test_b", true)]
        [TestCase("contest", false)]
        [TestCase("latest", false)]
        [TestCase("protest_db", false)]
        [TestCase("hexwars", false)]
        public void TheServerCopyAndTheFixtureCopy_AnswerIdentically(string database, bool disposable)
        {
            // Two copies of one rule, kept identical on purpose because the server assembly cannot
            // reference the test project. A drift between them is a self-test that would drop a schema the
            // test fixture would have refused, or the other way round.
            Assert.That(ServerGuard.IsDisposable(Url(database), NoEnvironment, out string server),
                Is.EqualTo(disposable), server);
            Assert.That(DisposableDatabaseGuard.IsDisposable(Url(database), NoEnvironment, out string fixture),
                Is.EqualTo(disposable), fixture);
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
        public void TheFixture_AcceptsATestDatabaseAndHandsBackBothFormsOfIt()
        {
            (string connectionString, string url) =
                PostgresTestDatabase.RequireDisposable(Url("hexwars_test"), NoEnvironment);

            Assert.That(DisposableDatabaseGuard.DatabaseName(connectionString), Is.EqualTo("hexwars_test"));
            Assert.That(DisposableDatabaseGuard.DatabaseName(url), Is.EqualTo("hexwars_test"));
        }

        [TestCase("prod?_test")]
        [TestCase("test_/prod")]
        [TestCase("test_#prod")]
        [TestCase("test_ prod")]
        public void ADatabaseNameCarryingAUrlDelimiter_IsCheckedAndHandedOutAsTheSameDatabase(string database)
        {
            // The fixture hands out a URL as well as a connection string, and host tests feed that URL
            // into DATABASE_URL. A name like "prod?_test" satisfies the name rule, but dropped into a URL
            // unescaped the question mark starts a query string and the URL reads back as "prod": the
            // database that was approved and the database that would have been migrated are not the same
            // one, and the second is somebody's production data.
            string target = KeyValue(database);
            Assert.That(DisposableDatabaseGuard.DatabaseName(target), Is.EqualTo(database),
                "the check has to be about the database Npgsql resolves, not the raw text");

            (string connectionString, string url) =
                PostgresTestDatabase.RequireDisposable(target, NoEnvironment);

            Assert.That(DisposableDatabaseGuard.DatabaseName(connectionString), Is.EqualTo(database));
            Assert.That(DisposableDatabaseGuard.DatabaseName(url), Is.EqualTo(database),
                "the URL form has to name the database that was approved and no other");
        }
    }
}
