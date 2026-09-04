using HexWars.NetServer.Configuration;
using HexWars.NetServer.Persistence;
using Npgsql;
using NUnit.Framework;

namespace HexWars.NetServer.Tests
{
    /// <summary>
    /// DATABASE_URL is parsed by Npgsql itself, not by a hand-rolled semicolon split. These cases are the
    /// ones a naive splitter gets wrong: a quoted value containing a semicolon, an invalid port, and a
    /// keyword Npgsql does not support. Getting the first one wrong put a password fragment into the
    /// startup log, because the splitter read Port out of the middle of the password.
    /// </summary>
    [TestFixture]
    public class DatabaseUrlTests
    {
        /// <summary>A valid ADO.NET string whose Password is one quoted value. A semicolon split reads
        /// PASSWORD-LEAK as the port and prints it.</summary>
        const string QuotedPassword =
            "Host=db.internal;Database=hexwars;Password=\"foo;Port=PASSWORD-LEAK;X=bar\"";

        [Test]
        public void QuotedPasswordContainingSemicolons_IsAcceptedAndNeverDescribedAsThePort()
        {
            Assert.DoesNotThrow(() => DatabaseUrl.ToNpgsqlConnectionString(QuotedPassword));

            Assert.That(DatabaseUrl.DescribeTarget(QuotedPassword), Is.EqualTo("db.internal:5432/hexwars"));
        }

        [Test]
        public void QuotedPasswordContainingSemicolons_PassesConfigurationValidationAndStaysOutOfTheReport()
        {
            var settings = TestConfig.ValidSteamSettings();
            settings["DATABASE_URL"] = QuotedPassword;

            var result = HexWarsConfiguration.Read(TestConfig.Config(settings), TestConfig.Env("Production"));

            Assert.That(result.IsValid, Is.True, string.Join(" | ", result.Errors));
            string json = EnvironmentReport.Describe(result.Steam, result.Match, TestConfig.Env("Production")).ToJson();
            Assert.That(json, Does.Contain("db.internal:5432/hexwars"));
            Assert.That(json, Does.Not.Contain("PASSWORD-LEAK"));
            Assert.That(json, Does.Not.Contain("foo"));
        }

        [Test]
        public void QuotedPasswordContainingASemicolon_RoundTripsAsOneValue()
        {
            string cs = DatabaseUrl.ToNpgsqlConnectionString(
                "Host=db;Database=x;Password=\"valid;password\"");

            Assert.That(new NpgsqlConnectionStringBuilder(cs).Password, Is.EqualTo("valid;password"));
        }

        [TestCase("Host=db;Database=x;Port=not-a-port", TestName = "InvalidPortIsRejected")]
        [TestCase("Host=db;Database=x;MadeUp=y", TestName = "UnsupportedKeywordIsRejected")]
        public void NpgsqlRejectsWhatItCannotUse(string raw)
        {
            Assert.Throws<FormatException>(() => DatabaseUrl.ToNpgsqlConnectionString(raw));
            Assert.That(DatabaseUrl.DescribeTarget(raw), Is.EqualTo("invalid"));
        }

        [Test]
        public void RejectionMessage_NamesTheProblemWithoutEchoingAValue()
        {
            var ex = Assert.Throws<FormatException>(() => DatabaseUrl.ToNpgsqlConnectionString(
                "Host=db;Database=x;Password=\"unterminated-SECRETVALUE"));

            Assert.That(ex!.Message, Does.StartWith("DATABASE_URL is not a valid Npgsql connection string:"));
            Assert.That(ex.Message, Does.Not.Contain("SECRETVALUE"));
        }

        [Test]
        public void UriForm_RoundTripsAnEncodedPasswordIntoTheBuilder()
        {
            string cs = DatabaseUrl.ToNpgsqlConnectionString(
                "postgres://hex%40user:p%40ss%3Aw0rd@db.internal:6432/hexwars?sslmode=require");

            var builder = new NpgsqlConnectionStringBuilder(cs);
            Assert.That(builder.Host, Is.EqualTo("db.internal"));
            Assert.That(builder.Port, Is.EqualTo(6432));
            Assert.That(builder.Database, Is.EqualTo("hexwars"));
            Assert.That(builder.Username, Is.EqualTo("hex@user"));
            Assert.That(builder.Password, Is.EqualTo("p@ss:w0rd"));
            Assert.That(builder.SslMode, Is.EqualTo(SslMode.Require));
            Assert.That(cs, Does.Contain("Trust Server Certificate=True"));
        }

        [Test]
        public void UriForm_UnknownSslModeIsRejected()
        {
            Assert.Throws<FormatException>(() =>
                DatabaseUrl.ToNpgsqlConnectionString("postgres://u:p@h.invalid/d?sslmode=bogus"));
        }

        [Test]
        public void KeyValueForm_IsNormalisedThroughNpgsqlAndKeepsTheAliases()
        {
            Assert.That(DatabaseUrl.ToNpgsqlConnectionString("  Server=db;DB=x;Username=u;Password=p  "),
                Is.EqualTo("Host=db;Database=x;Username=u;Password=p"));
            Assert.That(DatabaseUrl.DescribeTarget("Server=db;DB=x;Username=u;Password=p"),
                Is.EqualTo("db:5432/x"));
        }

        [Test]
        public void DescribeTarget_NeverIncludesTheCredentials()
        {
            string target = DatabaseUrl.DescribeTarget(
                "postgres://hexwars:s3cr3t@db.internal:6432/hexwars?sslmode=require");

            Assert.That(target, Is.EqualTo("db.internal:6432/hexwars"));
            Assert.That(target, Does.Not.Contain("s3cr3t"));
            Assert.That(target, Does.Not.Contain("hexwars:"));
        }
    }
}
