using HexWars.NetServer.Auth;

namespace HexWars.NetServer.Tests.Fakes
{
    /// <summary>
    /// A credential service that forwards everything, and can be made to throw once on validation.
    ///
    /// The coordinator catches the failures it expects and turns them into a fail code of its own, so an
    /// exception from the credential lookup is by construction the one thing it did not anticipate. That
    /// is the only way to reach the internal stage of the authentication counter, and a counter nothing
    /// can reach is a counter nobody can trust.
    /// </summary>
    public sealed class FaultInjectingCredentialService(IMatchCredentialService inner)
        : IMatchCredentialService
    {
        Exception? _nextValidateFailure;

        /// <summary>Arms the next ValidateAsync to throw, once. Null disarms it.</summary>
        public void FailNextValidate(Exception? failure) => _nextValidateFailure = failure;

        public Task<IssuedCredential> IssueAsync(Guid matchId, string steamId, CancellationToken ct) =>
            inner.IssueAsync(matchId, steamId, ct);

        public Task<CredentialValidation?> ValidateAsync(Guid matchId, string credential, CancellationToken ct)
        {
            Exception? failure = _nextValidateFailure;
            if (failure is null) return inner.ValidateAsync(matchId, credential, ct);

            _nextValidateFailure = null;
            throw failure;
        }

        public Task<bool> IsStillValidAsync(
            byte[] credentialHash, Guid matchId, DateTimeOffset now, CancellationToken ct) =>
            inner.IsStillValidAsync(credentialHash, matchId, now, ct);
    }
}
