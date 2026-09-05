namespace HexWars.NetServer.Contracts
{
    /// <summary>
    /// What a lobby owner sends to allocate a match.
    ///
    /// Note what is NOT here: no roster, no seat, no Steam id. The only thing that says who is calling is
    /// the ticket, which the server presents to Valve; a body-supplied id would be a claim nobody vouched
    /// for, and accepting one would let any caller name themselves the owner of any lobby.
    ///
    /// Every member is nullable because a JSON body is whatever the client sent: a missing field arrives
    /// as null whatever the declaration says, and pretending otherwise only hides the check.
    /// </summary>
    /// <param name="SteamLobbyId">The Steam lobby to allocate from. A lobby id, not an account id.</param>
    /// <param name="Ticket">A GetAuthTicketForWebApi ticket in hex, issued for the identity hexwars-match.</param>
    /// <param name="RequestedSetup">Optional: the setup the client believed the lobby carried. When it no
    /// longer matches, the request is refused rather than starting a game the players did not agree to.</param>
    public sealed record CreateSteamMatchRequest(string? SteamLobbyId, string? Ticket, string? RequestedSetup);
}
