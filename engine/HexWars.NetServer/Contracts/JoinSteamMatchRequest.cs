namespace HexWars.NetServer.Contracts
{
    /// <summary>
    /// What a lobby member sends to enter, or re-enter, a match the owner already allocated. The match is
    /// named by the route, so the body carries only the proof of who is asking.
    /// </summary>
    /// <param name="Ticket">A GetAuthTicketForWebApi ticket in hex, issued for the identity hexwars-match.</param>
    public sealed record JoinSteamMatchRequest(string? Ticket);
}
