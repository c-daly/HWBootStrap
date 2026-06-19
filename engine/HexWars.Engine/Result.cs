namespace HexWars.Engine
{
    /// <summary>
    /// The outcome of <see cref="GameEngine.Apply"/>. On success, <see cref="NewState"/> is the new
    /// (post-command) state; on rejection, <see cref="NewState"/> is the unchanged input state and
    /// <see cref="Reason"/> explains why.
    /// </summary>
    public sealed class Result
    {
        public bool Success { get; }
        public RejectionReason Reason { get; }
        public GameState NewState { get; }

        private Result(bool success, RejectionReason reason, GameState newState)
        {
            Success = success;
            Reason = reason;
            NewState = newState;
        }

        public static Result Ok(GameState newState) => new Result(true, RejectionReason.None, newState);
        public static Result Reject(GameState original, RejectionReason reason) => new Result(false, reason, original);
    }
}
