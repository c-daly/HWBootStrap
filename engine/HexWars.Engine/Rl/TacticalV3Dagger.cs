using System;
using System.Security.Cryptography;
using System.Text;

namespace HexWars.Engine.Rl
{
    public sealed class TacticalV3SelectiveDaggerInspection
    {
        public TacticalV3SelectiveDaggerInspection(
            long decisionId,
            int learnerCandidateId,
            DaggerEligibilityReason reasons,
            string stateHash,
            int stateOccurrence,
            double normalizedAdvantage,
            int opponentLivingUnitCount,
            int productiveLegalActionCount)
        {
            if (learnerCandidateId < 0)
                throw new ArgumentOutOfRangeException(nameof(learnerCandidateId));
            if (string.IsNullOrEmpty(stateHash))
                throw new ArgumentException("state hash must not be empty", nameof(stateHash));
            if (stateOccurrence < 1)
                throw new ArgumentOutOfRangeException(nameof(stateOccurrence));
            if (double.IsNaN(normalizedAdvantage) || double.IsInfinity(normalizedAdvantage))
                throw new ArgumentOutOfRangeException(nameof(normalizedAdvantage));
            if (opponentLivingUnitCount < 0)
                throw new ArgumentOutOfRangeException(nameof(opponentLivingUnitCount));
            if (productiveLegalActionCount < 0)
                throw new ArgumentOutOfRangeException(nameof(productiveLegalActionCount));

            DecisionId = decisionId;
            LearnerCandidateId = learnerCandidateId;
            Reasons = reasons;
            StateHash = stateHash;
            StateOccurrence = stateOccurrence;
            NormalizedAdvantage = normalizedAdvantage;
            OpponentLivingUnitCount = opponentLivingUnitCount;
            ProductiveLegalActionCount = productiveLegalActionCount;
        }

        public long DecisionId { get; }
        public int LearnerCandidateId { get; }
        public DaggerEligibilityReason Reasons { get; }
        public string StateHash { get; }
        public int StateOccurrence { get; }
        public double NormalizedAdvantage { get; }
        public int OpponentLivingUnitCount { get; }
        public int ProductiveLegalActionCount { get; }

        internal static string HashState(GameState state)
        {
            string key = SelectiveDaggerObserver.CanonicalStateKey(state);
            byte[] bytes = Encoding.UTF8.GetBytes(key);
            using SHA256 sha = SHA256.Create();
            return BitConverter.ToString(sha.ComputeHash(bytes))
                .Replace("-", string.Empty).ToLowerInvariant();
        }

        internal static double Material(GameState state, PlayerId seat, float pointsWeight)
        {
            PlayerState player = state.Player(seat);
            double material = pointsWeight * player.Points;
            foreach (Unit unit in player.UnitsOnBoard)
                if (unit.IsAlive)
                    material += unit.Stats.PointCost * (double)unit.CurrentHp /
                        unit.Stats.Health;
            return material;
        }
    }
}
