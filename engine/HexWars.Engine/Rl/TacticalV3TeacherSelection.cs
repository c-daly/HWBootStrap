using System;

namespace HexWars.Engine.Rl
{
    public sealed class TacticalV3TeacherSelection
    {
        public TacticalV3TeacherSelection(
            long decisionId,
            int candidateId,
            int searchDepth,
            int expansionBudget,
            int actualExpansions,
            string heuristicIdentity)
        {
            if (decisionId < 0) throw new ArgumentOutOfRangeException(nameof(decisionId));
            if (candidateId < 0) throw new ArgumentOutOfRangeException(nameof(candidateId));
            if (searchDepth < 0) throw new ArgumentOutOfRangeException(nameof(searchDepth));
            if (expansionBudget < 0)
                throw new ArgumentOutOfRangeException(nameof(expansionBudget));
            if (actualExpansions < 0 || actualExpansions > expansionBudget)
                throw new ArgumentOutOfRangeException(nameof(actualExpansions));
            if (string.IsNullOrEmpty(heuristicIdentity))
                throw new ArgumentException(
                    "heuristic identity must not be empty", nameof(heuristicIdentity));

            DecisionId = decisionId;
            CandidateId = candidateId;
            SearchDepth = searchDepth;
            ExpansionBudget = expansionBudget;
            ActualExpansions = actualExpansions;
            HeuristicIdentity = heuristicIdentity;
        }

        public long DecisionId { get; }
        public int CandidateId { get; }
        public int SearchDepth { get; }
        public int ExpansionBudget { get; }
        public int ActualExpansions { get; }
        public string HeuristicIdentity { get; }
    }
}
