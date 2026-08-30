using System;
using System.Collections.Generic;
using HexWars.Engine;
using HexWars.Engine.Rl;

namespace HexWars.Presentation
{
    public readonly struct ModelDuelView
    {
        public readonly float[] Observation;
        public readonly bool[] ActionMask;
        public readonly int Seat;
        public readonly int Winner;
        public readonly bool Terminated;
        public readonly bool Truncated;
        public readonly bool DeploymentComplete;
        public readonly TacticalV3View StructuredDecision;

        public ModelDuelView(float[] observation, bool[] actionMask, int seat, int winner,
            bool terminated, bool truncated, bool deploymentComplete)
        {
            Observation = observation;
            ActionMask = actionMask;
            Seat = seat;
            Winner = winner;
            Terminated = terminated;
            Truncated = truncated;
            DeploymentComplete = deploymentComplete;
            StructuredDecision = null;
        }

        public ModelDuelView(TacticalV3View decision)
        {
            if (decision == null) throw new ArgumentNullException(nameof(decision));
            Observation = null;
            ActionMask = null;
            Seat = (int)decision.Seat;
            Winner = decision.Winner;
            Terminated = decision.Terminated;
            Truncated = decision.Truncated;
            DeploymentComplete = true;
            StructuredDecision = decision;
        }
    }

    public interface IModelDuelEnvironment
    {
        MlEnvironmentContract Environment { get; }
        ModelDuelContractIdentity ContractIdentity { get; }
        GameState CurrentState { get; }
        bool RequiresContinuation { get; }

        /// <summary>Opt-in (default false): when true, every accepted command captures a
        /// <see cref="DuelTransition"/> the viewer can drain and play back. The Unity arena driver
        /// turns this on for every duel it presents; headless training never touches it.</summary>
        bool CaptureTransitions { get; set; }

        ModelDuelView Reset(int seed, IAgent controller0, IAgent controller1);
        ModelDuelView Continue();

        /// <summary>Every captured accepted-command transition since the last drain (or Reset), in
        /// order. Draining advances a presentation cursor without removing replay history.</summary>
        IReadOnlyList<DuelTransition> DrainTransitions();
    }

    public interface ILegacyModelDuelEnvironment : IModelDuelEnvironment
    {
        MlContract Contract { get; }
        ModelDuelView Step(int action);
    }

    public interface IStructuredModelDuelEnvironment : IModelDuelEnvironment
    {
        TacticalV3Contract StructuredContract { get; }
        ModelDuelView Step(long decisionId, int candidateId);
    }

    public static class ModelDuelEnvironmentFactory
    {
        public static IModelDuelEnvironment Create(MlEnvironmentContract environment) =>
            Create(TrainingScenario.CreateStandard(
                MlEnvironmentContracts.CliValue(environment)));

        public static IModelDuelEnvironment Create(TrainingScenario scenario)
        {
            if (scenario == null) throw new ArgumentNullException(nameof(scenario));
            if (scenario.Environment == MlContract.AdaptiveVersion)
                return new AdaptiveModelDuelEnvironment(scenario.BuildAdaptive());
            if (scenario.Environment == MlContract.CurrentVersion)
                return new TacticalModelDuelEnvironment(scenario.BuildTactical());
            if (scenario.Environment == MlContract.TacticalV2Version)
                return new TacticalV2ModelDuelEnvironment(scenario.BuildTacticalV2());
            if (scenario.Environment == MlContract.TacticalV3Version)
                return new TacticalV3ModelDuelEnvironment(scenario.BuildTacticalV3());
            throw new ArgumentException(
                "scenario environment must be tactical-v1, tactical-v2, tactical-v3, or adaptive-v1",
                nameof(scenario));
        }

        public static ModelDuelContractIdentity ContractIdentity(MlEnvironmentContract environment)
        {
            return ContractIdentity(TrainingScenario.CreateStandard(
                MlEnvironmentContracts.CliValue(environment)));
        }

        public static ModelDuelContractIdentity ContractIdentity(TrainingScenario scenario)
        {
            var duel = Create(scenario);
            return duel.ContractIdentity;
        }
    }

    public enum ModelDuelRenderDirective { Suppress, Initialize, Update }

    public sealed class ModelDuelPresentationState
    {
        readonly MlEnvironmentContract _environment;
        bool _initialized;

        public ModelDuelPresentationState(MlEnvironmentContract environment)
        {
            _environment = environment;
        }

        public bool ShouldRender(bool deploymentComplete) =>
            _environment == MlEnvironmentContract.TacticalV1 ||
            _environment == MlEnvironmentContract.TacticalV2 ||
            _environment == MlEnvironmentContract.TacticalV3 ||
            deploymentComplete;

        public ModelDuelRenderDirective Next(bool deploymentComplete)
        {
            if (!ShouldRender(deploymentComplete)) return ModelDuelRenderDirective.Suppress;
            if (_initialized) return ModelDuelRenderDirective.Update;
            _initialized = true;
            return ModelDuelRenderDirective.Initialize;
        }
    }

    public static class ModelDuelContractCompatibility
    {
        public static IReadOnlyList<string> Validate(ModelDuelContractIdentity expected,
            bool p0IsModel, PolicySeatInfo p0, bool p1IsModel, PolicySeatInfo p1)
        {
            var errors = new List<string>();
            ValidateSeat(errors, "Seat 0", p0IsModel, p0, expected);
            ValidateSeat(errors, "Seat 1", p1IsModel, p1, expected);
            return errors;
        }

        static void ValidateSeat(List<string> errors, string label, bool isModel,
            PolicySeatInfo info, ModelDuelContractIdentity expected)
        {
            if (!isModel) return;
            if (info == null)
            {
                errors.Add(label + " model metadata is missing contract identity.");
                return;
            }
            if (string.IsNullOrWhiteSpace(info.Environment))
                errors.Add(label + " model metadata is missing environment.");
            else if (!string.Equals(info.Environment, expected.Environment, StringComparison.Ordinal))
                errors.Add(label + " environment " + info.Environment +
                    " does not match selected environment " + expected.Environment + ".");
            if (string.IsNullOrWhiteSpace(info.ContractVersion))
            {
                errors.Add(label + " model metadata is missing contract_version.");
            }
            else if (!string.Equals(info.ContractVersion, expected.Version, StringComparison.Ordinal))
                errors.Add(label + " contract " + info.ContractVersion +
                    " does not match selected environment " + expected.Version + ".");
            if (string.IsNullOrWhiteSpace(info.EncodingHash))
                errors.Add(label + " model metadata is missing encoding_hash.");
            else if (!string.Equals(info.EncodingHash, expected.EncodingHash, StringComparison.Ordinal))
                errors.Add(label + " encoding hash " + info.EncodingHash +
                    " does not match expected " + expected.EncodingHash + ".");
            if (!string.IsNullOrWhiteSpace(expected.CapacityHash))
            {
                if (string.IsNullOrWhiteSpace(info.CapacityHash))
                    errors.Add(label + " model metadata is missing capacity_hash.");
                else if (!string.Equals(
                    info.CapacityHash, expected.CapacityHash, StringComparison.Ordinal))
                    errors.Add(label + " capacity hash " + info.CapacityHash +
                        " does not match expected " + expected.CapacityHash + ".");
            }
        }
    }

    sealed class TacticalModelDuelEnvironment : ILegacyModelDuelEnvironment
    {
        readonly DuelEnv _environment;

        public TacticalModelDuelEnvironment(EnvConfig config)
        {
            if (config == null) throw new ArgumentNullException(nameof(config));
            _environment = new DuelEnv(config);
            Contract = MlContract.Create(config, MlEnvironmentKind.Duel);
            ContractIdentity = new ModelDuelContractIdentity(
                Contract.Version, Contract.Version, Contract.EncodingHash);
        }

        public MlEnvironmentContract Environment => MlEnvironmentContract.TacticalV1;
        public MlContract Contract { get; }
        public ModelDuelContractIdentity ContractIdentity { get; }
        public GameState CurrentState => _environment.State;
        public bool RequiresContinuation => false;

        public bool CaptureTransitions
        {
            get => _environment.CaptureTransitions;
            set => _environment.CaptureTransitions = value;
        }

        public ModelDuelView Reset(int seed, IAgent controller0, IAgent controller1) =>
            Convert(_environment.Reset(seed, controller0, controller1, PlayerId.Player0));

        public ModelDuelView Step(int action) => Convert(_environment.Step(action));

        public ModelDuelView Continue() => throw new InvalidOperationException(
            "the tactical environment has no pending continuation");

        public IReadOnlyList<DuelTransition> DrainTransitions() => _environment.DrainTransitions();

        static ModelDuelView Convert(DuelEnv.View view) => new ModelDuelView(
            view.Observation, view.ActionMask, view.Seat, view.Winner,
            view.Terminated, view.Truncated, deploymentComplete: true);
    }

    /// <summary>Arena adapter for tactical-v2, backed by <see cref="TacticalV2DuelEnv"/>. Tactical-v2 has
    /// no hidden deployment phase (rosters are snapshotted and placement is automatic/symmetric), so —
    /// like <see cref="TacticalModelDuelEnvironment"/> — it is immediately renderable from the first
    /// reset and never requires an external continuation.</summary>
    sealed class TacticalV2ModelDuelEnvironment : ILegacyModelDuelEnvironment
    {
        readonly TacticalV2DuelEnv _environment;

        public TacticalV2ModelDuelEnvironment(TacticalV2Config config)
        {
            if (config == null) throw new ArgumentNullException(nameof(config));
            _environment = new TacticalV2DuelEnv(config);
            Contract = MlContract.CreateTacticalV2(config, MlEnvironmentKind.Duel);
            ContractIdentity = new ModelDuelContractIdentity(
                Contract.Version, Contract.Version, Contract.EncodingHash);
        }

        public MlEnvironmentContract Environment => MlEnvironmentContract.TacticalV2;
        public MlContract Contract { get; }
        public ModelDuelContractIdentity ContractIdentity { get; }
        public GameState CurrentState => _environment.State;
        public bool RequiresContinuation => false;

        public bool CaptureTransitions
        {
            get => _environment.CaptureTransitions;
            set => _environment.CaptureTransitions = value;
        }

        public ModelDuelView Reset(int seed, IAgent controller0, IAgent controller1) =>
            Convert(_environment.Reset(seed, controller0, controller1, PlayerId.Player0));

        public ModelDuelView Step(int action) => Convert(_environment.Step(action));

        public ModelDuelView Continue() => throw new InvalidOperationException(
            "the tactical-v2 environment has no pending continuation");

        public IReadOnlyList<DuelTransition> DrainTransitions() => _environment.DrainTransitions();

        static ModelDuelView Convert(TacticalV2DuelEnv.View view) => new ModelDuelView(
            view.Observation, view.ActionMask, (int)view.Seat,
            view.Winner,
            view.Terminated, view.Truncated, deploymentComplete: true);
    }

    sealed class AdaptiveModelDuelEnvironment : ILegacyModelDuelEnvironment
    {
        readonly AdaptiveDuelEnv _environment;

        public AdaptiveModelDuelEnvironment(AdaptiveEnvConfig config)
        {
            _environment = new AdaptiveDuelEnv(
                config ?? throw new ArgumentNullException(nameof(config)));
        }

        public MlEnvironmentContract Environment => MlEnvironmentContract.AdaptiveV1;
        public MlContract Contract => _environment.Contract;
        public ModelDuelContractIdentity ContractIdentity => new ModelDuelContractIdentity(
            Contract.Version, Contract.Version, Contract.EncodingHash);
        public GameState CurrentState => _environment.DeploymentComplete ? _environment.State : null;
        public bool RequiresContinuation => _environment.AwaitingPostRevealAdvance;

        public bool CaptureTransitions
        {
            get => _environment.CaptureTransitions;
            set => _environment.CaptureTransitions = value;
        }

        public ModelDuelView Reset(int seed, IAgent controller0, IAgent controller1)
        {
            IDeploymentPolicy deployment0 = controller0 == null
                ? null : new CombinedArmsDeploymentPolicy(seed * 2 + 1);
            IDeploymentPolicy deployment1 = controller1 == null
                ? null : new CombinedArmsDeploymentPolicy(seed * 2 + 2);
            return Convert(_environment.Reset(seed, controller0, controller1,
                deployment0, deployment1, PlayerId.Player0));
        }

        public ModelDuelView Step(int action) => Convert(_environment.Step(action));

        public ModelDuelView Continue() => Convert(_environment.ContinueAfterReveal());

        public IReadOnlyList<DuelTransition> DrainTransitions() => _environment.DrainTransitions();

        static ModelDuelView Convert(AdaptiveDuelEnv.View view) => new ModelDuelView(
            view.Observation, view.ActionMask, view.Seat, view.Winner,
            view.Terminated, view.Truncated, view.DeploymentComplete);
    }

    sealed class TacticalV3ModelDuelEnvironment : IStructuredModelDuelEnvironment
    {
        readonly TacticalV3DuelEnv _environment;

        public TacticalV3ModelDuelEnvironment(TacticalV3Config config)
        {
            if (config == null) throw new ArgumentNullException(nameof(config));
            _environment = new TacticalV3DuelEnv(config);
            StructuredContract = TacticalV3Contract.Create(config, MlEnvironmentKind.Duel);
            ContractIdentity = new ModelDuelContractIdentity(
                StructuredContract.Version,
                StructuredContract.Version,
                StructuredContract.EncodingHash,
                StructuredContract.CapacityHash);
        }

        public MlEnvironmentContract Environment => MlEnvironmentContract.TacticalV3;
        public TacticalV3Contract StructuredContract { get; }
        public ModelDuelContractIdentity ContractIdentity { get; }
        public GameState CurrentState => _environment.State;
        public bool RequiresContinuation => false;
        public bool CaptureTransitions { get; set; }

        public ModelDuelView Reset(int seed, IAgent controller0, IAgent controller1)
        {
            ModelDuelView view = new ModelDuelView(_environment.Reset(
                seed, controller0, controller1, PlayerId.Player0));
            DiscardUncapturedTransitions();
            return view;
        }

        public ModelDuelView Step(long decisionId, int candidateId)
        {
            ModelDuelView view =
                new ModelDuelView(_environment.Step(decisionId, candidateId));
            DiscardUncapturedTransitions();
            return view;
        }

        public ModelDuelView Step(int action) => throw new InvalidOperationException(
            "tactical-v3 requires decision and candidate identity stepping");

        public ModelDuelView Continue() => throw new InvalidOperationException(
            "the tactical-v3 environment has no pending continuation");

        public IReadOnlyList<DuelTransition> DrainTransitions() =>
            _environment.DrainTransitions();

        void DiscardUncapturedTransitions()
        {
            if (!CaptureTransitions) _environment.DrainTransitions();
        }
    }
}
