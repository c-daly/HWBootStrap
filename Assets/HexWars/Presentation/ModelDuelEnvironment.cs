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
        }
    }

    public interface IModelDuelEnvironment
    {
        MlEnvironmentContract Environment { get; }
        MlContract Contract { get; }
        GameState CurrentState { get; }
        bool RequiresContinuation { get; }
        ModelDuelView Reset(int seed, IAgent controller0, IAgent controller1);
        ModelDuelView Step(int action);
        ModelDuelView Continue();
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
            throw new ArgumentException(
                "scenario environment must be tactical-v1 or adaptive-v1",
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
            return new ModelDuelContractIdentity(
                scenario.Environment,
                duel.Contract.Version,
                duel.Contract.EncodingHash);
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
            _environment == MlEnvironmentContract.TacticalV1 || deploymentComplete;

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
        }
    }

    sealed class TacticalModelDuelEnvironment : IModelDuelEnvironment
    {
        readonly DuelEnv _environment;

        public TacticalModelDuelEnvironment(EnvConfig config)
        {
            if (config == null) throw new ArgumentNullException(nameof(config));
            _environment = new DuelEnv(config);
            Contract = MlContract.Create(config, MlEnvironmentKind.Duel);
        }

        public MlEnvironmentContract Environment => MlEnvironmentContract.TacticalV1;
        public MlContract Contract { get; }
        public GameState CurrentState => _environment.State;
        public bool RequiresContinuation => false;

        public ModelDuelView Reset(int seed, IAgent controller0, IAgent controller1) =>
            Convert(_environment.Reset(seed, controller0, controller1, PlayerId.Player0));

        public ModelDuelView Step(int action) => Convert(_environment.Step(action));

        public ModelDuelView Continue() => throw new InvalidOperationException(
            "the tactical environment has no pending continuation");

        static ModelDuelView Convert(DuelEnv.View view) => new ModelDuelView(
            view.Observation, view.ActionMask, view.Seat, view.Winner,
            view.Terminated, view.Truncated, deploymentComplete: true);
    }

    sealed class AdaptiveModelDuelEnvironment : IModelDuelEnvironment
    {
        readonly AdaptiveDuelEnv _environment;

        public AdaptiveModelDuelEnvironment(AdaptiveEnvConfig config)
        {
            _environment = new AdaptiveDuelEnv(
                config ?? throw new ArgumentNullException(nameof(config)));
        }

        public MlEnvironmentContract Environment => MlEnvironmentContract.AdaptiveV1;
        public MlContract Contract => _environment.Contract;
        public GameState CurrentState => _environment.DeploymentComplete ? _environment.State : null;
        public bool RequiresContinuation => _environment.AwaitingPostRevealAdvance;

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

        static ModelDuelView Convert(AdaptiveDuelEnv.View view) => new ModelDuelView(
            view.Observation, view.ActionMask, view.Seat, view.Winner,
            view.Terminated, view.Truncated, view.DeploymentComplete);
    }
}
