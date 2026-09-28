using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using HexWars.Engine;
using HexWars.Engine.Rl;
using Newtonsoft.Json;
using UnityEngine;

namespace HexWars.Presentation
{
    /// <summary>One pinned model, one seat and one cancellable inference stream for a match.</summary>
    public sealed class AiPolicySession : IDisposable
    {
        public const int DecisionTimeoutMs = 30000;
        // Hosted calls may include queueing (30s), cold startup (90s), and inference (30s).
        public const int HostedRequestTimeoutMs = 155000;
        readonly AiModelSelection _selection;
        readonly PlayerId _seat;
        readonly IAgent _scripted;
        PolicyBridge _bridge;
        PlayableModelAdapter _adapter;
        long _decisionId;
        bool _disposed;

        public AiPolicySession(AiModelSelection selection, PlayerId seat, int seed)
        {
            _selection = selection ?? throw new ArgumentNullException(nameof(selection));
            _seat = seat;
            if (selection.Model.kind == "greedy") _scripted = new GreedyAgent(seed);
            else if (selection.Model.kind == "random") _scripted = new RandomAgent(seed);
            else if (!selection.Model.IsTrained) throw new ArgumentException("Unknown configured AI kind.");
        }

        public string ModelLabel => _selection.DisplayName;
        public bool IsTrained => _selection.Model.IsTrained;

        public async Task<Command> DecideAsync(GameState state, CancellationToken cancellation)
        {
            cancellation.ThrowIfCancellationRequested();
            if (_disposed) throw new ObjectDisposedException(nameof(AiPolicySession));
            if (state.ActivePlayer != _seat) throw new InvalidOperationException("AI request is for another seat.");
            if (_scripted != null) return _scripted.Decide(state);
            if (_adapter == null) await InitializeAsync(state, cancellation);
            cancellation.ThrowIfCancellationRequested();
            if (_disposed) throw new OperationCanceledException("AI match ended.");
            var frame = _adapter.CreateFrame(state, _seat, _decisionId++);
            var payload = TacticalV3PolicyPayload.From(frame);
            PolicyCandidateResult selected;
            if (_selection.UsesHttp)
            {
                selected = await SelectHostedAsync(_selection, (int)_seat, payload, cancellation);
            }
            else
                selected = await _bridge.ActStructuredAsync((int)_seat, payload, DecisionTimeoutMs, cancellation);
            cancellation.ThrowIfCancellationRequested();
            if (_disposed) throw new OperationCanceledException("AI match ended.");
            return _adapter.Resolve(frame, selected, state);
        }

        async Task InitializeAsync(GameState state, CancellationToken cancellation)
        {
            var adapter = new PlayableModelAdapter(state, _seat);
            var expected = adapter.ContractIdentity;
            if (_selection.Model.contract_version != expected.Version ||
                _selection.Model.encoding_hash != expected.EncodingHash || _selection.Model.capacity_hash != expected.CapacityHash)
                throw new InvalidOperationException("The selected AI model is incompatible with this match.");
            if (!_selection.UsesHttp)
            {
#if UNITY_WEBGL && !UNITY_EDITOR
                throw new InvalidOperationException("Browser AI requires the configured inference server.");
#else
                string project = Directory.GetParent(Application.dataPath)?.FullName
                    ?? throw new InvalidOperationException("AI runtime root is unavailable.");
                var launch = PlayableModelResolver.Resolve(project, _selection);
                _bridge = new PolicyBridge();
                bool ready = await _bridge.StartAsync(launch.PythonExecutable, launch.ServerScript,
                    _seat == PlayerId.Player0 ? launch.ControllerSpec : null,
                    _seat == PlayerId.Player1 ? launch.ControllerSpec : null,
                    launch.WorkingDirectory, expected.Environment, expected.Version, expected.EncodingHash,
                    expected.CapacityHash, PolicyBridge.DefaultStartupTimeoutMs, cancellation);
                cancellation.ThrowIfCancellationRequested();
                if (_disposed) throw new OperationCanceledException("AI match ended.");
                if (!ready) throw new InvalidOperationException("The configured AI model could not start. " + _bridge.StderrTail);
                var errors = ModelDuelContractCompatibility.Validate(expected, _seat == PlayerId.Player0,
                    _bridge.Seat0, _seat == PlayerId.Player1, _bridge.Seat1);
                if (errors.Count != 0) throw new InvalidOperationException(string.Join(" ", errors));
                var seat = _seat == PlayerId.Player0 ? _bridge.Seat0 : _bridge.Seat1;
                if (seat == null || seat.CheckpointSha256 != _selection.Model.checkpoint_sha256)
                    throw new InvalidOperationException("The loaded AI checkpoint differs from the selected model.");
#endif
            }
            _adapter = adapter;
        }

        internal static async Task<PolicyCandidateResult> SelectHostedAsync(AiModelSelection selection,
            int seat, TacticalV3ViewDto payload, CancellationToken cancellation)
        {
            string json = JsonConvert.SerializeObject(new DecisionRequest
            {
                difficulty_id = selection.DifficultyId, model_id = selection.Model.id, catalog_revision = selection.CatalogRevision,
                checkpoint_sha256 = selection.Model.checkpoint_sha256, seat = seat, decision = payload,
            });
            string response = await AiHttp.RequestAsync(selection.ServerUrl + selection.InferencePath,
                json, HostedRequestTimeoutMs, cancellation);
            return ParseHostedSelection(response, selection, payload.decision_id);
        }

        internal static PolicyCandidateResult ParseHostedSelection(string json, AiModelSelection selection, long decisionId)
        {
            var response = JsonConvert.DeserializeObject<DecisionResponse>(json, AiModelSettings.JsonSettings);
            if (response == null || response.difficulty_id != selection.DifficultyId || response.model_id != selection.Model.id ||
                response.catalog_revision != selection.CatalogRevision ||
                response.checkpoint_sha256 != selection.Model.checkpoint_sha256)
                throw new InvalidOperationException("The AI server returned a different configured model.");
            if (response.decision_id != decisionId || response.candidate_id < 0)
                throw new InvalidOperationException("The AI server returned an invalid or stale decision.");
            return new PolicyCandidateResult { DecisionId = response.decision_id, CandidateId = response.candidate_id };
        }

        internal static void RequireCurrentState(GameState requested, GameState current)
        {
            if (!ReferenceEquals(requested, current))
                throw new OperationCanceledException("The game changed while the AI was deciding.");
        }

        public void Dispose()
        {
            _disposed = true;
            _bridge?.Abort();
            _bridge = null;
        }

        sealed class DecisionRequest
        {
            public string difficulty_id, model_id, catalog_revision, checkpoint_sha256;
            public int seat;
            public TacticalV3ViewDto decision;
        }

        sealed class DecisionResponse
        {
            [JsonProperty(Required = Required.Always)] public string difficulty_id;
            [JsonProperty(Required = Required.Always)] public string model_id;
            [JsonProperty(Required = Required.Always)] public string catalog_revision;
            [JsonProperty(Required = Required.Always)] public string checkpoint_sha256;
            [JsonProperty(Required = Required.Always)] public long decision_id;
            [JsonProperty(Required = Required.Always)] public int candidate_id;
        }
    }
}
