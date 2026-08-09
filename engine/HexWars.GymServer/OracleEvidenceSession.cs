using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using HexWars.Engine;
using HexWars.Engine.Rl;

namespace HexWars.GymServer
{
    internal sealed class OracleEvidenceSession
    {
        internal const int MaxScheduleItems = 480;
        internal const int MaxArtifactBytes = 64 * 1024 * 1024;
        private readonly OracleEvidenceBeginRequest _request;
        private readonly OracleEvidenceRuntimeIdentity _runtime;
        private int _next;
        private string _chain;
        private bool _ended;

        private OracleEvidenceSession(OracleEvidenceBeginRequest request, OracleEvidenceRuntimeIdentity runtime,
            string sessionId, string initialChain)
        {
            _request = request;
            _runtime = runtime;
            SessionId = sessionId;
            _chain = initialChain;
        }

        internal string SessionId { get; }
        internal OracleEvidenceScheduleItem? Expected => _next < _request.CandidatesBySchedule.Count
            ? _request.CandidatesBySchedule[_next] : null;
        internal bool Ended => _ended;

        internal static OracleEvidenceBeginResponse Begin(OracleEvidenceBeginRequest request,
            OracleEvidenceRuntimeIdentity runtime)
        {
            if (request == null) throw new ArgumentNullException(nameof(request));
            if (runtime == null) throw new ArgumentNullException(nameof(runtime));
            request.Validate(runtime);
            string sessionId = Sha256(Encoding.UTF8.GetBytes("gymserver-evidence-v1|" + request.Nonce));
            byte[] body = CanonicalBegin(request, runtime, sessionId);
            string initial = Sha256(body);
            return new OracleEvidenceBeginResponse(new OracleEvidenceSession(request, runtime, sessionId, initial),
                request.Nonce, sessionId, request.PreflightScheduleSha256, initial, Sha256(body));
        }

        internal OracleEvidenceGameResponse CloseGame(OracleEvidenceGameContext context, byte[] traceUtf8,
            byte[] replayUtf8, byte[] benchmarkUtf8)
        {
            if (_ended) throw new InvalidDataException("evidence session is closed");
            if (context == null) throw new ArgumentNullException(nameof(context));
            OracleEvidenceScheduleItem expected = Expected ?? throw new InvalidDataException("evidence schedule is complete");
            context.Validate(SessionId, _request.Nonce, expected);
            ValidateArtifact(traceUtf8, "trace"); ValidateArtifact(replayUtf8, "replay");
            ValidateArtifact(benchmarkUtf8, "benchmark");
            var trace = new OracleEvidenceArtifact(traceUtf8);
            var replay = new OracleEvidenceArtifact(replayUtf8);
            var benchmark = new OracleEvidenceArtifact(benchmarkUtf8);
            int sequence = _next + 1;
            byte[] body = CanonicalReceipt(_request, _runtime, SessionId, sequence, _chain, expected,
                context, trace, replay, benchmark);
            string receiptHash = Sha256(body);
            var receipt = new OracleEvidenceReceipt(_request.Nonce, SessionId, sequence, _chain, expected,
                context, trace, replay, benchmark, receiptHash, body);
            _chain = receiptHash;
            _next++;
            return new OracleEvidenceGameResponse(receipt, trace, replay, benchmark);
        }

        internal OracleEvidenceEndResponse End(string sessionId, string nonce)
        {
            if (_ended) throw new InvalidDataException("evidence session is closed");
            if (!string.Equals(sessionId, SessionId, StringComparison.Ordinal) ||
                !string.Equals(nonce, _request.Nonce, StringComparison.Ordinal))
                throw new InvalidDataException("evidence end session or nonce does not match");
            if (Expected != null) throw new InvalidDataException("evidence schedule is incomplete");
            byte[] body = CanonicalEnd(SessionId, nonce, _next, _chain);
            _ended = true;
            return new OracleEvidenceEndResponse(SessionId, nonce, _next, _chain, Sha256(body));
        }

        internal static OracleEvidenceBeginRequest ParseBegin(JsonElement root)
        {
            RequireExact(root, "cmd", "schema_version", "purpose", "nonce", "panel_sha256", "repository",
                "scenario_sha256", "contract_hash", "encoding_hash", "oracle", "candidates",
                "preflight_schedule", "preflight_schedule_sha256", "candidates_by_schedule");
            RequireString(root, "cmd", "duel_evidence_begin"); RequireInt(root, "schema_version", 1);
            RequireString(root, "purpose", "oracle-preflight");
            string nonce = RequireHex(root, "nonce");
            string panel = RequireHex(root, "panel_sha256");
            JsonElement repository = root.GetProperty("repository");
            RequireExact(repository, "commit", "source_tree", "dirty");
            string commit = RequireGit(repository, "commit"); string tree = RequireGit(repository, "source_tree");
            if (repository.GetProperty("dirty").ValueKind != JsonValueKind.False)
                throw new InvalidDataException("repository dirty must be false");
            string scenario = RequireHex(root, "scenario_sha256");
            string contract = RequireHex(root, "contract_hash"); string encoding = RequireHex(root, "encoding_hash");
            OracleEvidenceOracle oracle = ParseOracle(root.GetProperty("oracle"), full: false);
            OracleEvidenceOracle[] candidates = root.GetProperty("candidates").EnumerateArray()
                .Select(element => ParseOracle(element, full: true)).ToArray();
            OracleEvidenceScheduledDuel[] schedule = root.GetProperty("preflight_schedule").EnumerateArray()
                .Select(ParseDuel).ToArray();
            string scheduleHash = RequireHex(root, "preflight_schedule_sha256");
            OracleEvidenceScheduleItem[] expanded = root.GetProperty("candidates_by_schedule").EnumerateArray()
                .Select(ParseScheduleItem).ToArray();
            return new OracleEvidenceBeginRequest(nonce, panel, commit, tree, scenario, contract, encoding, oracle,
                candidates, schedule, scheduleHash, expanded);
        }

        internal static (string SessionId, string Nonce) ParseSessionRequest(JsonElement root, string command)
        {
            RequireExact(root, "cmd", "schema_version", "session_id", "nonce");
            RequireString(root, "cmd", command); RequireInt(root, "schema_version", 1);
            return (RequireHex(root, "session_id"), RequireHex(root, "nonce"));
        }

        private static OracleEvidenceOracle ParseOracle(JsonElement element, bool full)
        {
            if (full) RequireExact(element, "oracle_type", "depth", "expansion_budget", "use_heuristic", "heuristic_identity", "code_hash");
            else RequireExact(element, "oracle_type", "heuristic_identity", "code_hash");
            RequireString(element, "oracle_type", "bounded-search");
            RequireString(element, "heuristic_identity", BoundedSearchAgent.HeuristicIdentity);
            int depth = full ? RequirePositive(element, "depth") : 0;
            int budget = full ? RequirePositive(element, "expansion_budget") : 0;
            if (full && element.GetProperty("use_heuristic").ValueKind != JsonValueKind.True)
                throw new InvalidDataException("oracle use_heuristic must be true");
            return new OracleEvidenceOracle(depth, budget, RequireHex(element, "code_hash"));
        }

        private static OracleEvidenceScheduledDuel ParseDuel(JsonElement element)
        {
            RequireExact(element, "schedule_index", "map_seed", "episode_seed", "profile", "reference_seat", "learner_seat");
            int reference = RequireSeat(element, "reference_seat"); int learner = RequireSeat(element, "learner_seat");
            return new OracleEvidenceScheduledDuel(RequireNonnegative(element, "schedule_index"),
                RequireNonnegative(element, "map_seed"), RequireNonnegative(element, "episode_seed"),
                RequireProfile(element, "profile"), reference, learner);
        }

        private static OracleEvidenceScheduleItem ParseScheduleItem(JsonElement element)
        {
            RequireExact(element, "candidate_index", "game_index", "oracle", "scheduled_duel");
            return new OracleEvidenceScheduleItem(RequireNonnegative(element, "candidate_index"),
                RequireNonnegative(element, "game_index"), ParseOracle(element.GetProperty("oracle"), true),
                ParseDuel(element.GetProperty("scheduled_duel")));
        }

        private static void RequireExact(JsonElement element, params string[] names)
        {
            if (element.ValueKind != JsonValueKind.Object || !element.EnumerateObject().Select(p => p.Name)
                .OrderBy(name => name, StringComparer.Ordinal).SequenceEqual(names.OrderBy(name => name, StringComparer.Ordinal)))
                throw new InvalidDataException("evidence request has unknown or missing fields");
        }
        private static string RequireHex(JsonElement element, string name)
        {
            string value = RequireString(element, name, null);
            if (value.Length != 64 || value.Any(ch => !(ch >= '0' && ch <= '9' || ch >= 'a' && ch <= 'f')))
                throw new InvalidDataException($"evidence {name} must be lowercase 64-hex");
            return value;
        }
        private static string RequireGit(JsonElement element, string name)
        {
            string value = RequireString(element, name, null);
            if ((value.Length != 40 && value.Length != 64) || value.Any(ch => !(ch >= '0' && ch <= '9' || ch >= 'a' && ch <= 'f')))
                throw new InvalidDataException($"evidence {name} must be a lowercase Git identity");
            return value;
        }
        private static void RequireInt(JsonElement element, string name, int expected) { if (!element.TryGetProperty(name, out JsonElement value) || !value.TryGetInt32(out int actual) || actual != expected) throw new InvalidDataException($"evidence {name} is invalid"); }
        private static string RequireString(JsonElement element, string name, string? expected)
        {
            if (!element.TryGetProperty(name, out JsonElement value) || value.ValueKind != JsonValueKind.String || value.GetString() is not string text ||
                (expected != null && !string.Equals(text, expected, StringComparison.Ordinal)))
                throw new InvalidDataException($"evidence {name} is invalid");
            return text;
        }
        private static int RequirePositive(JsonElement element, string name) { int value = RequireNonnegative(element, name); if (value < 1) throw new InvalidDataException($"evidence {name} must be positive"); return value; }
        private static int RequireNonnegative(JsonElement element, string name)
        { if (!element.TryGetProperty(name, out JsonElement value) || !value.TryGetInt32(out int number) || number < 0) throw new InvalidDataException($"evidence {name} must be nonnegative"); return number; }
        private static int RequireSeat(JsonElement element, string name) { int seat = RequireNonnegative(element, name); if (seat > 1) throw new InvalidDataException($"evidence {name} must be 0 or 1"); return seat; }
        private static string RequireProfile(JsonElement element, string name)
        { string profile = RequireString(element, name, null); if (string.IsNullOrWhiteSpace(profile)) throw new InvalidDataException("evidence profile is invalid"); return profile; }
        private static void ValidateArtifact(byte[] bytes, string name)
        { if (bytes == null || bytes.Length > MaxArtifactBytes) throw new InvalidDataException($"evidence {name} artifact exceeds its size limit"); }
        internal static string Sha256(byte[] bytes) { using var sha = SHA256.Create(); return Convert.ToHexString(sha.ComputeHash(bytes)).ToLowerInvariant(); }
        private static void WriteHash(Utf8JsonWriter writer, string name, string value) { writer.WriteString(name, value); }
        private static byte[] CanonicalBegin(OracleEvidenceBeginRequest r, OracleEvidenceRuntimeIdentity runtime, string session)
        { using var stream = new MemoryStream(); using (var w = new Utf8JsonWriter(stream)) { w.WriteStartObject(); w.WriteNumber("schema_version", 1); w.WriteString("purpose", "oracle-preflight"); WriteHash(w,"nonce",r.Nonce); WriteHash(w,"session_id",session); WriteHash(w,"schedule_sha256",r.PreflightScheduleSha256); w.WriteString("environment",runtime.Environment); WriteHash(w,"scenario_sha256",runtime.ScenarioSha256); WriteHash(w,"contract_hash",runtime.ContractHash); WriteHash(w,"encoding_hash",runtime.EncodingHash); WriteHash(w,"oracle_code_sha256",runtime.OracleCodeSha256); w.WriteEndObject(); } return stream.ToArray(); }
        private static byte[] CanonicalReceipt(OracleEvidenceBeginRequest r, OracleEvidenceRuntimeIdentity runtime, string session, int sequence, string previous, OracleEvidenceScheduleItem item, OracleEvidenceGameContext context, OracleEvidenceArtifact trace, OracleEvidenceArtifact replay, OracleEvidenceArtifact benchmark)
        { using var stream = new MemoryStream(); using (var w = new Utf8JsonWriter(stream)) { w.WriteStartObject(); w.WriteNumber("schema_version",1); WriteHash(w,"session_id",session); WriteHash(w,"nonce",r.Nonce); w.WriteNumber("sequence",sequence); WriteHash(w,"previous_receipt_sha256",previous); w.WriteNumber("candidate_index",item.CandidateIndex); w.WriteNumber("game_index",item.GameIndex); WriteDuel(w,item.Duel); WriteOracle(w,item.Oracle); WriteHash(w,"scenario_sha256",runtime.ScenarioSha256); WriteHash(w,"contract_hash",runtime.ContractHash); WriteHash(w,"encoding_hash",runtime.EncodingHash); w.WriteString("engine_protocol","gymserver-evidence-v1"); w.WriteString("outcome",context.Outcome); if(context.Winner.HasValue) w.WriteNumber("winner",context.Winner.Value); else w.WriteNull("winner"); w.WriteNumber("transition_count",context.TransitionCount); w.WriteNumber("benchmark_sample_count",context.BenchmarkSampleCount); w.WriteNumber("expansion_total",context.ExpansionTotal); WriteArtifact(w,"trace",trace); WriteArtifact(w,"replay",replay); WriteArtifact(w,"benchmark",benchmark); w.WriteEndObject(); } return stream.ToArray(); }
        private static byte[] CanonicalEnd(string session, string nonce, int count, string final) { using var stream=new MemoryStream(); using(var w=new Utf8JsonWriter(stream)){w.WriteStartObject();w.WriteNumber("schema_version",1);WriteHash(w,"session_id",session);WriteHash(w,"nonce",nonce);w.WriteNumber("receipt_count",count);WriteHash(w,"final_receipt_sha256",final);w.WriteEndObject();}return stream.ToArray(); }
        private static void WriteOracle(Utf8JsonWriter w, OracleEvidenceOracle value) { w.WritePropertyName("oracle"); w.WriteStartObject(); w.WriteString("oracle_type","bounded-search"); w.WriteNumber("depth",value.Depth); w.WriteNumber("expansion_budget",value.ExpansionBudget); w.WriteBoolean("use_heuristic",true); w.WriteString("heuristic_identity",BoundedSearchAgent.HeuristicIdentity); WriteHash(w,"code_hash",value.CodeHash); w.WriteEndObject(); }
        private static void WriteDuel(Utf8JsonWriter w, OracleEvidenceScheduledDuel value) { w.WritePropertyName("scheduled_duel"); w.WriteStartObject(); w.WriteNumber("schedule_index",value.ScheduleIndex);w.WriteNumber("map_seed",value.MapSeed);w.WriteNumber("episode_seed",value.EpisodeSeed);w.WriteString("profile",value.Profile);w.WriteNumber("reference_seat",value.ReferenceSeat);w.WriteNumber("learner_seat",value.LearnerSeat);w.WriteEndObject(); }
        private static void WriteArtifact(Utf8JsonWriter w,string name,OracleEvidenceArtifact value){w.WritePropertyName(name);w.WriteStartObject();WriteHash(w,"sha256",value.Sha256);w.WriteNumber("byte_size",value.Bytes.Length);w.WriteEndObject();}
    }

    internal sealed class OracleEvidenceRuntimeIdentity { internal OracleEvidenceRuntimeIdentity(string environment,string scenario,string contract,string encoding,string oracle){Environment=environment;ScenarioSha256=scenario;ContractHash=contract;EncodingHash=encoding;OracleCodeSha256=oracle;} internal string Environment{get;} internal string ScenarioSha256{get;} internal string ContractHash{get;} internal string EncodingHash{get;} internal string OracleCodeSha256{get;} }
    internal sealed class OracleEvidenceOracle { internal OracleEvidenceOracle(int depth,int budget,string code){Depth=depth;ExpansionBudget=budget;CodeHash=code;} internal int Depth{get;} internal int ExpansionBudget{get;} internal string CodeHash{get;} }
    internal sealed class OracleEvidenceScheduledDuel { internal OracleEvidenceScheduledDuel(int index,int map,int episode,string profile,int reference,int learner){ScheduleIndex=index;MapSeed=map;EpisodeSeed=episode;Profile=profile;ReferenceSeat=reference;LearnerSeat=learner;} internal int ScheduleIndex{get;} internal int MapSeed{get;} internal int EpisodeSeed{get;} internal string Profile{get;} internal int ReferenceSeat{get;} internal int LearnerSeat{get;} }
    internal sealed class OracleEvidenceScheduleItem { internal OracleEvidenceScheduleItem(int candidate,int game,OracleEvidenceOracle oracle,OracleEvidenceScheduledDuel duel){CandidateIndex=candidate;GameIndex=game;Oracle=oracle;Duel=duel;} internal int CandidateIndex{get;} internal int GameIndex{get;} internal OracleEvidenceOracle Oracle{get;} internal OracleEvidenceScheduledDuel Duel{get;} }
    internal sealed class OracleEvidenceBeginRequest { internal OracleEvidenceBeginRequest(string nonce,string panel,string commit,string tree,string scenario,string contract,string encoding,OracleEvidenceOracle oracle,OracleEvidenceOracle[] candidates,OracleEvidenceScheduledDuel[] schedule,string scheduleHash,OracleEvidenceScheduleItem[] expanded){Nonce=nonce;PanelSha256=panel;Commit=commit;SourceTree=tree;ScenarioSha256=scenario;ContractHash=contract;EncodingHash=encoding;Oracle=oracle;Candidates=candidates;PreflightSchedule=schedule;PreflightScheduleSha256=scheduleHash;CandidatesBySchedule=expanded;} internal string Nonce{get;} internal string PanelSha256{get;} internal string Commit{get;} internal string SourceTree{get;} internal string ScenarioSha256{get;} internal string ContractHash{get;} internal string EncodingHash{get;} internal OracleEvidenceOracle Oracle{get;} internal IReadOnlyList<OracleEvidenceOracle>Candidates{get;} internal IReadOnlyList<OracleEvidenceScheduledDuel>PreflightSchedule{get;} internal string PreflightScheduleSha256{get;} internal IReadOnlyList<OracleEvidenceScheduleItem>CandidatesBySchedule{get;} internal void Validate(OracleEvidenceRuntimeIdentity runtime){if(runtime.Environment!="tactical-v2"||ScenarioSha256!=runtime.ScenarioSha256||ContractHash!=runtime.ContractHash||EncodingHash!=runtime.EncodingHash||Oracle.CodeHash!=runtime.OracleCodeSha256)throw new InvalidDataException("evidence runtime identity does not match"); if(Candidates.Count==0||PreflightSchedule.Count==0||CandidatesBySchedule.Count!=Candidates.Count*PreflightSchedule.Count||CandidatesBySchedule.Count>OracleEvidenceSession.MaxScheduleItems)throw new InvalidDataException("evidence schedule count is invalid"); if(OracleEvidenceSession.Sha256(CanonicalSchedule(PreflightSchedule))!=PreflightScheduleSha256)throw new InvalidDataException("evidence schedule hash does not match"); for(int c=0;c<Candidates.Count;c++)for(int g=0;g<PreflightSchedule.Count;g++){var item=CandidatesBySchedule[c*PreflightSchedule.Count+g];if(item.CandidateIndex!=c||item.GameIndex!=g||!Same(item.Oracle,Candidates[c])||!Same(item.Duel,PreflightSchedule[g]))throw new InvalidDataException("evidence schedule order is noncanonical");}} private static bool Same(OracleEvidenceOracle a,OracleEvidenceOracle b)=>a.Depth==b.Depth&&a.ExpansionBudget==b.ExpansionBudget&&a.CodeHash==b.CodeHash; private static bool Same(OracleEvidenceScheduledDuel a,OracleEvidenceScheduledDuel b)=>a.ScheduleIndex==b.ScheduleIndex&&a.MapSeed==b.MapSeed&&a.EpisodeSeed==b.EpisodeSeed&&a.Profile==b.Profile&&a.ReferenceSeat==b.ReferenceSeat&&a.LearnerSeat==b.LearnerSeat; private static byte[] CanonicalSchedule(IReadOnlyList<OracleEvidenceScheduledDuel> schedule){using var s=new MemoryStream();using(var w=new Utf8JsonWriter(s)){w.WriteStartArray();foreach(var d in schedule){w.WriteStartObject();w.WriteNumber("episode_seed",d.EpisodeSeed);w.WriteNumber("learner_seat",d.LearnerSeat);w.WriteNumber("map_seed",d.MapSeed);w.WriteString("profile",d.Profile);w.WriteNumber("reference_seat",d.ReferenceSeat);w.WriteNumber("schedule_index",d.ScheduleIndex);w.WriteEndObject();}w.WriteEndArray();}return s.ToArray();} }
    internal sealed class OracleEvidenceArtifact { internal OracleEvidenceArtifact(byte[] bytes){Bytes=(byte[])bytes.Clone();Sha256=OracleEvidenceSession.Sha256(Bytes);} internal byte[] Bytes{get;} internal string Sha256{get;} }
    internal sealed class OracleEvidenceGameContext { internal OracleEvidenceGameContext(string session,string nonce,int candidate,int game,string outcome,int? winner,int transitions,int samples,int expansions){SessionId=session;Nonce=nonce;CandidateIndex=candidate;GameIndex=game;Outcome=outcome;Winner=winner;TransitionCount=transitions;BenchmarkSampleCount=samples;ExpansionTotal=expansions;} internal string SessionId{get;}internal string Nonce{get;}internal int CandidateIndex{get;}internal int GameIndex{get;}internal string Outcome{get;}internal int? Winner{get;}internal int TransitionCount{get;}internal int BenchmarkSampleCount{get;}internal int ExpansionTotal{get;} internal void Validate(string session,string nonce,OracleEvidenceScheduleItem expected){if(SessionId!=session||Nonce!=nonce||CandidateIndex!=expected.CandidateIndex||GameIndex!=expected.GameIndex||TransitionCount<1||BenchmarkSampleCount<0||ExpansionTotal<0||!(Outcome=="win"||Outcome=="loss"||Outcome=="draw"))throw new InvalidDataException("evidence game close context is invalid");} }
    internal sealed class OracleEvidenceReceipt { internal OracleEvidenceReceipt(string nonce,string session,int sequence,string previous,OracleEvidenceScheduleItem item,OracleEvidenceGameContext context,OracleEvidenceArtifact trace,OracleEvidenceArtifact replay,OracleEvidenceArtifact benchmark,string hash,byte[] body){Nonce=nonce;SessionId=session;Sequence=sequence;PreviousReceiptSha256=previous;Item=item;Context=context;Trace=trace;Replay=replay;Benchmark=benchmark;ReceiptSha256=hash;Utf8=body;} internal string Nonce{get;}internal string SessionId{get;}internal int Sequence{get;}internal string PreviousReceiptSha256{get;}internal OracleEvidenceScheduleItem Item{get;}internal OracleEvidenceGameContext Context{get;}internal OracleEvidenceArtifact Trace{get;}internal OracleEvidenceArtifact Replay{get;}internal OracleEvidenceArtifact Benchmark{get;}internal string ReceiptSha256{get;}internal byte[] Utf8{get;} }
    internal sealed class OracleEvidenceBeginResponse { internal OracleEvidenceBeginResponse(OracleEvidenceSession session,string nonce,string id,string schedule,string chain,string hash){Session=session;Nonce=nonce;SessionId=id;ScheduleSha256=schedule;InitialChainSha256=chain;BeginContentSha256=hash;} internal OracleEvidenceSession Session{get;}public int schema_version=>1;public string nonce=>Nonce;public string session_id=>SessionId;public string schedule_sha256=>ScheduleSha256;public string environment=>"tactical-v2";public int sequence=>0;public string initial_chain_sha256=>InitialChainSha256;public string begin_content_sha256=>BeginContentSha256; internal string Nonce{get;}internal string SessionId{get;}internal string ScheduleSha256{get;}internal string InitialChainSha256{get;}internal string BeginContentSha256{get;} }
    internal sealed class OracleEvidenceGameResponse { internal OracleEvidenceGameResponse(OracleEvidenceReceipt receipt,OracleEvidenceArtifact trace,OracleEvidenceArtifact replay,OracleEvidenceArtifact benchmark){Receipt=receipt;Trace=trace;Replay=replay;Benchmark=benchmark;} internal OracleEvidenceReceipt Receipt{get;} internal OracleEvidenceArtifact Trace{get;} internal OracleEvidenceArtifact Replay{get;} internal OracleEvidenceArtifact Benchmark{get;} }
    internal sealed class OracleEvidenceEndResponse { internal OracleEvidenceEndResponse(string id,string nonce,int count,string final,string hash){SessionId=id;Nonce=nonce;ReceiptCount=count;FinalReceiptSha256=final;EndContentSha256=hash;} public int schema_version=>1;public string session_id=>SessionId;public string nonce=>Nonce;public int receipt_count=>ReceiptCount;public string final_receipt_sha256=>FinalReceiptSha256;public string end_content_sha256=>EndContentSha256; internal string SessionId{get;}internal string Nonce{get;}internal int ReceiptCount{get;}internal string FinalReceiptSha256{get;}internal string EndContentSha256{get;} }
}