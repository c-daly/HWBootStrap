using System;
using System.IO;
using UnityEngine;

namespace HexWars.Presentation
{
    /// <summary>The fixed policy package and local Python runtime used by a playable model match.</summary>
    public sealed class PlayableModelLaunch
    {
        public PlayableModelLaunch(
            string pythonExecutable,
            string serverScript,
            string workingDirectory,
            string runDirectory)
        {
            PythonExecutable = pythonExecutable;
            ServerScript = serverScript;
            WorkingDirectory = workingDirectory;
            RunDirectory = runDirectory;
        }

        public string PythonExecutable { get; }
        public string ServerScript { get; }
        public string WorkingDirectory { get; }
        public string RunDirectory { get; }
        public string ModelName => Path.GetFileName(
            RunDirectory.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));
        public string ControllerSpec => "run:" + RunDirectory;
    }

    /// <summary>
    /// Resolves the one intentionally selected playable lineage.  It never scans for a newest folder:
    /// an in-progress structured-DAGGER inventory resolves exactly as ML Lab does, to its published
    /// model when present or its declared source policy until publication.  The pinned completed model
    /// is used only when the selected lineage is absent; once a lineage exists, invalid metadata fails
    /// closed so it can never be hidden behind a plausible-looking fallback.
    /// </summary>
    public static class PlayableModelResolver
    {
        public const string SelectedLineage = "besttestrunyet";
        public const string PinnedCompletedModel = "fdafasagg-model";

        public static PlayableModelLaunch Resolve(string projectRoot)
        {
            if (string.IsNullOrWhiteSpace(projectRoot))
                throw new ArgumentException("project root is required", nameof(projectRoot));

            string root = Path.GetFullPath(projectRoot);
            string sharedRoot = ResolveSharedRepositoryRoot(root);
            string pythonDirectory = Path.Combine(root, "python");
            string sharedPythonDirectory = Path.Combine(sharedRoot, "python");
            string pythonExecutable = ExistingFileOrLocal(
                Path.Combine(pythonDirectory, "winenv", "Scripts", "python.exe"),
                Path.Combine(sharedPythonDirectory, "winenv", "Scripts", "python.exe"));
            string serverScript = Path.Combine(pythonDirectory, "policy_server.py");
            string runsDirectory = Path.Combine(pythonDirectory, "runs");
            string sharedRunsDirectory = Path.Combine(sharedPythonDirectory, "runs");
            if (!Directory.Exists(Path.Combine(runsDirectory, SelectedLineage)) &&
                !Directory.Exists(Path.Combine(runsDirectory, PinnedCompletedModel)) &&
                !PathEquals(runsDirectory, sharedRunsDirectory))
                runsDirectory = sharedRunsDirectory;

            RequireFile(pythonExecutable, "Windows Python environment");
            RequireFile(serverScript, "policy server");

            string selected = Path.Combine(runsDirectory, SelectedLineage);
            string fallback = Path.Combine(runsDirectory, PinnedCompletedModel);
            string resolved = fallback;
            if (Directory.Exists(selected))
            {
                RequireSafeRunDirectory(runsDirectory, selected);
                resolved = ResolveStructuredInventory(root, runsDirectory, selected);
            }
            RequireSafeRunDirectory(runsDirectory, resolved);
            RequireCompletedModel(resolved);

            return new PlayableModelLaunch(
                pythonExecutable, serverScript, pythonDirectory, resolved);
        }

        static string ExistingFileOrLocal(string local, string shared) =>
            File.Exists(local) || PathEquals(local, shared) ? local : shared;

        static string ResolveSharedRepositoryRoot(string projectRoot)
        {
            try
            {
                string marker = Path.Combine(projectRoot, ".git");
                if (!File.Exists(marker)) return projectRoot;
                string text = File.ReadAllText(marker).Trim();
                const string prefix = "gitdir:";
                if (!text.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
                    return projectRoot;

                string gitDirectory = text.Substring(prefix.Length).Trim();
                if (!Path.IsPathRooted(gitDirectory))
                    gitDirectory = Path.GetFullPath(Path.Combine(projectRoot, gitDirectory));
                string commonMarker = Path.Combine(gitDirectory, "commondir");
                if (!File.Exists(commonMarker)) return projectRoot;

                string commonDirectory = File.ReadAllText(commonMarker).Trim();
                if (!Path.IsPathRooted(commonDirectory))
                    commonDirectory = Path.GetFullPath(Path.Combine(
                        gitDirectory, commonDirectory));
                DirectoryInfo repository = Directory.GetParent(commonDirectory);
                return repository?.FullName ?? projectRoot;
            }
            catch (Exception error) when (
                error is IOException || error is UnauthorizedAccessException ||
                error is ArgumentException || error is NotSupportedException)
            {
                return projectRoot;
            }
        }

        static string ResolveStructuredInventory(
            string projectRoot, string runsDirectory, string directory)
        {
            string manifestPath = Path.Combine(directory, "run.json");
            RequireRegularFile(manifestPath, "selected model-lineage manifest");

            InventoryManifest manifest;
            try
            {
                manifest = JsonUtility.FromJson<InventoryManifest>(
                    File.ReadAllText(manifestPath));
            }
            catch (Exception error) when (
                error is IOException || error is UnauthorizedAccessException ||
                error is ArgumentException || error is NotSupportedException)
            {
                throw new InvalidDataException(
                    "selected model-lineage manifest could not be read: " +
                    manifestPath + ": " + error.Message, error);
            }

            if (manifest == null || manifest.schema_version != 1 ||
                !string.Equals(manifest.config?.algorithm, "structured_dagger",
                    StringComparison.Ordinal) ||
                !string.Equals(manifest.contract?.environment, "tactical-v3",
                    StringComparison.Ordinal))
                throw new InvalidDataException(
                    "selected model-lineage manifest is not structured-DAGGER tactical-v3: " +
                    manifestPath);

            bool hasPublication = !string.IsNullOrWhiteSpace(manifest.published_run);
            string declared = hasPublication
                ? manifest.published_run
                : manifest.source_policy?.run;
            if (string.IsNullOrWhiteSpace(declared))
                throw new InvalidDataException(
                    "selected model lineage declares neither a published run nor a source policy: " +
                    manifestPath);

            string resolved = ResolveDeclaredRun(
                projectRoot, runsDirectory, declared);
            if (resolved == null)
                throw new DirectoryNotFoundException(
                    (hasPublication ? "published" : "source") +
                    " model declared by the selected lineage was not found locally: " + declared);
            return resolved;
        }

        static string ResolveDeclaredRun(
            string projectRoot, string runsDirectory, string declared)
        {
            if (string.IsNullOrWhiteSpace(declared)) return null;
            string portable = declared.Replace('\\', '/').TrimEnd('/');
            string[] components = portable.Split('/');
            foreach (string component in components)
                if (component == "." || component == "..")
                    throw new InvalidDataException(
                        "declared model path contains a traversal component: " + declared);

            string leaf = components.Length == 0
                ? string.Empty
                : components[components.Length - 1];
            if (string.IsNullOrWhiteSpace(leaf) ||
                leaf.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0)
                throw new InvalidDataException(
                    "declared model path has an invalid run name: " + declared);

            try
            {
                string direct = Path.IsPathRooted(declared)
                    ? Path.GetFullPath(declared)
                    : Path.GetFullPath(Path.Combine(
                        projectRoot,
                        portable.Replace('/', Path.DirectorySeparatorChar)));
                if (IsStrictDescendant(runsDirectory, direct) && Directory.Exists(direct))
                {
                    RequireSafeRunDirectory(runsDirectory, direct);
                    return direct;
                }
            }
            catch (Exception error) when (
                error is IOException || error is UnauthorizedAccessException ||
                error is ArgumentException || error is NotSupportedException)
            {
                // A run recorded on another checkout can still be resolved by its declared leaf below.
            }

            string local = Path.GetFullPath(Path.Combine(runsDirectory, leaf));
            if (!IsStrictDescendant(runsDirectory, local) || !Directory.Exists(local))
                return null;
            RequireSafeRunDirectory(runsDirectory, local);
            return local;
        }

        static void RequireCompletedModel(string runDirectory)
        {
            if (string.IsNullOrWhiteSpace(runDirectory) || !Directory.Exists(runDirectory))
                throw new DirectoryNotFoundException(
                    "playable trained model directory was not found: " + runDirectory);
            string manifestPath = Path.Combine(runDirectory, "run.json");
            string identityPath = Path.Combine(runDirectory, "policy-identity.json");
            RequireRegularFile(manifestPath, "trained model manifest");
            RequireRegularFile(identityPath, "trained model identity");

            ModelManifest manifest;
            try
            {
                manifest = JsonUtility.FromJson<ModelManifest>(
                    File.ReadAllText(manifestPath));
            }
            catch (Exception error) when (
                error is IOException || error is UnauthorizedAccessException ||
                error is ArgumentException || error is NotSupportedException)
            {
                throw new InvalidDataException(
                    "trained model manifest could not be read: " + error.Message, error);
            }

            if (manifest == null || manifest.schema_version != 2 ||
                !string.Equals(manifest.state, "completed", StringComparison.Ordinal) ||
                !string.Equals(manifest.config?.algorithm, "structured_imitation",
                    StringComparison.Ordinal) ||
                !string.Equals(manifest.contract?.environment, "tactical-v3",
                    StringComparison.Ordinal) ||
                !string.Equals(manifest.contract?.version, "tactical-v3",
                    StringComparison.Ordinal) ||
                !string.Equals(manifest.contract?.environment_kind, "duel",
                    StringComparison.Ordinal) ||
                !IsLowerSha256(manifest.contract?.contract_hash) ||
                !IsLowerSha256(manifest.contract?.encoding_hash) ||
                !IsLowerSha256(manifest.contract?.capacity_hash) ||
                !string.Equals(manifest.policy_identity, "policy-identity.json",
                    StringComparison.Ordinal) ||
                !string.Equals(
                    manifest.latest_checkpoint?.Replace('\\', '/'),
                    "checkpoints/best.pt", StringComparison.Ordinal))
                throw new InvalidDataException(
                    "playable trained model is not a completed structured-imitation " +
                    "tactical-v3 package: " + manifestPath);

            IdentityManifest identity;
            try
            {
                identity = JsonUtility.FromJson<IdentityManifest>(
                    File.ReadAllText(identityPath));
            }
            catch (Exception error) when (
                error is IOException || error is UnauthorizedAccessException ||
                error is ArgumentException || error is NotSupportedException)
            {
                throw new InvalidDataException(
                    "trained model identity could not be read: " + identityPath +
                    ": " + error.Message, error);
            }
            if (identity == null ||
                !string.Equals(identity.contract_version, manifest.contract.version,
                    StringComparison.Ordinal) ||
                !string.Equals(identity.environment_kind, manifest.contract.environment_kind,
                    StringComparison.Ordinal) ||
                !string.Equals(identity.contract_hash, manifest.contract.contract_hash,
                    StringComparison.Ordinal) ||
                !string.Equals(identity.encoding_hash, manifest.contract.encoding_hash,
                    StringComparison.Ordinal) ||
                !string.Equals(identity.capacity_hash, manifest.contract.capacity_hash,
                    StringComparison.Ordinal))
                throw new InvalidDataException(
                    "trained model identity does not match its run manifest: " + identityPath);

            string checkpoints = Path.GetFullPath(Path.Combine(runDirectory, "checkpoints"));
            string checkpoint = Path.GetFullPath(Path.Combine(checkpoints, "best.pt"));
            if (!IsStrictDescendant(runDirectory, checkpoint) ||
                !IsStrictDescendant(runDirectory, checkpoints))
                throw new InvalidDataException(
                    "trained model checkpoint escapes its run package: " + checkpoint);
            RequireNoReparsePoints(runDirectory, checkpoints);
            RequireRegularFile(checkpoint, "trained model checkpoint");
        }

        static void RequireFile(string path, string label)
        {
            if (!File.Exists(path))
                throw new FileNotFoundException(label + " was not found: " + path, path);
        }

        static void RequireRegularFile(string path, string label)
        {
            RequireFile(path, label);
            if ((File.GetAttributes(path) & FileAttributes.ReparsePoint) != 0)
                throw new InvalidDataException(
                    label + " must not be a reparse point: " + path);
        }

        static void RequireSafeRunDirectory(string runsDirectory, string runDirectory)
        {
            string root = Path.GetFullPath(runsDirectory);
            string candidate = Path.GetFullPath(runDirectory);
            if (!IsStrictDescendant(root, candidate))
                throw new InvalidDataException(
                    "playable model must be inside the local runs directory: " + candidate);
            RequireNoReparsePoints(root, candidate);
        }

        static void RequireNoReparsePoints(string rootDirectory, string descendant)
        {
            string root = Path.GetFullPath(rootDirectory);
            DirectoryInfo current = new DirectoryInfo(Path.GetFullPath(descendant));
            while (true)
            {
                if (!current.Exists)
                    throw new DirectoryNotFoundException(
                        "model package directory was not found: " + current.FullName);
                if ((current.Attributes & FileAttributes.ReparsePoint) != 0)
                    throw new InvalidDataException(
                        "model package path must not contain a reparse point: " +
                        current.FullName);
                if (PathEquals(current.FullName, root)) return;
                current = current.Parent ?? throw new InvalidDataException(
                    "model package path escaped its expected root");
            }
        }

        static bool IsStrictDescendant(string rootDirectory, string candidate)
        {
            string root = Path.GetFullPath(rootDirectory)
                .TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar) +
                Path.DirectorySeparatorChar;
            string path = Path.GetFullPath(candidate);
            return path.StartsWith(root, PathComparison());
        }

        static bool PathEquals(string left, string right) =>
            string.Equals(
                Path.GetFullPath(left).TrimEnd(
                    Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar),
                Path.GetFullPath(right).TrimEnd(
                    Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar),
                PathComparison());

        static StringComparison PathComparison() =>
            Path.DirectorySeparatorChar == '\\'
                ? StringComparison.OrdinalIgnoreCase
                : StringComparison.Ordinal;

        static bool IsLowerSha256(string value)
        {
            if (value == null || value.Length != 64) return false;
            foreach (char ch in value)
                if (!((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f')))
                    return false;
            return true;
        }

        [Serializable] sealed class InventoryManifest
        {
            public int schema_version;
            public InventoryConfig config;
            public Contract contract;
            public string published_run;
            public SourcePolicy source_policy;
        }

        [Serializable] sealed class InventoryConfig { public string algorithm; }
        [Serializable] sealed class SourcePolicy { public string run; }

        [Serializable] sealed class ModelManifest
        {
            public int schema_version;
            public string state;
            public string latest_checkpoint;
            public string policy_identity;
            public ModelConfig config;
            public Contract contract;
        }

        [Serializable] sealed class ModelConfig { public string algorithm; }

        [Serializable] sealed class Contract
        {
            public string environment;
            public string version;
            public string environment_kind;
            public string contract_hash;
            public string encoding_hash;
            public string capacity_hash;
        }

        [Serializable] sealed class IdentityManifest
        {
            public string contract_version;
            public string environment_kind;
            public string contract_hash;
            public string encoding_hash;
            public string capacity_hash;
        }
    }
}
