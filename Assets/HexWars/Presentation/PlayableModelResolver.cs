using System;
using System.IO;
using System.Security.Cryptography;

namespace HexWars.Presentation
{
    public sealed class PlayableModelLaunch
    {
        public PlayableModelLaunch(string pythonExecutable, string serverScript, string workingDirectory, string runDirectory)
        {
            PythonExecutable = pythonExecutable; ServerScript = serverScript;
            WorkingDirectory = workingDirectory; RunDirectory = runDirectory;
        }
        public string PythonExecutable { get; }
        public string ServerScript { get; }
        public string WorkingDirectory { get; }
        public string RunDirectory { get; }
        public string ModelName => Path.GetFileName(RunDirectory.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));
        public string ControllerSpec => "run:" + RunDirectory;
    }

    /// <summary>Resolves the catalog's pinned deployment package, never a research run or newest directory.</summary>
    public static class PlayableModelResolver
    {
        public static PlayableModelLaunch Resolve(string projectRoot, AiModelSelection selection) => Resolve(
            projectRoot, selection, Environment.GetEnvironmentVariable("HEXWARS_AI_RUNTIME_ROOT"),
            Environment.GetEnvironmentVariable("HEXWARS_AI_PYTHON"));

        internal static PlayableModelLaunch Resolve(string projectRoot, AiModelSelection selection,
            string runtimeOverride, string pythonOverride)
        {
            if (string.IsNullOrWhiteSpace(projectRoot)) throw new ArgumentException("Project root is required.");
            if (selection == null || !selection.Model.IsTrained) throw new ArgumentException("A trained model selection is required.");
            selection.Model.Validate();
            string root = Path.GetFullPath(string.IsNullOrWhiteSpace(runtimeOverride) ? projectRoot : runtimeOverride);
            string python = Path.Combine(root, "python");
            string executable = string.IsNullOrWhiteSpace(pythonOverride)
                ? Path.Combine(python, "winenv", "Scripts", "python.exe") : Path.GetFullPath(pythonOverride);
            string script = Path.Combine(python, "policy_server.py");
            string package = ContainedPath(root, selection.ModelsRoot + "/" + selection.Model.package);
            string checkpoint = ContainedPath(package, "checkpoints/best.pt");
            RequireFile(executable, "AI Python runtime (set HEXWARS_AI_PYTHON)");
            RequireFile(script, "AI policy server (set HEXWARS_AI_RUNTIME_ROOT)");
            RequireFile(Path.Combine(package, "run.json"), "AI model package manifest");
            RequireFile(Path.Combine(package, "policy-identity.json"), "AI model identity");
            RequireFile(checkpoint, "AI checkpoint");
            for (var directory = new DirectoryInfo(Path.GetDirectoryName(checkpoint)); directory != null; directory = directory.Parent)
            {
                if ((directory.Attributes & FileAttributes.ReparsePoint) != 0)
                    throw new InvalidDataException("AI package paths cannot contain links.");
                if (string.Equals(directory.FullName, root, StringComparison.OrdinalIgnoreCase)) break;
            }
            string actual;
            using (var stream = File.OpenRead(checkpoint))
            using (var hash = SHA256.Create())
                actual = BitConverter.ToString(hash.ComputeHash(stream)).Replace("-", "").ToLowerInvariant();
            if (!string.Equals(actual, selection.Model.checkpoint_sha256, StringComparison.Ordinal))
                throw new InvalidDataException("AI checkpoint does not match the configured model: " + selection.Model.id);
            return new PlayableModelLaunch(executable, script, python, package);
        }

        internal static string ContainedPath(string root, string relative)
        {
            HexWars.Engine.AI.AiModelCatalog.RequireRelativePath(relative, "AI package path");
            string parent = Path.GetFullPath(root).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            string path = Path.GetFullPath(Path.Combine(parent, relative.Replace('/', Path.DirectorySeparatorChar)));
            if (!path.StartsWith(parent + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException("AI package escapes runtime root.");
            return path;
        }

        static void RequireFile(string path, string label)
        {
            if (!File.Exists(path)) throw new FileNotFoundException(label + " was not found: " + path, path);
            if ((File.GetAttributes(path) & FileAttributes.ReparsePoint) != 0)
                throw new InvalidDataException(label + " cannot be a link.");
        }
    }
}
