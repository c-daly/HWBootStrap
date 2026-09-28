using System;
using System.Collections.Generic;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;

namespace HexWars.Engine.AI
{
    /// <summary>The portable model selection contract shared by Unity and the inference host.</summary>
    [Serializable]
    public sealed class AiModelCatalog
    {
        public int schema_version = 1;
        public string default_difficulty_id = string.Empty;
        public string inference_path = "/api/ai/decision";
        public string models_root = "models/ai";
        public string catalog_revision = string.Empty;
        public AiModelDefinition[] models = Array.Empty<AiModelDefinition>();
        public AiDifficultyDefinition[] difficulties = Array.Empty<AiDifficultyDefinition>();

        public void Validate()
        {
            if (schema_version != 1) throw new ArgumentException("Unsupported AI model catalog version.");
            if (models == null || models.Length == 0)
                throw new ArgumentException("The AI model catalog is empty.");
            RequireRelativePath(models_root, nameof(models_root));
            if (string.IsNullOrWhiteSpace(inference_path) || !inference_path.StartsWith("/", StringComparison.Ordinal)
                || inference_path.StartsWith("//", StringComparison.Ordinal) || inference_path.Contains("\\")
                || inference_path.Contains("?") || inference_path.Contains("#"))
                throw new ArgumentException("AI inference_path must be a same-origin absolute path.");
            var ids = new HashSet<string>(StringComparer.Ordinal);
            foreach (AiModelDefinition model in models)
            {
                if (model == null) throw new ArgumentException("The AI model catalog contains an empty entry.");
                model.Validate();
                if (!ids.Add(model.id)) throw new ArgumentException("Duplicate AI model ID: " + model.id);
            }
            if (difficulties == null || difficulties.Length == 0)
                throw new ArgumentException("The AI catalog has no difficulties.");
            var difficultyIds = new HashSet<string>(StringComparer.Ordinal);
            foreach (AiDifficultyDefinition difficulty in difficulties)
            {
                if (difficulty == null) throw new ArgumentException("The AI catalog contains an empty difficulty.");
                difficulty.Validate();
                if (!difficultyIds.Add(difficulty.id))
                    throw new ArgumentException("Duplicate AI difficulty ID: " + difficulty.id);
                if (!ids.Contains(difficulty.model_id))
                    throw new ArgumentException("AI difficulty references an unavailable model: " + difficulty.model_id);
            }
            if (!difficultyIds.Contains(default_difficulty_id))
                throw new ArgumentException("The default AI difficulty is not in the catalog: " + default_difficulty_id);
            if (!string.IsNullOrEmpty(catalog_revision)) RequireSha256(catalog_revision, nameof(catalog_revision));
        }

        public AiModelDefinition Resolve(string? modelId = null)
        {
            Validate();
            string id = string.IsNullOrEmpty(modelId) ? ResolveDifficulty().model_id : modelId!;
            foreach (AiModelDefinition model in models)
                if (string.Equals(model.id, id, StringComparison.Ordinal)) return model;
            throw new ArgumentException("The selected AI model is not available: " + id);
        }

        public AiDifficultyDefinition ResolveDifficulty(string? difficultyId = null)
        {
            Validate();
            string id = string.IsNullOrEmpty(difficultyId) ? default_difficulty_id : difficultyId!;
            foreach (AiDifficultyDefinition difficulty in difficulties)
                if (string.Equals(difficulty.id, id, StringComparison.Ordinal)) return difficulty;
            throw new ArgumentException("The selected AI difficulty is not available: " + id);
        }

        public static string RevisionOf(string json)
        {
            if (json == null) throw new ArgumentNullException(nameof(json));
            using (SHA256 sha = SHA256.Create())
            {
                byte[] digest = sha.ComputeHash(Encoding.UTF8.GetBytes(json));
                var result = new StringBuilder(64);
                foreach (byte value in digest) result.Append(value.ToString("x2"));
                return result.ToString();
            }
        }

        public static void RequireSha256(string value, string field)
        {
            if (value == null || !Regex.IsMatch(value, "\\A[0-9a-f]{64}\\z", RegexOptions.CultureInvariant))
                throw new ArgumentException(field + " must be a lowercase SHA-256 digest.");
        }

        public static void RequireRelativePath(string value, string field)
        {
            if (string.IsNullOrWhiteSpace(value) || value.StartsWith("/", StringComparison.Ordinal)
                || value.Contains("\\") || value.Contains(":"))
                throw new ArgumentException(field + " must be a portable relative path.");
            foreach (string part in value.Split('/'))
                if (part.Length == 0 || part == "." || part == ".."
                    || !Regex.IsMatch(part, "\\A[A-Za-z0-9_.-]+\\z", RegexOptions.CultureInvariant))
                    throw new ArgumentException(field + " contains an invalid path component.");
        }
    }

    [Serializable]
    public sealed class AiDifficultyDefinition
    {
        public string id = string.Empty;
        public string label = string.Empty;
        public string model_id = string.Empty;

        public void Validate()
        {
            if (id == null || !Regex.IsMatch(id, "\\A[a-z0-9][a-z0-9_-]{0,63}\\z", RegexOptions.CultureInvariant))
                throw new ArgumentException("AI difficulty IDs must use lowercase letters, digits, hyphens or underscores.");
            if (string.IsNullOrWhiteSpace(label) || label.Length > 100)
                throw new ArgumentException("AI difficulty label must contain 1 to 100 characters.");
            if (string.IsNullOrEmpty(model_id)) throw new ArgumentException("AI difficulty must select a model.");
        }
    }

    [Serializable]
    public sealed class AiModelDefinition
    {
        public string id = string.Empty;
        public string label = string.Empty;
        public string kind = "trained";
        public string package = string.Empty;
        public string checkpoint_sha256 = string.Empty;
        public string contract_version = string.Empty;
        public string encoding_hash = string.Empty;
        public string capacity_hash = string.Empty;

        public bool IsTrained => string.Equals(kind, "trained", StringComparison.Ordinal);

        public void Validate()
        {
            if (id == null || !Regex.IsMatch(id, "\\A[a-z0-9][a-z0-9_-]{0,63}\\z", RegexOptions.CultureInvariant))
                throw new ArgumentException("AI model IDs must use lowercase letters, digits, hyphens or underscores.");
            if (string.IsNullOrWhiteSpace(label) || label.Length > 100)
                throw new ArgumentException("AI model label must contain 1 to 100 characters.");
            if (kind != "trained" && kind != "greedy" && kind != "random")
                throw new ArgumentException("Unknown AI model kind: " + kind);
            if (!IsTrained)
            {
                if (!string.IsNullOrEmpty(package) || !string.IsNullOrEmpty(checkpoint_sha256)
                    || !string.IsNullOrEmpty(contract_version) || !string.IsNullOrEmpty(encoding_hash)
                    || !string.IsNullOrEmpty(capacity_hash))
                    throw new ArgumentException("Scripted AI entries cannot claim trained-model metadata.");
                return;
            }
            AiModelCatalog.RequireRelativePath(package, nameof(package));
            AiModelCatalog.RequireSha256(checkpoint_sha256, nameof(checkpoint_sha256));
            AiModelCatalog.RequireSha256(encoding_hash, nameof(encoding_hash));
            AiModelCatalog.RequireSha256(capacity_hash, nameof(capacity_hash));
            if (contract_version != "tactical-v3")
                throw new ArgumentException("Playable trained models require the tactical-v3 contract.");
        }
    }
}
