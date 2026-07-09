using UnityEditor;
using UnityEngine;

namespace HexWars.Presentation.EditorTools
{
    /// <summary>
    /// One-shot editor script: sets AudioImporter settings for the 17 user-supplied clips under
    /// Assets/HexWars/Resources/Audio/. Run once from the menu after new clips land (or whenever the
    /// clip list changes) — Unity's default import (DecompressOnLoad, quality 1, no background load)
    /// is fine for nothing here, so every clip needs an explicit pass.
    ///
    /// Loop beds (TitleMusic/AmbientBed/DesignerHum) use CompressedInMemory: Streaming is NOT supported
    /// on WebGL, and these clips loop for the life of a scene so keeping them compressed in memory
    /// (rather than fully decompressed) matters for the 23 MB title track especially. One-shot SFX
    /// (DeployDoor + the 13 weapon/rack clips) use DecompressOnLoad: short, played on demand, cheapest
    /// as PCM once loaded so PlayOneShot has no per-call decode cost.
    /// </summary>
    public static class AudioImportSettings
    {
        const string Dir = "Assets/HexWars/Resources/Audio";

        [MenuItem("HexWars/Audio/Apply Import Settings")]
        public static void Apply()
        {
            // three looping beds — CompressedInMemory (WebGL can't stream), background-loaded so they
            // don't hitch the frame that starts them
            ApplyOne("TitleMusic", AudioClipLoadType.CompressedInMemory, 0.30f, loadInBackground: true);
            ApplyOne("AmbientBed", AudioClipLoadType.CompressedInMemory, 0.40f, loadInBackground: true);
            ApplyOne("DesignerHum", AudioClipLoadType.CompressedInMemory, 0.40f, loadInBackground: true);

            // deploy door + tiered weapon shots + the design-racked clip — short one-shots, decompressed
            // fully on load so PlayOneShot never pays a decode cost mid-battle
            ApplyOne("DeployDoor", AudioClipLoadType.DecompressOnLoad, 0.45f, loadInBackground: false);
            foreach (var family in new[] { "AttackLight", "AttackMid", "AttackHeavy" })
                for (int i = 0; i < 4; i++)
                    ApplyOne($"{family}_{i}", AudioClipLoadType.DecompressOnLoad, 0.45f, loadInBackground: false);
            ApplyOne("CreateRacked", AudioClipLoadType.DecompressOnLoad, 0.45f, loadInBackground: false);

            AssetDatabase.SaveAssets();
            Debug.Log("[AudioImportSettings] Applied import settings to 17 clips under " + Dir);
        }

        static void ApplyOne(string clipName, AudioClipLoadType loadType, float quality, bool loadInBackground)
        {
            string path = $"{Dir}/{clipName}.wav";
            var importer = AssetImporter.GetAtPath(path) as AudioImporter;
            if (importer == null)
            {
                Debug.LogWarning($"[AudioImportSettings] No AudioImporter found at {path} — skipped.");
                return;
            }

            var settings = importer.defaultSampleSettings;
            settings.loadType = loadType;
            settings.compressionFormat = AudioCompressionFormat.Vorbis;
            settings.quality = quality;
            importer.defaultSampleSettings = settings;
            importer.forceToMono = false;
            importer.loadInBackground = loadInBackground;

            EditorUtility.SetDirty(importer);
            importer.SaveAndReimport();
        }
    }
}
