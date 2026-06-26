using System.Diagnostics;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace HexWars.Presentation.EditorTools
{
    /// <summary>
    /// Start a FRESH training run from the editor: pick the algorithm + opponent, name the run, hit Start.
    /// Launches the matching trainer in the Windows GPU venv (MaskablePPO with the spatial CNN, or the
    /// experimental masked DQN), which writes checkpoints into python/runs/&lt;run&gt;/checkpoints/. Then watch
    /// it learn live via HexWars ▸ Watch Live Training (point it at that checkpoints folder). The trainer
    /// runs as a detached background process; the live viewer reads its checkpoints off disk (CPU inference,
    /// so it never contends with training for the GPU).
    /// </summary>
    public sealed class TrainingLauncher : EditorWindow
    {
        enum Algo { PPO_MaskablePPO, DQN_Masked }
        enum Opponent { Greedy, Random }

        Algo _algo = Algo.PPO_MaskablePPO;
        Opponent _opponent = Opponent.Greedy;
        string _runName = "run1";
        int _timesteps = 300000;
        int _checkpointEvery = 10000;
        int _seed = 1;

        [MenuItem("HexWars/Start Training...")]
        static void Open() => GetWindow<TrainingLauncher>(true, "HexWars — Start Training");

        void OnGUI()
        {
            EditorGUILayout.LabelField("Fresh training run (GPU · Windows venv)", EditorStyles.boldLabel);
            EditorGUILayout.Space();

            _algo = (Algo)EditorGUILayout.EnumPopup("Algorithm", _algo);
            _opponent = (Opponent)EditorGUILayout.EnumPopup("Opponent", _opponent);
            _runName = EditorGUILayout.TextField("Run name", _runName);
            _timesteps = EditorGUILayout.IntField("Timesteps", _timesteps);
            _checkpointEvery = EditorGUILayout.IntField("Checkpoint every", _checkpointEvery);
            _seed = EditorGUILayout.IntField("Seed", _seed);

            EditorGUILayout.Space();
            EditorGUILayout.HelpBox(
                "PPO = MaskablePPO with the spatial CNN (the strong path). DQN = experimental masked " +
                "value-based agent.\nWrites checkpoints to python/runs/<run>/checkpoints — then use " +
                "HexWars ▸ Watch Live Training on that folder to watch it improve.",
                MessageType.Info);

            EditorGUILayout.Space();
            using (new EditorGUI.DisabledScope(string.IsNullOrWhiteSpace(_runName) || _timesteps <= 0))
                if (GUILayout.Button("Start Training", GUILayout.Height(32)))
                    StartTraining();
        }

        void StartTraining()
        {
            string pyDir = Path.Combine(Directory.GetParent(Application.dataPath).FullName, "python");
            string pyExe = Path.Combine(pyDir, "winenv", "Scripts", "python.exe");
            if (!File.Exists(pyExe))
            {
                EditorUtility.DisplayDialog("HexWars", "Windows venv Python not found at:\n" + pyExe, "OK");
                return;
            }

            string runDir = Path.Combine(pyDir, "runs", _runName);
            if (Directory.Exists(runDir) && !EditorUtility.DisplayDialog(
                    "HexWars", $"Run '{_runName}' already exists and will be overwritten:\n{runDir}", "Overwrite", "Cancel"))
                return;

            string script = _algo == Algo.PPO_MaskablePPO ? "train_maskable_ppo.py" : "train_dqn.py";
            string opp = _opponent == Opponent.Greedy ? "greedy" : "random";
            string logPath = Path.Combine(pyDir, _runName + ".log");

            // Run through cmd so stdout/stderr redirect to a log file without us having to drain pipes.
            // cmd /c "..." keeps everything between the outer quotes verbatim, so the inner quoted paths work.
            string inner = $"\"{pyExe}\" {script} --opponent {opp} --out {_runName} " +
                           $"--timesteps {_timesteps} --checkpoint-freq {_checkpointEvery} --seed {_seed} > \"{logPath}\" 2>&1";
            var psi = new ProcessStartInfo
            {
                FileName = "cmd.exe",
                Arguments = "/c \"" + inner + "\"",
                WorkingDirectory = pyDir,
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            try { Process.Start(psi); }
            catch (System.Exception e)
            {
                EditorUtility.DisplayDialog("HexWars", "Failed to launch training:\n" + e.Message, "OK");
                return;
            }

            UnityEngine.Debug.Log($"HexWars: started {script} vs {opp} -> runs/{_runName} (log: {_runName}.log). " +
                                  $"Watch via HexWars > Watch Live Training -> runs/{_runName}/checkpoints");
            EditorUtility.DisplayDialog("HexWars",
                $"Training started: {_algo} vs {opp}\n\nCheckpoints: python/runs/{_runName}/checkpoints\n\n" +
                "Give it ~a minute for the first checkpoint, then:\nHexWars > Watch Live Training… and pick that folder.",
                "OK");
            Close();
        }
    }
}
