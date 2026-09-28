# Playable policy packages

`focused-epoch2/` is a portable inference export of the saved focused checkpoint.
Its `run.json` marks **export completion**, while the copied source manifest still
records stopped training and no published training run. Exporting this package
does not change the research retention-gate results.
The checkpoint stores zero-based `best_epoch=1` (the second training epoch), so
the policy-server `step` metadata is 1; the source run's display epoch is 2.

Ship the complete package inventory: `run.json`, `policy-identity.json`,
`checkpoints/best.pt`, `provenance/source-run.json`, and the two evidence reports.
The package is 593,906 bytes; the checkpoint is 575,945 bytes. The shared AI catalog
pins the checkpoint digest. `run.json` additionally seals all package files, and
the loader replays the checkpoint's embedded CPU fixture before accepting it.

The runtime is `python/policy_server.py` plus the repository's Python modules.
Structured inference needs PyTorch and NumPy; Gymnasium and SB3 are unnecessary.
Use one Torch/OMP/MKL thread for the strict fixture and CPU inference. CUDA is not
required: the recorded `cuda:0` device is historical training provenance. The
export was checked with Python 3.14.4, Torch 2.12.1+cu130 executing on CPU, and
NumPy 2.4.6. Deployment environments must also pass the exact fixture; the saved
version strings do not certify untested environments.

To create a different package, use an unused output directory and explicitly pin
the input checkpoint:

```sh
PYTHONPATH=python OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m ml_lab.tactical_v3_playable \
  --checkpoint /path/to/run/checkpoints/best.pt \
  --expected-checkpoint-sha256 <sha256> \
  --source-manifest /path/to/run/run.json \
  --package-id chosen-model --output models/ai/chosen-model \
  --evidence /path/to/completed-evaluation-report.md
```

The source manifest must identify that checkpoint. Existing output directories
are never overwritten. The exporter reserves a new directory exclusively and
publishes `run.json` last, so an interrupted export cannot be loaded as complete.
No source training files are rewritten or training state fabricated.
