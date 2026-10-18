# Training Pipeline

`vos_memory_training` exposes the cached-state training workflow through `vos-memory-training`. Run commands from the repository root in a prepared environment with authorized data and the pinned research core. Data-dependent examples below are templates; replace their paths with the corresponding approved artifacts.

## Frozen external core

The optional `research-core` extra pins `memorybridge-team/vos-memory-translator-nonlinear` at:

```text
150a9e66581e80d9837efdceb84c125da3d1a726
```

It provides `vos_memory_inspector` model and schema dependencies. The model-design files originate from revision `746ea3e7d84c366c2d7ac06159e90a1f684bca56` and retain these LF-normalized SHA256 values:

| External file | SHA256 |
|---|---|
| `transformer_translator.py` | `60d56b6785e83c10d2dc313986189d93e8d223ebc031cc6054706986b63f4421` |
| `frozen_tensor_api.py` | `6e647775fa96726b26ca6f56e9ac4ef144ccd617c28d334e3baf664e7ad8e0ff` |

The training adapter retains the frozen-source checks. Establish a new model binding before using a different model revision. The package consumes the external model and state contracts through imports; its public source covers the training pipeline.

## Interface

```bash
python -m vos_memory_training --help
```

```bash
vos-memory-training train --help
```

| Stage | Command | Purpose |
|---|---|---|
| Inventory | `inventory` | Compare an official video list with RGB directory metadata |
| Frozen membership | `freeze` | Create a new MOSE video split or reuse matching frozen membership |
| Role plan | `plan` | Report roles, storage locations, and exclusions |
| Inspection plan | `inspection-plan` | Separate reusable file-bound inspection evidence from new inspection work |
| Training index | `index` | Bind roles, tensors, valid records, alignment, and file checksums |
| Statistics | `statistics` | Compute train-only target RMS and validation target sufficient statistics |
| DDP training | `train` | Run bounded smoke or official CUDA training |
| Status | `monitor` | Read `LIVE_STATUS.json` |

## Input preparation

Obtain datasets, original inventories, case manifests, paired-state caches, and weights under their access terms. `configs/state_training_datasets.example.json` illustrates the required path fields. Paths resolve from the working directory; JSON paths use literal values.

The recorded MOSE membership is 3,300 train / 183 validation / 183 internal test, seed 7. LVOS uses official 420/140 train/validation membership. Cache storage directories such as `fit` and `development` are locations; frozen video membership determines the training role. Original group lineage and the exact inventory are part of a split binding.

For an existing experiment, reuse the frozen split and matching index/statistics after checking their binding and SHA sidecars. For a new experiment, create new contracts in a separate output directory. Keep internal-test records outside training, RMS statistics, and checkpoint selection.

### Inventory example

```bash
python -m vos_memory_training inventory --dataset MOSEv2 --official-split train --video-root data/MOSEv2/train/JPEGImages --official-list data/contracts/mose_official_train_videos.txt --output artifacts/inventories/mose_train.json
```

### New split example

Use this only when establishing a genuinely new split:

```bash
python -m vos_memory_training freeze --inventory artifacts/inventories/mose_train.json --output artifacts/new-mose-split --seed 7
```

Use the authorized original `data/frozen-mose` directory for exact membership reproduction. The following preparation examples assume that directory already exists.

### Inspection planning

```bash
python -m vos_memory_training inspection-plan --datasets configs/state_training_datasets.example.json --split data/frozen-mose --output artifacts/inspection_plan.json
```

### Index and statistics

Run CPU preparation on the intended data host. In a POSIX shell, hide CUDA explicitly:

```bash
CUDA_VISIBLE_DEVICES="" python -m vos_memory_training index --datasets configs/state_training_datasets.example.json --split data/frozen-mose --output artifacts/index --operator-completed "confirmed-source-cache-preparation" --max-new-inspections 0
```

Set `--operator-completed` from the actual completion record. With `--max-new-inspections 0`, reusable inspection evidence must match the same cache files. The index validates Small/Base+ direction, object/frame/slot/validity alignment, tensor shape/dtype/finite values, role membership, and checksums. Store new index outputs separately from source caches.

```bash
CUDA_VISIBLE_DEVICES="" python -m vos_memory_training statistics --index artifacts/index/index.json --output artifacts/statistics --config configs/state_training_r2.json
```

Training-target RMS scales the spatial and pointer losses. Validation targets provide R² sufficient statistics. Inference inputs retain their original representation. In PowerShell, set `$env:CUDA_VISIBLE_DEVICES = ''` before CPU preparation instead of using the POSIX prefix.

## Objective and DDP policy

Training minimizes spatial MSE plus pointer MSE, with each term divided by its train-target RMS squared and a denominator floor of `1e-12`. Padding and invalid records are excluded. Logs separate the two component losses.

The saved configuration uses AdamW, LR `3e-4`, weight decay `1e-4`, seed 7, FP32, no augmentation, and global batch 64. For two GPUs, microbatch 32 × two ranks × accumulation 1 gives 64. The scheduler assigns complete paired records across ranks, preserves record metadata, synchronizes gradients, and weights updates by valid-record counts. When world size changes, `global_batch / (world_size × microbatch)` must remain a valid integer accumulation factor.

Selection uses `Q`, the equal mean of MOSE spatial/pointer and LVOS spatial/pointer validation R². Each score aggregates full-dataset sufficient statistics. The highest completed-epoch Q selects the checkpoint. Early stopping uses a separate meaningful-improvement reference: minimum 15 epochs, patience 8, absolute improvement threshold `0.001`, maximum 60 epochs. A small new best score can update the selected checkpoint while leaving the stopping counter unchanged.

## Smoke and official training

The trainer initializes a distributed process group even on CPU, so launch the training subcommand through `torchrun`. Inspect the full parser before supplying a new run's inputs:

```bash
python -m vos_memory_training train --help
```

A data-dependent CPU smoke template is:

```bash
CUDA_VISIBLE_DEVICES="" torchrun --standalone --nnodes=1 --nproc-per-node=1 -m vos_memory_training train --index artifacts/index/index.json --statistics artifacts/statistics --config configs/state_training_r2.json --output artifacts/cpu-smoke --mode smoke --device cpu --max-wall-seconds 120
```

This uses real records when the index points to real caches and can require substantial RAM. Dataset-free synthetic tests are a smaller first check.

For CUDA, supply the exact approved physical UUID for each local rank, explicit execution approval, start/deadline, whole-Pod hourly rate, budget, and wall limit. Confirm UUID mapping and run a small communication test before model smoke. Apply a transport workaround only to the diagnosed run's processes.

Official `--mode train` requires a real bound CUDA smoke result through `--smoke-evidence`. Smoke and training must share source, config, index, statistics, model identity, and rank mapping. Smoke verifies finite losses, both branch updates, strict reload, and rank-parameter equality. Preserve failed attempts and keep each smoke/training output in its own directory.

Cloud access, service authentication, and job launching are prepared separately. `remote_run.py` retains the research controller/watchdog and bounded cleanup implementation. Its `validate_config()` is specific to the recorded two-GPU approval: USD 1.16/hour, USD 30, and 12 hours, with an isolated `/workspace/r2-*` operation directory. Review and explicitly rebind those deployment checks before adopting the controller for another job. The `train` CLI separately accepts explicit rate, budget, and deadline inputs. Whole-Pod rates already include the GPUs specified by the provider; cost accounting includes preparation and idle time from the declared approval start. Stopping Python and stopping Pod billing are separate operations.

## Checkpoints and source identity

Atomic complete checkpoints preserve model, optimizer, scheduler, RNG, epoch, step, statistics, and source/data/config identity. Rank 0 saves checkpoints and exports. Resume uses the last complete epoch boundary and retains the selection and patience history.

The extracted `vos_memory_training` package has a new package-source identity. **Start a new run with this package.** A historical checkpoint resumes only in its exact original research source and environment with matching world size and input bindings. Keep that original code and checkpoint together rather than rewriting the identity to match this package.

The delivery loader, `vos_memory_training.delivery.load_translator`, loads a selected Translator export for inference under the frozen external core contract. Loading an export and resuming an optimizer checkpoint are separate operations. Downstream state injection and evaluation use the evaluation team's runtime implementation.

## Monitoring and artifacts

```bash
python -m vos_memory_training monitor --run artifacts/training
```

This reads the run's `LIVE_STATUS.json`. Outputs include epoch/component metrics, `history.json`, complete checkpoint metadata, validation-based selection, ETA files, and delivery artifacts. Check readiness metadata and SHA sidecars before transferring a delivery. Normal completion, a bounded partial stop, and an error remain distinct states.

The execution controller performs limit checks and artifact finalization remotely. Preserve source/config/index/statistics, logs, used/excluded lists, and the selected and last-complete checkpoint information for handoff. Downstream benchmark results are documented by the evaluation team under their own checkpoint-bound protocol.
