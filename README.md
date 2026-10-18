# Translator Training Pipeline for Cross-Model VOS Memory

Training-pipeline portfolio for the collaborative study **Segmentation Quality and Handoff Cost of Cross-Model Memory-State Transfer for Video Object Segmentation**.

## Purpose

This project trains a Translator to reconstruct **SAM 2.1 Base+** spatial memory and object pointers from aligned **SAM 2.1 Small** cached states. It packages the data contracts, distributed training, model selection, checkpointing, and bounded execution workflow in `vos_memory_training`.

The trainable Translator and tensor contract come from the model-design team's frozen research core. SAM2 weights remain frozen. This repository contains the training pipeline; the research core is an external, revision-pinned dependency.

## My contributions

- Designed reproducible video-level splits and role-aware indexes while preserving fixed train/validation/internal-test membership.
- Connected paired-state caches to the loader with object/frame/slot/validity checks and file bindings.
- Implemented train-only RMS loss statistics and full-validation R² aggregation across datasets and state components.
- Built DDP training with valid-record weighting, gradient accumulation, and a fixed global batch of 64.
- Implemented validation-based checkpoint selection, strict reload, reproducible resume, and deadline/cost guards.
- Coordinated verification and training operations, interpreted results, and contributed to manuscript preparation.

See [Contributions](docs/contributions.md) for the team's responsibility boundaries.

## Verified training result

The historical training run completed **28 epochs and 128,996 optimizer updates**. Full-validation state R² selected **epoch 26**, with **Q = 0.6190369666836515**.

| Validation dataset | Spatial R² | Pointer R² |
|---|---:|---:|
| MOSEv2 | 0.806915 | 0.505647 |
| LVOSv2 | 0.805633 | 0.357952 |

`Q` is the equal mean of these four target-state reconstruction scores. The experiment used 294,793 training records and 25,311 validation records, AdamW, FP32, global batch 64, and train-only RMS loss scaling.

These results come from the completed research run at `165d81aad15f1ab7a05d319d1da975a1e7322093`. [Results](docs/results.md) records the evidence and coverage; [Validation](docs/validation.md) separates that historical execution from checks of this extracted package.

## Installation

Use a separate environment with an appropriate PyTorch build. The following is an installation command for that environment:

```bash
python -m pip install -e ".[dev,research-core]"
```

The `research-core` extra installs the team repository at `150a9e66581e80d9837efdceb84c125da3d1a726`; repository access is required. It supplies the frozen model, canonical tensor schema, and cache contracts. The model files retain their revision and SHA locks. See [Training](docs/training.md#frozen-external-core).

## CLI

```bash
python -m vos_memory_training --help
```

The installed console entry point exposes the same interface:

```bash
vos-memory-training --help
```

Subcommands are `inventory`, `freeze`, `plan`, `inspection-plan`, `index`, `statistics`, `train`, and `monitor`. See [Training](docs/training.md) for preparation, DDP policy, smoke requirements, and output files, and [Data](docs/data.md) for input contracts.

## Scope and reproducibility

Datasets, paired caches, original membership/manifests, trained weights, and private operational records require separately authorized access. Reproducing the recorded experiment requires its exact source and input bindings; a split seed alone is insufficient.

The extracted package has a new source identity. **Start a new run with this package.** Resume a historical checkpoint only with its original source, environment, world size, index, statistics, and recorded bindings. Selected-weight inference loading is a separate operation.

Source attribution and licensing are documented in [Provenance](PROVENANCE.md) and [Third-Party Notices](THIRD_PARTY_NOTICES.md).
