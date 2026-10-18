# Source Provenance and Contribution Scope

This repository presents the contributor's **Translator training-pipeline
work**: frozen video-level splits, role-aware indexes, training-only loss
statistics, distributed MSE optimization and validation R² aggregation,
checkpoint selection/resume, bounded remote execution, and training
artifact export.

## Source lineage

| Item | Revision |
|---|---|
| Collaborative source project | `memorybridge-team/vos-memory-translator-nonlinear` |
| Reviewed pipeline snapshot selected for extraction | `150a9e66581e80d9837efdceb84c125da3d1a726` |
| Source used by the completed training run | `165d81aad15f1ab7a05d319d1da975a1e7322093` |
| External model-team architecture and tensor API origin | `746ea3e7d84c366c2d7ac06159e90a1f684bca56` |

The original collaborative source and research artifacts are preserved.
The completed-run metrics in [Results](docs/results.md) are reused from
audited delivery metadata and scalar history.

## Training-only extraction

- The pipeline namespace is `vos_memory_training`; its console entry point
  is `vos-memory-training`.
- The training objective, validation R² reduction, optimizer/scheduler
  policy, batch contract, and default settings retain the selected source's
  computations.
- Core imports resolve to a separately installed team dependency pinned to
  `150a9e66581e80d9837efdceb84c125da3d1a726`.
- The frozen model lock resolves the architecture and tensor API from that
  external dependency. Package-source hashing covers this extracted
  pipeline, while the external frozen files retain their own hashes.

Namespace, import, and hash-scope changes create a new package identity.
Use it for newly bound runs. Historical run resume uses the original
source and bindings; the extraction does not relax those checks.

## External team core

The model architecture, frozen tensor API, canonical state schema,
export/injection adapters, and SAM runtime are **external dependencies**.
Their code is excluded from this repository and remains credited to the
respective project contributors. The pipeline contributor's scope is
distinct from model architecture authorship and SAM2 foundation-model
development.

The original migration record credits handle `fragile` for the model-team
revision above. Required LF-normalized frozen-file SHA256 values are:

| External file | SHA256 |
|---|---|
| `transformer_translator.py` | `60d56b6785e83c10d2dc313986189d93e8d223ebc031cc6054706986b63f4421` |
| `frozen_tensor_api.py` | `6e647775fa96726b26ca6f56e9ac4ef144ccd617c28d334e3baf664e7ad8e0ff` |

The external source is available at the
[team repository snapshot](https://github.com/memorybridge-team/vos-memory-translator-nonlinear/tree/150a9e66581e80d9837efdceb84c125da3d1a726).
Dataset, checkpoint, cache, manuscript, private-log, and access-configuration
files are also excluded. See [Third-Party Notices](THIRD_PARTY_NOTICES.md)
for dependency and licensing scope.
