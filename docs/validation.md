# Validation and Reused Evidence

This repository extracts the Translator training pipeline into
`vos_memory_training`. Validation distinguishes source-level checks of this
extraction from previously recorded research execution.

## Reused execution evidence

The completed two-GPU run used source revision
`165d81aad15f1ab7a05d319d1da975a1e7322093`. Its final delivery records 28
complete epochs, selection of epoch 26 by full-validation state R², strict
checkpoint reload, updates in both Translator branches, and zero maximum
rank-parameter difference. Final delivery metadata and scalar history were
audited on October 11, 2026. [Results](results.md) summarizes that run.

The later source review at `150a9e6` recorded unchanged executable Python
AST relative to the completed run, apart from removing an unreferenced
helper. These records establish the origin of the selected pipeline code.

## Checks of this extraction

Checks executed on October 18, 2026 produced the following results:

| Check | Observed result | Scope |
|---|---|---|
| Python syntax | 22 AST files parsed | 18 training modules and 4 test modules |
| Retained computations | PASS | Namespace-normalized AST equality for 17 modules; retained model-computation functions match the selected source |
| External frozen-source lock | PASS | Expected external file hashes accepted and a changed text fixture rejected |
| CLI module `--help` | 10 invocations, all exit 0 | Main parser, eight subcommands, and remote-run parser |
| Standard-library synthetic tests | 32 passed; 0 failures, errors, or skips | Synthetic metadata and scalar numerical contracts |
| Release boundary | PASS | Architecture and runtime files excluded; external core revision pinned |

The synthetic suite reported `torch_imported=false` and
`core_imported=false`. The frozen-source lock check used text-file hashes;
it did not instantiate the external model. CLI checks exercised parser
availability through `python -m`, rather than an installed-console or
training execution. The release review retains
`TRAINING_RELEASE_VERIFICATION.json`, the ten help logs, and
`training_metadata_tests.stdout.log`/`training_metadata_tests.stderr.log`.

The earlier complete-project bundle recorded a 72-test CPU suite and a
separate synthetic model test. Those counts belong to that larger bundle,
which included the model and runtime core. They are excluded from this
training-only repository's test totals.

## Execution scope

No new tensor operation, CUDA/NCCL test, model execution, training run, or
real-cache index/statistics regeneration was performed while preparing this
extraction. The historical checkpoint binaries were not loaded for this
review. The external team core supplies the locked Translator and state
contracts. Running the extracted trainer requires that exact dependency,
authorized paired caches, and newly bound run artifacts.

The namespace and dependency-boundary changes alter package identity. Use
this package for a new bound run; resume the historical run with its exact
original source, environment, index, statistics, and checkpoint bindings.
