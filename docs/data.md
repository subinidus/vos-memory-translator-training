# Data and input contracts

The pipeline reads authorized paired Small/Base+ states. Its cache adapter
checks direction, object/frame/slot identity, conditioning flags, validity,
tensor shape/dtype, finite values, file stability, and SHA bindings. The
canonical state classes come from the separately installed research core.

## Roles and coverage

The recorded experiment used a frozen MOSE official-train membership of
3,300 / 183 / 183 train/validation/internal-test videos, with seed 7. LVOS
retained its official 420 / 140 train/validation membership. Existing cache
folders named `fit` and `development` identify storage locations; video
membership determines the training role.

Original inventory, group mapping, and frozen membership are required to
reproduce the split. Use an existing authorized frozen split for reproduction;
the `freeze` command creates a contract for a new experiment. Internal-test
records remain sealed from training, normalization, and checkpoint selection.

## User-managed inputs

`configs/state_training_datasets.example.json` shows the required fields.
Paths resolve from the working directory. Supply official full-video
inventories, case manifests, cache storage memberships, and cache roots
before generating an index. Keep input files and SHA sidecars together.

```text
data/contracts/      original inventories, case/storage manifests
data/frozen-mose/    immutable split and its SHA sidecars
data/paired/         authorized .pt cache files and SHA sidecars
artifacts/index/    derived index and used/excluded inventory
artifacts/stats/    train-only RMS and validation target statistics
artifacts/run/      history, complete checkpoints, and delivery markers
```

The example is a schema guide. Actual cache preparation is owned by the data
pipeline, and original records remain unchanged during training preparation.
`--max-new-inspections 0` reuses matching prior inspection evidence; a fresh
inspection needs an explicit bounded allowance.

## Artifacts and weights

Videos, GT masks, original manifests, cache tensors, and model weights require
separately authorized access. This repository distributes training code,
configuration examples, tests, and a scalar summary of the recorded result.
The selected model's output can be reproduced with its weights and original
bindings; fresh training uses the new package's own source identity.
