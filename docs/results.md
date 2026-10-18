# Translator Training Results

The completed research run trained a Translator from cached SAM 2.1 Small
states to paired SAM 2.1 Base+ states. It used 294,793 valid training records
and selected epoch 26 with a full-validation state R² score of **0.619037**.
The pipeline performs state-supervised optimization, distributed metric
aggregation, checkpoint selection, and reproducible artifact export.

## Completed run

| Item | Recorded value |
|---|---|
| Training source revision | `165d81aad15f1ab7a05d319d1da975a1e7322093` |
| Completed epochs / optimizer updates | 28 / 128,996 |
| Stop reason | Validation state R² early stopping |
| Selected epoch / validation score Q | 26 / 0.6190369666836515 |
| Optimizer / initial learning rate / weight decay | AdamW / 0.0003 / 0.0001 |
| Global batch / world size / accumulation | 64 / 2 / 1 |
| Precision / augmentation / seed | FP32 / none / 7 |
| Initialization | Fresh Translator parameters |

Training optimized only the Translator. The cached-state trainer reads stored
spatial memories and object pointers; SAM model parameters remain frozen.
The recorded environment was Python 3.12.11, PyTorch 2.8.0+cu128, and CUDA
12.8. The run used `NCCL_P2P_DISABLE=1` for its two-GPU communication setup.

### Data coverage

| Dataset and role | Frozen membership videos | Used videos | Valid records |
|---|---:|---:|---:|
| MOSEv2 train | 3,300 | 2,963 | 265,945 |
| LVOSv2 train | 420 | 420 | 28,848 |
| **Training total** | **3,720** | **3,383** | **294,793** |
| MOSEv2 validation | 183 | 166 | 13,887 |
| LVOSv2 validation | 140 | 140 | 11,424 |
| **Validation total** | **323** | **306** | **25,311** |
| MOSEv2 internal test | 183 | Sealed | 0 for training or selection |

The MOSEv2 membership was frozen with seed 7. LVOSv2 retained its official
420/140 train/validation membership. Internal-test records were excluded
from training statistics and model selection.

### Selected-epoch state reconstruction

| Validation dataset | Spatial R² | Pointer R² |
|---|---:|---:|
| MOSEv2 | 0.806915 | 0.505647 |
| LVOSv2 | 0.805633 | 0.357952 |
| **Equal mean Q** | **0.619037 across all four values** | |

For each dataset and component, R² is one minus eligible-feature summed
squared prediction error divided by summed target variation about the
feature-wise validation mean. Spatial features are channels and pointer
features are coordinates. Features with population variance at most
`1e-12` are excluded. Q equally averages the four dataset/component values.

The optimization objective sums spatial and pointer MSE, each divided by
its training-target RMS squared with a `1e-12` denominator floor. The
recorded RMS values were 0.8042736065258159 and 0.6333327365530066,
respectively. These statistics scale the loss; inference inputs retain
their original values.

### Checkpoint selection and stopping

Best-checkpoint selection retains the highest completed-epoch Q. Early
stopping maintains a separate meaningful-improvement reference, updated
only by an absolute gain greater than 0.001. The configured minimum,
patience, and maximum were 15, 8, and 60 epochs. Epoch 26 improved the best
score while remaining below that reference's improvement threshold;
training completed after epoch 28.

## Evidence and interpretation

These numbers reuse the completed run's final delivery metadata and scalar
history, audited on October 11, 2026. The recorded producer checks include
strict checkpoint reload, parameter updates in both branches, and a maximum
rank-parameter difference of zero. The sanitized summary is
[`results/verified_training.json`](../results/verified_training.json).

This training-only extraction uses the later reviewed source snapshot
`150a9e6`, with namespace and external-dependency changes described in
[Provenance](../PROVENANCE.md). Its validation is reported separately in
[Validation](validation.md). The historical run remains bound to `165d81a`.

Q measures cached-state reconstruction. Downstream mask J&F, runtime
handoff, learned-linear comparisons, and architecture ablations are outside
the evidence reported here. MOSEv2 coverage reflects available selected
cache cases: 337 train and 17 validation membership videos had no selected
cases. Tested video/group intersections were zero; original clip lineage
remains incompletely verified. Raw data, caches, checkpoints, and private
run logs are retained outside this public repository.
