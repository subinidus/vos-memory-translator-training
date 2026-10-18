# Contributions and Scope

This repository presents the **training-pipeline contribution** to a collaborative cross-model VOS memory study.

| Area | Responsibility and source |
|---|---|
| Fixed video-level split and role-aware indexes | Portfolio contributor |
| Paired-state loader integration and record/file checks | Portfolio contributor |
| Train-only RMS loss scaling and full-validation R² aggregation | Portfolio contributor |
| DDP, fixed global batch, gradient accumulation, and valid-record weighting | Portfolio contributor |
| Checkpoint selection, strict reload, resume, and execution limits | Portfolio contributor |
| Verification coordination, training operations, interpretation, manuscript preparation | Portfolio contributor |
| Frozen Translator architecture and tensor API | Model-design team; external research core |
| Canonical schema, cache contracts, state export/injection, runtime diagnostics | Collaborative research core; external dependency |
| Downstream handoff, segmentation, and benchmark evaluation | Evaluation collaborators |
| SAM2 foundation model and predictor | Meta and the original SAM2 authors |

Training optimizes the Translator while SAM2 weights remain frozen. The individual repository packages the loader/index/statistics/training/checkpoint workflow; the model team's architecture and the runtime code retain their separate provenance and ownership in the team repository.

The historical experiment and the extracted package have distinct source identities. Results, evidence reuse, and release checks are documented in [Results](results.md) and [Validation](validation.md).

Final author order and publication metadata require collaborator confirmation. Source revisions and frozen model hashes are recorded in [Provenance](../PROVENANCE.md).
