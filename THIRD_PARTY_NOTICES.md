# Third-Party Notices and License Status

## Published scope

Publication authorized on October 18, 2026 covers the contributor's selected
Translator training pipeline. The collaborative model architecture,
canonical-state core, export/injection adapters, and SAM runtime remain
external to this repository. Their original provenance is recorded in
[PROVENANCE.md](PROVENANCE.md).

No project-wide license was present in the selected source snapshot. This
repository adds no MIT, Apache, or other license grant to the extracted
project code. Reuse permissions follow the applicable code owners' terms.

## External team core

The separately installed dependency is pinned to the
[team source revision `150a9e6`](https://github.com/memorybridge-team/vos-memory-translator-nonlinear/tree/150a9e66581e80d9837efdceb84c125da3d1a726).
That dependency includes model-team contributions whose authorship remains
with their original contributors. Installing the dependency is separate
from redistributing its source.

## SAM2 and other dependencies

SAM2 is developed by Meta and the original SAM2 authors. The upstream pin
used by the collaborative project is
[`2b90b9f`](https://github.com/facebookresearch/sam2/tree/2b90b9f5ceec907a1c18123530e92e794ad901a4).
SAM2 source and weights are obtained separately under their upstream terms.

PyTorch, NumPy, Pillow, test/build tools, and dependencies installed with
the external core retain their respective upstream terms. They are not
vendored in this repository. Dataset owners' terms govern videos, masks,
split files, and derived artifacts. This release contains pipeline code,
documentation, and a sanitized scalar training summary; datasets, cache
tensors, weights, private logs, and manuscripts remain outside its scope.
