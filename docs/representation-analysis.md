# Representation analysis

Probing asks whether an off-the-shelf segmentation encoder already separates contrasts **without fine-tuning**. The gap between in-domain saturation and external linear performance bounds how far “free” TotalSegmentator features go before task-specific models are required.

## What we plot

Figures in [`assets/`](../assets/) are **2D PCA projections** of frozen U-Net encoder embeddings (global average pool per scan), colored by contrast (`t2w` vs `t2star`). Decision boundaries show the fitted linear or MLP probe in the same 2D subspace.

These are aggregate visualizations for the portfolio demo. They do **not** include subject IDs, DICOM paths, or downloadable embedding tensors.

## How to read the figures

### In-domain linear PCA (`indomain_linear_pca.png`)

Train and test share the clinical full-spine domain. Classes separate cleanly; the linear probe reaches ~98% balanced accuracy. This confirms contrast signal is present in frozen encoder features.

### External linear PCA (`external_linear_pca.png`)

Same probe trained on full-spine, evaluated on spine-generic (different domain). Separation remains visible (~86% balanced accuracy) but overlap increases versus in-domain — partial transfer, not saturation.

### External MLP PCA (`external_mlp_pca.png`)

The MLP probe looks strong in-domain but collapses externally (~52% balanced accuracy, near chance). Extra capacity fits source-domain quirks that do not generalize.

### Layer panels (`embeddings_layer_panels.png`)

PCA geometry across encoder depths. Layer selection (`layer_01`) was driven by k-fold balanced accuracy on the training partition, not by eyeballing these panels alone.

## Takeaway

| Observation | Implication |
|---|---|
| Linear probe transfers partially OOD | Frozen TotalSeg features are usable but not deployment-ready alone |
| MLP overfits source domain | Prefer simple probes when generalization matters |
| Intensity baseline ~chance OOD | Contrast cue lives in learned representations, not raw histogram stats |

See also: [`layer-sweep.md`](layer-sweep.md) · [`../results/comparison_A_vs_B.json`](../results/comparison_A_vs_B.json)
