# TotalSegmentator Probe Study

Frozen-encoder representation analysis: can `vertebrae_mr` U-Net features separate spine MRI contrasts (`t2w` vs `t2star`) without updating segmentation weights?

> Aggregate metrics + synthetic smoke only. Encoder was **frozen** throughout — this is linear/MLP probing, not U-Net fine-tuning.

## Setup

| Item | Value |
|------|------:|
| Embeddings extracted | 3,175 scan×layer vectors across **6** encoder layers |
| Methods | intensity baseline · logistic linear probe · MLP probe |
| Selection | layer chosen by k-fold balanced accuracy on train |
| External n | 534 spine-generic scans |

## Results

| Split | Method | Balanced accuracy |
|-------|--------|------------------:|
| In-domain test | linear (`layer_01`) | **0.977** |
| In-domain test | MLP | **0.981** |
| External | intensity baseline | 0.536 |
| External | **linear probe** | **0.861** |
| External | MLP | 0.517 |

Source: [`results/comparison_A_vs_B.json`](results/comparison_A_vs_B.json) (aggregate block only).

## Representation analysis

1. **In-domain probes saturate** — frozen TotalSegmentator features carry contrast-discriminative signal when train/test share the clinical domain.
2. **External linear transfer is partial** — 86.1% vs ~chance intensity baseline shows usable geometry/texture cues, but a ~12–15 pt drop vs in-domain indicates domain shift the probe cannot absorb.
3. **MLP overfits the source domain** — strong in-domain, collapses externally; capacity without adaptation hurts OOD.
4. **Implication for labeling** — probing de-risks “is there signal?” cheaply; a dedicated contrast classifier (see sibling benchmark repo) is still warranted before trusting deployment across sites.

## Smoke

```bash
pip install -r requirements.txt
python scripts/run_smoke_probes.py
```

## Layout

```text
src/        probe trainers + intensity baseline (sklearn)
scripts/    synthetic embedding smoke
results/    aggregate A/B comparison
docs/       layer-sweep + analysis notes
```

## Intentionally omitted

Real embeddings, clinical volumes, subject-level prediction CSVs, TotalSegmentator weights redistributed from upstream.
