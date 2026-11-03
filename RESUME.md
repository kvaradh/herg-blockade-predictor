# HW1 hERG classifier — state as of 2026-11-02

All numbers below are measured on the frozen harness. Scaffold-split figures are the
honest estimate of the grader's unseen, differently-composed test set; random-split
figures are shown only to expose how optimistic they are.

## Environment

`.venv/` is Python 3.12 built with `uv`. There is no conda on this machine and the system
Python is 3.15, which **cannot install the assignment's pins**. Recreate with:

```
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python numpy==1.26.4 pandas==2.3.3 \
  scikit-learn==1.6.1 onnxruntime==1.19.2 skl2onnx==1.20.0 rdkit==2023.9.6
```

## Tooling

| file | purpose |
|---|---|
| `harness.py` | FROZEN protocol: StratifiedGroupKFold(5, seed 0) on Murcko scaffolds + random CV, threshold sweep, specificity at sensitivity>=0.70, TestModel contract checks. `python harness.py experiments/<c>.py` |
| `tools_timing.py` | serial, uncontended featurize throughput (the earlier parallel numbers were wrong by up to 7x) |
| `tools_onnx_check.py` | export + size + sklearn/onnxruntime parity + inference throughput. `python tools_onnx_check.py <candidate>` |
| `experiments/` | ~100 throwaway probes. NOT submission code. |
| `results/` | one JSON per run; ignore any with `n_features == -1` or `scaffold_auroc == 0` (partial writes from killed agents) |

## Dataset

10,000 rows, 52.4% blockers, no duplicate SMILES, no conflicting labels.
Length min 12 / median 55 / max 177 (grader feeds up to ~200).
239 salts, 3,866 charged, 3,924 stereo.
4,609 Murcko scaffolds; **70.8% of molecules share a scaffold with another**; largest
series 82. That sharing is exactly what makes random CV optimistic.

## THE TWO CANDIDATES TO CHOOSE BETWEEN

### For "highest specificity with sensitivity > 0.7"  →  `desc_desalt_et100_msl2`

39 features (25 desalted RDKit descriptors + 14 raw-SMILES string features),
ExtraTrees(n_estimators=100, min_samples_leaf=2) in a StandardScaler pipeline.

```
scaffold AUROC      0.8399     random AUROC 0.9043  (optimism +0.064)
SPECIFICITY         0.8149  at sensitivity 0.7038,  threshold 0.550
featurize            3,424 molecules/sec
ONNX                 18.81 MB (limit 50), parity 2.1e-07, 454,583 rows/sec
```

### For "fastest model with AUROC > 0.725"  →  `str_ngram1024` (recommended) or `str_bincount_density`

Pure-string featurizers: no RDKit, no molecule parsing at all.

| candidate | scaffold AUROC | margin over 0.725 | featurize mol/sec | ONNX |
|---|---|---|---|---|
| `str_bincount_density` | 0.7311 | +0.006 — thin | **593,387** | 0.01 MB, 5.3M rows/sec |
| `str_ngram1024` | 0.7563 | +0.031 — safe | 180,378 | 0.01 MB, 11.0M rows/sec |
| `str_ngram4096` | 0.7915 | +0.067 | 103,711 | 0.04 MB, 1.2M rows/sec |
| `str_bincount256` | 0.7255 | +0.0005 — do not use | 1,352,174 | — |

ONNX inference is 10-60x faster than featurize in every case, so **combined throughput is
essentially the featurize rate**. For reference ECFP-2048 runs at 10,993 mol/sec, so
`str_ngram1024` is ~16x faster and `str_bincount_density` ~54x.

**You cannot win both competitions with one file** — the submission is a single ONNX.
Decide which category to enter.

## Findings worth putting in the writeup

1. **Nothing reached scaffold AUROC 0.85** across ~110 candidates; ceiling was 0.8399.
   The "fastest with AUROC > 0.85" competition is likely unwinnable for the whole class.
2. **Desalting was the single biggest featurizer win** — strip to the largest fragment
   before computing descriptors. 239 molecules are salts and their descriptors are
   otherwise computed over the counter-ion too.
3. **A 39-feature descriptor set beats 2048 fingerprint bits** (0.8149 vs 0.7840
   specificity) and is interpretable. Fingerprints were not the answer here.
4. **Regularisation matters more than feature choice on ECFP**: C=1 -> C=0.1 moved
   specificity 0.7590 -> 0.7840. The optimum is a plateau over C in [0.03, 0.1];
   C=0.3 falls to 0.7725. Bracketed, no further gain.
5. **The 50 MB ONNX limit is a real design constraint, not a formality.**
   ExtraTrees(500) fully grown = **138 MB, disqualified**. Shrinking to
   (100 trees, min_samples_leaf=2) = 18.81 MB cost only 0.003 specificity
   (0.8176 -> 0.8149). Check size early.
6. **Trees generalise worse across scaffolds than regularised LR** (+0.064 vs +0.041
   optimism gap), but their scaffold score is still higher, so they win on the honest metric.
7. Tree ONNX parity drifts when leaves are tiny (2e-03 at min_samples_leaf=1 vs 2e-07 at
   min_samples_leaf>=2) because thresholds are stored in float32. Another reason to avoid
   fully grown trees.
8. **Never trust throughput measured under load**: identical featurize code measured
   10,093 vs 1,430 mol/sec depending on CPU contention.

## Next steps

1. Decide: specificity category or speed category.
2. Write `Part1_<GROUP_NAMES>.py` by hand — `featurize`, `build_model`, `THRESHOLD`.
   The experiments are evidence only. Read the chosen candidate, understand it, rewrite it.
3. **Threshold**: do not use the sens>=0.70 threshold directly. The measured
   random->scaffold sensitivity drop was ~0.064, so tune at sensitivity ~0.76 for margin.
   For `desc_desalt_et100_msl2` the sens>=0.76 point was threshold 0.510 -> specificity 0.7666.
   Remember THRESHOLD must be in (0,1) and != 0.5.
4. Confirm with Peter whether the grader binarises with YOUR threshold or sweeps the ROC
   itself. The whole specificity strategy assumes the former.
5. Run `tools_onnx_check.py` on the final model, then `TestModel.py` in a fresh env.
6. Part 2: Gradio app on HF Spaces. Real-time SMILES validation gating the submit button
   (`Chem.MolFromSmiles` is not None). `Part 2.zip` must contain a byte-identical .onnx.
7. `git init` here; `.gitignore` should exclude `.venv/`, `experiments/`, `results/`, `data/`.

## Contract reminders (TestModel.py enforces)

- `featurize` must be STATELESS per molecule — batch vs one-at-a-time compared at 1e-4.
  No batch statistics. Scaling belongs inside the sklearn Pipeline.
- Must handle salts, charges, stereo, a 194-char SMILES, and never raise. Finite values only.
- `requirements.txt` must be a SUBSET of TestModel's ALLOWED set — no matplotlib.
- README.md must contain "http".
