# hERG Blockade Prediction - CDD 203 Homework 1

**Group members:** Armine Lalani, Awani Gadre, Faris Raza, Kavin Varadharajulu

**Web app:** TODO - paste the public URL here

Competition entered: **highest specificity with sensitivity > 0.70**.

| | scaffold split (honest) | random split |
| --- | --- | --- |
| specificity | **0.791** | 0.859 |
| sensitivity | 0.727 | 0.787 |
| ROC-AUC | 0.840 | 0.900 |

## Writeup

**Decision threshold.** We submit 0.53 rather than the default 0.5. A prediction is
made by testing `probability >= THRESHOLD`, so raising the threshold flags fewer
molecules as blockers: specificity rises and sensitivity falls. Both errors are
costly here, but differently. A false negative lets a cardiotoxic compound advance;
a false positive discards a safe compound that might have become a useful drug.
Our assigned target was the highest specificity we could reach while still catching
at least 70% of true blockers, so we pushed the threshold up to the point where
sensitivity is about to break that floor. Choosing that point needs care, because
the estimate of sensitivity is itself noisy. We tuned on out-of-fold predictions
from a **scaffold-split** cross-validation (below), then bootstrapped by resampling
whole scaffold groups, and kept the highest threshold whose sensitivity still
cleared 0.70 at the 2.5th percentile of its interval. That is 0.53: sensitivity
0.727 with a lower bound of 0.707, and specificity 0.791. The next step up, 0.55,
looks better on the point estimate (specificity 0.815) but its sensitivity interval
reaches down to 0.681, which would fail the constraint outright.

**Feature engineering.** Each SMILES string becomes 39 numbers: 25 physicochemical
descriptors computed with RDKit plus 14 counts taken directly off the raw string
(length, element-letter counts, digits, branch and charge symbols). The width is
fixed regardless of input length, which satisfies the requirement that the
representation not depend on the SMILES string's size; we verified this explicitly
on valid molecules of 197 and 200 characters, longer than anything in
`hw1data.tab` (maximum 177), as well as on malformed and empty input. Two choices
mattered. First, **desalting**: 239 molecules in the dataset are salts, and RDKit
otherwise computes molecular weight, logP and TPSA across the counter-ion as well,
so we keep only the largest fragment and retain the fragment count and formal
charge as separate features. This was the single largest improvement we measured.
Second, a **SMARTS pattern for protonatable nitrogen**, which excludes amide,
nitro, sulfonamide and aniline nitrogens. A basic amine carries a positive charge
at physiological pH and ion-pairs in the hERG pore, so it is the key pharmacophore;
a plain nitrogen count does not capture it. We also tried 2048-bit Morgan
fingerprints, the standard approach, and these 39 descriptors beat them (specificity
0.815 vs 0.784) with 52x fewer columns.

**Model selection and evaluation.** The classifier is a scikit-learn `Pipeline` of
`StandardScaler` then `ExtraTreesClassifier(n_estimators=100, min_samples_leaf=2,
random_state=0)`. ExtraTrees beat RandomForest on this feature set (specificity
0.781 vs 0.766) and logistic regression well behind both; the features interact,
and a linear model cannot represent that. The two hyperparameters are not accuracy
tuning. A fully grown 500-tree forest scored higher (0.818) but exported to 138 MB
against the 50 MB ONNX limit, so 100 trees with `min_samples_leaf=2` is simply the
largest forest that fits; it costs 0.003 specificity and also brought float32
export parity from 2e-03 to 2e-07. Most importantly, we evaluated with
**scaffold splitting** (`StratifiedGroupKFold` over Bemis-Murcko scaffolds) rather
than a random split. 71% of these molecules share a scaffold with another, so a
random split lets the model see the same chemical series in train and test, and it
reports ROC-AUC 0.900 where the scaffold split reports 0.840. Since the held-out
test set is described as different in composition, we treated the scaffold number
as the real one and tuned the threshold against it.

## Use of LLMs

TODO - WRITE THIS YOURSELVES, HONESTLY. Worth covering if true: using Claude to
build the cross-validation harness and compare candidate featurizers and models;
having it measure the ONNX export size and memory footprint against the hosting
limits; that it found two bugs in the web app (a missing RDKit import that silently
zeroed every descriptor, and an unimported module that froze the validation state);
and which decisions and code each of you made yourselves.

## Repository contents

| file | what it is |
| --- | --- |
| `Part1_Armine-Lalani-Awani-Gadre-Faris-Raza-Kavin-Varadharajulu.py` | training script: featurize, build_model, threshold, ONNX export |
| `Part1_Armine-Lalani-Awani-Gadre-Faris-Raza-Kavin-Varadharajulu.onnx` | the submitted model, 18.8 MB |
| `requirements.txt` | pinned dependencies |
| `Part 2.zip` | web app source plus a byte-identical copy of the ONNX |
| `part2/` | the same app files, unzipped |

## Reproducing

```
python -m venv .venv && source .venv/bin/activate     # Python 3.10-3.12 only
pip install -r requirements.txt
python "Part1_Armine-Lalani-Awani-Gadre-Faris-Raza-Kavin-Varadharajulu.py"
python TestModel.py "Part1_Armine-Lalani-Awani-Gadre-Faris-Raza-Kavin-Varadharajulu.py"
```
