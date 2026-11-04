# hERG Blockade Prediction - CDD 203 Homework 1

**Group members:** Armine Lalani, Awani Gadre, Faris Raza, Kavin Varadharajulu

**Web app:** https://herg-blockade-predictor.onrender.com

> Hosted on Render's free tier, which suspends an idle instance. The first
> request after a quiet spell takes about 50 seconds to wake the service;
> every request after that is immediate.

Competition entered: **highest specificity with sensitivity > 0.70**.

| | scaffold split (honest) | random split |
| --- | --- | --- |
| specificity | **0.791** | 0.859 |
| sensitivity | 0.727 | 0.787 |
| ROC-AUC | 0.840 | 0.900 |

The scaffold-split column is our estimate of performance on the graded test set,
which is described as different in composition from the data we were given. The
random-split column is shown only to make the gap between the two visible.

## Where each submission item is

| required | where |
| --- | --- |
| training script | `Part1_Armine-Lalani-Awani-Gadre-Faris-Raza-Kavin-Varadharajulu.py` |
| saved model (ONNX, 18.8 MB) | `Part1_Armine-Lalani-Awani-Gadre-Faris-Raza-Kavin-Varadharajulu.onnx` |
| public web app URL | top of this file |
| deployed app code | `Part 2.zip`, also unzipped in `part2/` |
| writeup (threshold, features, model) | [Writeup](#writeup) below |
| reflection on LLM use | [Use of LLMs](#use-of-llms) below |

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

We used generative AI throughout the assignment as a coding, experimentation, and debugging tool. We wrote and reviewed parts of the modeling workflow ourselves, while using ChatGPT and Claude to help draft and revise code, understand assignment requirements, and troubleshoot problems. In particular, we used Claude to help build the scaffold-split cross-validation harness and systematically compare candidate featurizers, models, hyperparameters, and decision thresholds. We also used AI to measure model size and memory usage against the assignment's 50 MB ONNX limit and the deployment host's 512 MB memory limit, and to help build and debug the deployed web app. During this process, AI-assisted review caught several issues, including the app footer still describing a previous model, an unnecessary `sklearn` import left behind when the featurizer was moved into the app, and a float64/float32 mismatch that would have caused deployed predictions to fail. We did not rely on generated code without checking it: we reviewed the implementations, compared model performance ourselves, ran the provided `TestModel.py`, and manually tested the final deployed app to verify that its predictions matched the submitted model.

## The web app

`part2/app.py` is a single-file Gradio app, deployed on Render and served from the
URL above. It loads the submitted ONNX file directly, and `featurize` and
`THRESHOLD` are copied verbatim from the training script, so the app and the
submitted model agree by construction rather than by coincidence.

Validation runs as you type: the Predict button stays disabled until RDKit parses
the string, and the box shows either the reason it is invalid or the molecular
formula, heavy-atom count and a drawing of the structure.

Checked against the deployed service, every molecule `TestModel.py` prints comes
back identical from the live app:

| SMILES | TestModel | live app |
| --- | --- | --- |
| `CCO` | 0.105 | 10.5% |
| aspirin | 0.050 | 5.0% |
| diphenhydramine·HCl | 0.726 | 72.6% |
| fluoxetine | 0.905 | 90.5% |
| terfenadine | 0.917 | 91.7% |
| 194-character PEG chain | 0.284 | 28.4% |

## Repository contents

| file | what it is |
| --- | --- |
| `Part1_...-Varadharajulu.py` | training script: `featurize`, `build_model`, threshold, ONNX export |
| `Part1_...-Varadharajulu.onnx` | the submitted model, 18.8 MB |
| `Part 2.zip` | web app source plus a byte-identical copy of the ONNX |
| `part2/` | the same app files unzipped, which is what Render deploys |
| `requirements.txt` | pinned dependencies for training |
| `TestModel.py` | the course's checker, unmodified |
| `data/hw1data.tab` | the provided dataset, unmodified |
| `render.yaml` | deployment configuration |

## Reproducing

```
python -m venv .venv && source .venv/bin/activate     # Python 3.10-3.12 only
pip install -r requirements.txt
python "Part1_Armine-Lalani-Awani-Gadre-Faris-Raza-Kavin-Varadharajulu.py"
python TestModel.py "Part1_Armine-Lalani-Awani-Gadre-Faris-Raza-Kavin-Varadharajulu.py"
```

Training takes about 45 seconds on a laptop CPU. We ran this in a clean virtual
environment built from `requirements.txt`: it reproduces the committed `.onnx`
byte for byte, and `TestModel.py` reports 13 of 13 PASS.

### One note on the app's dependencies

`part2/requirements.txt` adds `requests==2.34.2`, which is not on the assignment's
allowed list. It is not used by our code. gradio 5.35.0 imports it internally
through `gradio.cli`, and huggingface-hub 2.x no longer depends on it, so a clean
install of the course `requirements.txt` cannot import gradio at all and the app
will not start. The graded `requirements.txt` at the repository root is unchanged.
