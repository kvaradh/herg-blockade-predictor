"""
Shared, FROZEN evaluation protocol for Part 1 featurizer/model candidates.

Every candidate is judged by exactly the same folds, the same thresholds and the
same timing procedure, so numbers from different experiments are comparable.
Nothing in here is submission code -- it is the measuring instrument.

A candidate is any .py file exposing the two template TODOs:
    featurize(smiles) -> 2-D float array, one row per molecule
    build_model()     -> unfitted sklearn estimator / Pipeline

Run:  .venv/bin/python harness.py experiments/<candidate>.py
"""

import importlib.util, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).parent
SEED, N_SPLITS = 0, 5
SENS_FLOOR = 0.70          # the competition constraint
TIMING_N = 2000            # molecules used for throughput measurement

# The six molecules TestModel.py will feed your featurizer.
SAMPLES = ["CCO",
           "CC(=O)Oc1ccccc1C(=O)O",
           "CN(C)CCOC(c1ccccc1)c1ccccc1.Cl",
           "C[NH2+]CC[C@@H](Oc1ccc(cc1)C(F)(F)F)c1ccccc1",
           "CC(C)(C)c1ccc(cc1)C(O)CCCN1CCC(CC1)C(O)(c1ccccc1)c1ccccc1",
           "C" + "OCC" * 64 + "O"]


def load_data():
    df = pd.read_csv(HERE / "data/hw1data.tab", sep="\t")
    df.columns = ["Drug_ID", "Drug", "Y"]
    return df


def scaffold_groups(smiles):
    """Bemis-Murcko scaffold id per molecule. Cached: parsing 10k molecules is slow.

    Molecules sharing a scaffold must never be split across folds -- that is what
    forces the model to generalise to unseen chemistry, the way the grader will.
    """
    cache = HERE / "data/scaffolds.csv"
    if cache.exists():
        return pd.read_csv(cache)["group"].to_numpy()
    scaffolds = []
    for smile in smiles:
        try:
            scaffolds.append(MurckoScaffold.MurckoScaffoldSmiles(smiles=smile, includeChirality=False))
        except Exception:
            scaffolds.append("")           # unparseable -> its own bucket below
    groups = pd.factorize(pd.Series(scaffolds))[0]
    pd.DataFrame({"group": groups}).to_csv(cache, index=False)
    return groups


def sens_spec_curve(y, probability):
    """Sensitivity and specificity across a fine grid of thresholds."""
    grid = np.round(np.arange(0.01, 1.00, 0.005), 3)
    positives, negatives = y == 1, y == 0
    sensitivity = np.array([(probability[positives] >= t).mean() for t in grid])
    specificity = np.array([(probability[negatives] < t).mean() for t in grid])
    return grid, sensitivity, specificity


def pick_threshold(grid, sensitivity, specificity, floor):
    """Highest threshold whose sensitivity still clears `floor` -> max specificity."""
    ok = sensitivity >= floor
    if not ok.any():
        return None
    index = np.max(np.flatnonzero(ok))
    return {"threshold": float(grid[index]),
            "sensitivity": float(sensitivity[index]),
            "specificity": float(specificity[index])}


def out_of_fold(featurize, build_model, X, y, splitter, groups=None):
    probability = np.zeros(len(y))
    for train, test in splitter.split(X, y, groups):
        model = build_model().fit(X[train], y[train])
        probability[test] = model.predict_proba(X[test])[:, 1]
    return probability


def check_contract(featurize):
    """The things TestModel.py will fail you on."""
    batch = np.asarray(featurize(SAMPLES), float)
    singly = np.vstack([np.asarray(featurize([s]), float) for s in SAMPLES])
    return {"two_d_one_row_each": bool(batch.ndim == 2 and len(batch) == len(SAMPLES)),
            "all_finite": bool(np.isfinite(batch).all()),
            "stateless": bool(batch.shape == singly.shape and np.allclose(batch, singly, atol=1e-4)),
            "n_features": int(batch.shape[1]) if batch.ndim == 2 else -1,
            "handles_194_char": bool(np.isfinite(np.asarray(featurize([SAMPLES[-1]]), float)).all())}


def time_featurize(featurize, smiles):
    """Molecules per second, best of 3 -- the dominant term in the speed competitions."""
    best = min(_timed(featurize, smiles) for _ in range(3))
    return len(smiles) / best


def _timed(featurize, smiles):
    start = time.perf_counter()
    featurize(smiles)
    return time.perf_counter() - start


def evaluate(path):
    name = Path(path).stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    featurize, build_model = module.featurize, module.build_model

    contract = check_contract(featurize)

    df = load_data()
    smiles, y = df["Drug"].tolist(), df["Y"].to_numpy()
    groups = scaffold_groups(smiles)
    X = np.asarray(featurize(smiles), np.float32)

    # --- scaffold split: the honest estimate, and what the threshold is tuned on
    scaffold_cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    scaffold_probability = out_of_fold(featurize, build_model, X, y, scaffold_cv, groups)
    grid, sensitivity, specificity = sens_spec_curve(y, scaffold_probability)
    at_floor = pick_threshold(grid, sensitivity, specificity, SENS_FLOOR)

    # --- random split: only to measure how optimistic it is
    random_cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    random_probability = out_of_fold(featurize, build_model, X, y, random_cv, None)

    result = {
        "name": name,
        "contract": contract,
        "scaffold_auroc": float(roc_auc_score(y, scaffold_probability)),
        "random_auroc": float(roc_auc_score(y, random_probability)),
        "scaffold_prauc": float(average_precision_score(y, scaffold_probability)),
        "spec_at_sens_070": at_floor,
        "spec_at_sens_076": pick_threshold(grid, sensitivity, specificity, 0.76),
        "featurize_mols_per_sec": time_featurize(featurize, smiles[:TIMING_N]),
    }
    # how much optimism the random split buys -- this gap sizes your threshold margin
    if at_floor:
        t = at_floor["threshold"]
        result["sens_gap_random_minus_scaffold"] = float(
            (random_probability[y == 1] >= t).mean() - at_floor["sensitivity"])

    result["auroc_gap_random_minus_scaffold"] = result["random_auroc"] - result["scaffold_auroc"]
    (HERE / "results" / (name + ".json")).write_text(json.dumps(result, indent=2))
    return result


def show(r):
    print("\n=== %s ===" % r["name"])
    c = r["contract"]
    print("contract      2D/rows %s  finite %s  STATELESS %s  194-char %s  (%d features)"
          % (c["two_d_one_row_each"], c["all_finite"], c["stateless"],
             c["handles_194_char"], c["n_features"]))
    print("AUROC         scaffold %.4f   random %.4f   (optimism +%.4f)"
          % (r["scaffold_auroc"], r["random_auroc"], r["auroc_gap_random_minus_scaffold"]))
    for key, label in [("spec_at_sens_070", "sens>=0.70"), ("spec_at_sens_076", "sens>=0.76")]:
        v = r[key]
        print("%-13s threshold %.3f -> sensitivity %.4f  SPECIFICITY %.4f"
              % (label, v["threshold"], v["sensitivity"], v["specificity"]) if v
              else "%-13s unreachable" % label)
    if "sens_gap_random_minus_scaffold" in r:
        print("shift penalty sensitivity drops %.4f going random -> scaffold"
              % r["sens_gap_random_minus_scaffold"])
    print("speed         %.0f molecules/sec through featurize" % r["featurize_mols_per_sec"])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for argument in sys.argv[1:]:
        show(evaluate(argument))
