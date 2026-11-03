"""Granularity / calibration / ONNX measurements for model-choice experiments.

Reuses harness.py's FROZEN folds and metrics by importing it -- nothing copied,
nothing modified.

    .venv/bin/python tools_model_analysis.py experiments/model_a_lr_c1.py [...]
    .venv/bin/python tools_model_analysis.py --onnx experiments/model_a_lr_c1.py [...]
"""

import importlib.util, json, sys, time
from pathlib import Path

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

import harness

HERE = Path(__file__).parent
OUT = HERE / "results"


def load_candidate(path):
    name = Path(path).stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return name, module.featurize, module.build_model


def scaffold_oof(featurize, build_model):
    df = harness.load_data()
    smiles, y = df["Drug"].tolist(), df["Y"].to_numpy()
    groups = harness.scaffold_groups(smiles)
    X = np.asarray(featurize(smiles), np.float32)
    cv = StratifiedGroupKFold(n_splits=harness.N_SPLITS, shuffle=True,
                              random_state=harness.SEED)
    probability = harness.out_of_fold(featurize, build_model, X, y, cv, groups)
    return X, y, probability


def analyse(path):
    name, featurize, build_model = load_candidate(path)
    X, y, p = scaffold_oof(featurize, build_model)

    grid, sens, spec = harness.sens_spec_curve(y, p)
    at_floor = harness.pick_threshold(grid, sens, spec, harness.SENS_FLOOR)

    distinct = int(np.unique(p).size)
    # how coarse the probabilities are right where the threshold has to sit
    if at_floor:
        t = at_floor["threshold"]
        near = p[(p > t - 0.02) & (p < t + 0.02)]
        distinct_near = int(np.unique(near).size)
    else:
        distinct_near = -1

    brier = float(brier_score_loss(y, p))
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    # mean |observed - predicted| over the 10 quantile bins = calibration error
    cal_mae = float(np.mean(np.abs(frac_pos - mean_pred)))

    result = {
        "name": name,
        "scaffold_auroc": float(roc_auc_score(y, p)),
        "spec_at_sens_070": at_floor,
        "distinct_probabilities": distinct,
        "distinct_within_0.02_of_threshold": distinct_near,
        "brier": brier,
        "calibration_mae_10q": cal_mae,
        "calibration_curve": {"mean_predicted": mean_pred.tolist(),
                              "fraction_positive": frac_pos.tolist()},
        "prob_min": float(p.min()), "prob_max": float(p.max()),
    }
    (OUT / (name + ".analysis.json")).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "calibration_curve"}))
    return result


def onnx_check(path):
    """Export, verify agreement with sklearn, record size and throughput."""
    from skl2onnx import to_onnx
    import onnxruntime as ort

    name, featurize, build_model = load_candidate(path)
    df = harness.load_data()
    smiles, y = df["Drug"].tolist(), df["Y"].to_numpy()
    X = np.asarray(featurize(smiles), np.float32)

    model = build_model().fit(X, y)
    reference = model.predict_proba(X[:2000])[:, 1]

    record = {"name": name, "n_features": int(X.shape[1])}
    try:
        onx = to_onnx(model, X[:1].astype(np.float32),
                      options={"zipmap": False}, target_opset=12)
        blob = onx.SerializeToString()
        out = OUT / (name + ".onnx")
        out.write_bytes(blob)
        record["onnx_mb"] = len(blob) / 1e6

        session = ort.InferenceSession(blob, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        batch = X[:2000].astype(np.float32)
        got = session.run(None, {input_name: batch})[1]
        got = np.asarray(got)
        got = got[:, 1] if got.ndim == 2 else got
        record["max_abs_prob_diff"] = float(np.max(np.abs(got - reference)))
        record["matches_1e-4"] = bool(record["max_abs_prob_diff"] < 1e-4)

        session.run(None, {input_name: batch})          # warm up
        best = min(_time_run(session, input_name, batch) for _ in range(3))
        record["onnx_mols_per_sec"] = len(batch) / best
        record["ok"] = True
    except Exception as exc:
        record["ok"] = False
        record["error"] = "%s: %s" % (type(exc).__name__, str(exc)[:400])

    (OUT / (name + ".onnx.json")).write_text(json.dumps(record, indent=2))
    print(json.dumps(record))
    return record


def _time_run(session, input_name, batch):
    start = time.perf_counter()
    session.run(None, {input_name: batch})
    return time.perf_counter() - start


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    do_onnx = "--onnx" in args
    for argument in [a for a in args if not a.startswith("--")]:
        (onnx_check if do_onnx else analyse)(argument)
