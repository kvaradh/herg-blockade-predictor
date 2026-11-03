"""Serial, uncontended throughput measurement. Timing only -- no CV.

The earlier numbers were taken while four agents competed for CPU and are not
comparable. This runs one candidate at a time on an idle machine.
"""
import importlib.util, json, sys, time
from pathlib import Path
import numpy as np
import harness

CANDIDATES = ["ref_ecfp_lr", "model_b_et_500", "desc_desalt_hybrid", "desc_core13_fast",
              "desc_core15", "fp_atompair_2048_cnt", "fp_maccs",
              "str_bincount256", "str_bincount_density", "str_density", "str_counts_basic",
              "str_ngram1024", "str_ngram4096", "str_hashvec", "str_substr"]
N = 5000


def load(name):
    path = Path("experiments") / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _t(featurize, smiles):
    start = time.perf_counter()
    featurize(smiles)
    return time.perf_counter() - start


smiles = harness.load_data()["Drug"].tolist()[:N]
print("%-26s %12s %10s" % ("candidate", "mol/sec", "us/mol"))
out = {}
for name in CANDIDATES:
    path = Path("experiments") / (name + ".py")
    if not path.exists():
        print("%-26s %12s" % (name, "MISSING")); continue
    try:
        featurize = load(name).featurize
        featurize(smiles[:50])                      # warm up caches / lazy imports
        best = min(_t(featurize, smiles) for _ in range(5))
        rate = N / best
        out[name] = rate
        print("%-26s %12.0f %10.2f" % (name, rate, 1e6 / rate))
    except Exception as error:
        print("%-26s %12s  %s" % (name, "ERROR", error))

Path("results/timing_serial.json").write_text(json.dumps(out, indent=2))
