"""Export a candidate to ONNX and verify the hard submission constraints:
size <= 50 MB, onnxruntime matches sklearn, and scoring throughput."""
import importlib.util, sys, time
from pathlib import Path
import numpy as np
import onnxruntime as ort
from skl2onnx import to_onnx
import harness

name = sys.argv[1]
module_path = Path("experiments") / (name + ".py")
spec = importlib.util.spec_from_file_location(name, module_path)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)

df = harness.load_data()
smiles, y = df["Drug"].tolist(), df["Y"].to_numpy()
X = np.asarray(module.featurize(smiles), np.float32)
model = module.build_model().fit(X, y)

onnx = to_onnx(model, X[:1], options={"zipmap": False}, target_opset=12)
out = Path("results") / (name + ".onnx")
out.write_bytes(onnx.SerializeToString())
size_mb = out.stat().st_size / 1e6

session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
inp = session.get_inputs()[0]
onnx_probability = np.asarray(session.run(None, {inp.name: X})[-1], float)[:, 1]
sklearn_probability = model.predict_proba(X)[:, 1]
max_difference = float(np.abs(onnx_probability - sklearn_probability).max())

sample = X[:5000]
best = min((lambda: (lambda s: (session.run(None, {inp.name: sample}), time.perf_counter() - s)[1])(time.perf_counter()))()
           for _ in range(3))
print("%-24s %8.2f MB   parity max-diff %.2e   onnx %9.0f rows/sec   %s"
      % (name, size_mb, max_difference, len(sample) / best,
         "OK" if size_mb <= 50 and max_difference <= 1e-4 else "*** CONSTRAINT FAILED ***"))
