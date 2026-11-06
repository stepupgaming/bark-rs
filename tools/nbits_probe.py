"""Checks ONNX Runtime's MatMulNBits on a real Bark weight: exact split into two 2-bit
ternary matrices versus 8-bit and fp32. Prints max error against fp32 and the time per call.

    python -I tools/nbits_probe.py <model dir>
"""
import importlib.util
import os
import sys
import time

import numpy as np

sys.path += [p for p in [os.environ.get("BARK_PYLIB")] if p]  # optional site dir for onnx
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

d = sys.argv[1]
spec = importlib.util.spec_from_file_location("official_bark", os.path.join(d, "bark_loader.py"))
bark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bark)
sd, _ = bark.load_state_dict(os.path.join(d, "model.bark"), as_numpy=True)
KEY = "encoder.layers.0.feed_forward1.linear1.weight"
W = sd[KEY].astype(np.float32)  # [N, K]
N, K = W.shape
BS = 128


def pack(q, bits):
    """q: uint values [N, K] -> MatMulNBits B [N, K/BS, BS*bits/8], low bits first."""
    per = 8 // bits
    q = q.reshape(N, K // BS, BS // per, per).astype(np.uint8)
    out = np.zeros(q.shape[:3], np.uint8)
    for i in range(per):
        out |= q[..., i] << (bits * i)
    return out


def nbits(name, q, scale, bits, inp, out):
    b = numpy_helper.from_array(pack(q, bits), name + "_B")
    s = numpy_helper.from_array(np.repeat(scale[:, None], K // BS, 1).astype(np.float32), name + "_S")
    node = helper.make_node("MatMulNBits", [inp, name + "_B", name + "_S"], [out], domain="com.microsoft",
                            K=K, N=N, bits=bits, block_size=BS)
    return node, [b, s]


def levels_split(W):
    """Rows of {0, +-a, +-b}: W = a * Ta + b * Tb with Ta, Tb in {-1, 0, 1}."""
    mags = [np.unique(np.abs(r[r != 0])) for r in W]
    k = max(len(m) for m in mags)
    terms = []
    for j in range(k):
        lev = np.array([m[j] if j < len(m) else 0 for m in mags], np.float32)
        t = np.where(np.isclose(np.abs(W), lev[:, None]) & (lev[:, None] > 0), np.sign(W), 0).astype(np.int8)
        terms.append((lev, t))
    rec = sum(l[:, None] * t for l, t in terms)
    assert np.array_equal(rec, W), "split is not exact"
    return terms


def model(nodes, inits):
    g = helper.make_graph(nodes, "m", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, "T", K])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, "T", N])], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)], ir_version=8)
    return ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])


x = np.random.default_rng(0).standard_normal((1, 841, K)).astype(np.float32)
ref = x @ W.T
terms = levels_split(W)
print(KEY, W.shape, "levels per row:", len(terms))

cases = {}
cases["fp32"] = model([helper.make_node("MatMul", ["x", "W"], ["y"])], [numpy_helper.from_array(W.T.copy(), "W")])
nodes, inits, outs = [], [], []
for j, (lev, t) in enumerate(terms):
    n, i = nbits(f"t{j}", t + 2, lev, 2, "x", f"y{j}")
    nodes.append(n); inits += i; outs.append(f"y{j}")
nodes.append(helper.make_node("Sum", outs, ["y"]))
cases["2bit x%d (exact)" % len(terms)] = model(nodes, inits)
s8 = np.abs(W).max(1) / 127
q8 = np.clip(np.round(W / s8[:, None]), -127, 127).astype(np.int32) + 128
cases["8bit per row"] = model(*[[a] if k == 0 else b for k, (a, b) in enumerate([nbits("q8", q8, s8, 8, "x", "y")] * 2)])
s4 = np.abs(W).max(1) / 7
q4 = np.clip(np.round(W / s4[:, None]), -7, 7).astype(np.int32) + 8
cases["4bit per row"] = model(*[[a] if k == 0 else b for k, (a, b) in enumerate([nbits("q4", q4, s4, 4, "x", "y")] * 2)])

for name, s in cases.items():
    try:
        y = s.run(None, {"x": x})[0]
        t0 = time.perf_counter()
        for _ in range(5):
            s.run(None, {"x": x})
        dt = (time.perf_counter() - t0) / 5
        err = np.abs(y - ref).max() / np.abs(ref).max()
        print(f"{name:20s} rel max err {err:.2e}  {dt * 1000:7.1f} ms")
    except Exception as e:
        print(f"{name:20s} FAILED {str(e)[:200]}")
