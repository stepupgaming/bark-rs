"""Turns full ONNX exports of Bark 3.0 into weightless graphs plus a manifest.

    python -I tools/make_graphs.py <model dir> <onnx dir> <out dir> [fp32|exact|q8|int8]

<onnx dir> holds encoder.onnx, decoder.onnx, joint.onnx and vocab.json as written by the
export (fp32, opset 17, TorchScript exporter). Every initializer that equals a tensor of
model.bark (as stored, or transposed) becomes an external initializer; the Rust crate builds
it from model.bark at load time and hands it to ONNX Runtime. What does not match (folded
BatchNorm, small constants) stays inside the graph.

manifest.json: {graph: {initializer: {"src": state-dict key, "op": ..., "shape": [...]}}}
  op "id": as stored; "t": [rows, rest] transposed; "lstm": rows regrouped from torch gate
  order (i, f, g, o) to ONNX (i, o, f, c); "lstm_b": regrouped src then regrouped "src2".

Packing (the last argument, default exact) keeps the encoder's big weights small in memory.
Every encoder MatMul and 1x1 Conv whose weight is a levels2 tensor (each row holds 0 and
+-K levels) becomes com.microsoft MatMulNBits, built from the stored codes, never from fp32:
  exact: one 2-bit ternary matrix per level, summed: W = sum_j lev_j * T_j. Bit-exact with
         fp32 weights, 2K bits a weight. ops "tern" (B, uint8) and "tern_s" (scales), "term" j.
  q8:    one 8-bit matrix, per-row scale lev_max / k with k <= 127 picked so every level
         lands closest to an integer. ops "q8" and "q8_s".
  int8:  q8 with accuracy_level 4: ONNX Runtime also quantizes the activations to int8, so
         ARM int8 dot-product kernels do the work. Fastest on phones, not exact.
fp32 leaves the graph as exported.
"""
import hashlib
import importlib.util
import json
import os
import shutil
import sys

import numpy as np

sys.path += [p for p in [os.environ.get("BARK_PYLIB")] if p]  # optional site dir for onnx
import onnx
from onnx import numpy_helper

model_dir, onnx_dir, out = sys.argv[1:4]
pack = sys.argv[4] if len(sys.argv) > 4 else "exact"
assert pack in ("fp32", "exact", "q8", "int8"), pack
BS = 128  # MatMulNBits block size
os.makedirs(out, exist_ok=True)
spec = importlib.util.spec_from_file_location("official_bark", os.path.join(model_dir, "bark_loader.py"))
bark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bark)
sd, _ = bark.load_state_dict(os.path.join(model_dir, "model.bark"), as_numpy=True)
with open(os.path.join(model_dir, "model.bark"), "rb") as f:
    header = json.loads(f.read(int.from_bytes(f.read(8), "little")))
kinds = {e["n"]: e for e in header["index"]}


def h(a):
    return hashlib.sha256(np.ascontiguousarray(a, np.float32).tobytes()).hexdigest()


table = {}
for k, v in sd.items():
    if v.dtype in (np.int64, np.int32) or v.size < 256:
        continue
    table.setdefault(h(v), (k, "id"))
    m = v.reshape(v.shape[0], -1)
    if m.shape[1] > 1:
        table.setdefault(h(m.T), (k, "t"))


def gates(a):
    """torch LSTM gate order (i, f, g, o) to ONNX (i, o, f, c)."""
    i, f, g, o = np.split(a, 4, axis=0)
    return np.concatenate([i, o, f, g], 0)


for k, v in sd.items():
    if ".lstm.weight_" in k:
        table.setdefault(h(gates(v)), (k, "lstm"))
    if ".lstm.bias_ih_" in k:
        k2 = k.replace("bias_ih", "bias_hh")
        table.setdefault(h(np.concatenate([gates(v), gates(sd[k2])])), (k, "lstm_b", k2))



def external(name, dtype, dims):
    t = onnx.TensorProto(name=name, data_type=dtype, dims=dims)
    t.data_location = onnx.TensorProto.EXTERNAL
    n = int(np.prod(dims)) * (1 if dtype == onnx.TensorProto.UINT8 else 4)
    for key, val in (("location", "bark.weights"), ("offset", "0"), ("length", str(n))):
        e = t.external_data.add()
        e.key, e.value = key, val
    return t


def pack_nodes(m, entries):
    """MatMul / 1x1 Conv on a levels2 weight -> MatMulNBits. Returns the packed weight count."""
    from onnx import helper
    nodes, done, count = [], set(), 0
    for nd in m.graph.node:
        w = nd.input[1] if len(nd.input) > 1 else None
        e = entries.get(w)
        src = e and kinds[e["src"]]
        if not (src and src["k"] == "levels2" and nd.op_type in ("MatMul", "Conv") and len(nd.input) == 2):
            nodes.append(nd)
            continue
        rows, cols = src["shape"][0], int(np.prod(src["shape"][1:]))
        if nd.op_type == "MatMul":
            assert e["op"] == "t", (w, e)
        else:
            a = {x.name: helper.get_attribute_value(x) for x in nd.attribute}
            assert a.get("kernel_shape") == [1] and a.get("group", 1) == 1, (w, a)
        nblk = -(-cols // BS)
        x, y, tag = nd.input[0], nd.output[0], w.replace("/", "_")
        if nd.op_type == "Conv":  # [1, C, T] -> [1, T, C] and back
            nodes.append(helper.make_node("Transpose", [x], [tag + "_xt"], perm=[0, 2, 1]))
            x, y = tag + "_xt", tag + "_yt"
        terms = range(src["K"]) if pack == "exact" else [None]
        outs = []
        for j in terms:
            bits = 2 if pack == "exact" else 8
            extra = {"accuracy_level": 4} if pack == "int8" else {}
            b, sc = f"{tag}_B{'' if j is None else j}", f"{tag}_S{'' if j is None else j}"
            op = "tern" if pack == "exact" else "q8"
            m.graph.initializer.append(external(b, onnx.TensorProto.UINT8, [rows, nblk, BS * bits // 8]))
            m.graph.initializer.append(external(sc, onnx.TensorProto.FLOAT, [rows, nblk]))
            entries[b] = {"src": e["src"], "op": op, "shape": [rows, nblk, BS * bits // 8], "cols": cols}
            entries[sc] = {"src": e["src"], "op": op + "_s", "shape": [rows, nblk], "cols": cols}
            if j is not None:
                entries[b]["term"] = entries[sc]["term"] = j
            o = y if len(terms) == 1 else f"{tag}_y{j}"
            nodes.append(helper.make_node("MatMulNBits", [x, b, sc], [o], domain="com.microsoft",
                                          K=cols, N=rows, bits=bits, block_size=BS, **extra))
            outs.append(o)
        if len(outs) > 1:
            nodes.append(helper.make_node("Sum", outs, [y]))
        if nd.op_type == "Conv":
            nodes.append(helper.make_node("Transpose", [y], [nd.output[0]], perm=[0, 2, 1]))
        done.add(w)
        count += rows * cols
    del m.graph.node[:]
    m.graph.node.extend(nodes)
    keep = [i for i in m.graph.initializer if i.name not in done]
    del m.graph.initializer[:]
    m.graph.initializer.extend(keep)
    for w in done:
        del entries[w]
    if not any(o.domain == "com.microsoft" for o in m.opset_import):
        m.opset_import.append(helper.make_opsetid("com.microsoft", 1))
    return count


manifest = {}
for g in ("encoder", "decoder", "joint"):
    m = onnx.load(os.path.join(onnx_dir, f"{g}.onnx"), load_external_data=True)
    entries, kept, moved = {}, 0, 0
    for init in m.graph.initializer:
        a = numpy_helper.to_array(init)
        hit = table.get(h(a)) if a.dtype == np.float32 and a.size >= 256 else None
        if hit is None:
            kept += a.nbytes
            continue
        entries[init.name] = {"src": hit[0], "op": hit[1], "shape": list(a.shape)}
        if len(hit) > 2:
            entries[init.name]["src2"] = hit[2]
        moved += a.nbytes
        init.ClearField("raw_data")
        init.ClearField("float_data")
        init.data_location = onnx.TensorProto.EXTERNAL
        del init.external_data[:]
        for key, val in (("location", "bark.weights"), ("offset", "0"), ("length", str(a.nbytes))):
            e = init.external_data.add()
            e.key, e.value = key, val
    if g == "encoder" and pack != "fp32":
        print(f"encoder: {pack_nodes(m, entries) / 1e6:.1f}M weights packed ({pack})")
    onnx.save(m, os.path.join(out, f"{g}.onnx"))
    manifest[g] = entries
    print(f"{g}: {len(entries)} from model.bark ({moved / 1e6:.1f} MB), {kept / 1e6:.2f} MB stays in the graph, "
          f"file {os.path.getsize(os.path.join(out, g + '.onnx')) / 1e6:.2f} MB")

json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"), indent=1)
shutil.copy(os.path.join(onnx_dir, "vocab.json"), os.path.join(out, "vocab.json"))
