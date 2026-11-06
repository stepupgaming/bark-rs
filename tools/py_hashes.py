"""Prints `name sha256` for every tensor the official loader decodes (floats as f32 bytes).

    python -I tools/py_hashes.py <model dir> > py.txt
"""
import hashlib
import importlib.util
import os
import sys

import numpy as np

d = sys.argv[1]
spec = importlib.util.spec_from_file_location("official_bark", os.path.join(d, "bark_loader.py"))
bark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bark)
sd, _ = bark.load_state_dict(os.path.join(d, "model.bark"), as_numpy=True)
for k, v in sd.items():
    v = np.ascontiguousarray(v if v.dtype in (np.int64, np.int32) else v.astype(np.float32))
    print(k, hashlib.sha256(v.tobytes()).hexdigest())
