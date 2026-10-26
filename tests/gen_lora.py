"""gen_lora.py - build a random PEFT-format LoRA adapter for testing.
usage: python gen_lora.py <model_dir> <out_dir> [rank=8] [alpha=16] [--seed N]
Targets q/k/v/o/gate/up/down projections of llama-family models; scale=alpha/rank.
--seed controls the adapter randomness (default 42: keeps the historical
models/test-lora adapter bit-identical across regenerations).
"""
import json
import os
import struct
import sys

import numpy as np

_args = sys.argv[1:]
SEED = 42
if "--seed" in _args:
    i = _args.index("--seed")
    SEED = int(_args[i + 1])
    _args = _args[:i] + _args[i + 2:]
MODEL, OUT = _args[0], _args[1]
RANK = int(_args[2]) if len(_args) > 2 else 8
ALPHA = float(_args[3]) if len(_args) > 3 else 16.0

cfg = json.load(open(f"{MODEL}/config.json"))
L = cfg["num_hidden_layers"]
E = cfg["hidden_size"]
I = cfg.get("intermediate_size", 4 * E)
KVH = cfg.get("num_key_value_heads", cfg["num_attention_heads"])
H = cfg["num_attention_heads"]
D = cfg.get("head_dim", E // H)
KD = D * KVH

rng = np.random.default_rng(SEED)
tensors = {}
shapes = {
    "self_attn.q_proj": (E, E),
    "self_attn.k_proj": (KD, E),
    "self_attn.v_proj": (KD, E),
    "self_attn.o_proj": (E, E),
    "mlp.gate_proj": (I, E),
    "mlp.up_proj": (I, E),
    "mlp.down_proj": (E, I),
}
for l in range(L):
    for name, (out_f, in_f) in shapes.items():
        tensors[f"model.layers.{l}.{name}.lora_A.weight"] = (
            rng.standard_normal((RANK, in_f)) * 0.02
        ).astype(np.float32)
        tensors[f"model.layers.{l}.{name}.lora_B.weight"] = (
            rng.standard_normal((out_f, RANK)) * 0.02
        ).astype(np.float32)

header = {}
off = 0
blob = b""
for k, v in tensors.items():
    b = v.tobytes()
    header[k] = {"dtype": "F32", "shape": list(v.shape), "data_offsets": [off, off + len(b)]}
    blob += b
    off += len(b)

os.makedirs(OUT, exist_ok=True)
with open(f"{OUT}/adapter.safetensors", "wb") as f:
    hb = json.dumps(header).encode()
    f.write(struct.pack("<Q", len(hb)) + hb + blob)
json.dump({"peft_type": "LORA", "r": RANK, "lora_alpha": ALPHA,
           "target_modules": list(shapes.keys())},
          open(f"{OUT}/adapter_config.json", "w"), indent=1)
print(f"saved {OUT}/adapter.safetensors: {len(tensors)} tensors, rank={RANK} alpha={ALPHA} seed={SEED}")
