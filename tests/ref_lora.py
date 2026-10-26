"""ref_lora.py - reference logits for base model + LoRA merge.
Merges B@A*scale into weights in numpy, runs the same forward as ref_llama.py,
saves ids/top5/logits5/greedy6 to tests/ref_lora.json.
"""
import json
import sys

import numpy as np

_ARGS = [sys.argv[0], sys.argv[1] if len(sys.argv) > 1 else "models/tiny-llama",
         sys.argv[2] if len(sys.argv) > 2 else "models/test-lora",
         sys.argv[3] if len(sys.argv) > 3 else "sglang-c/tests/ref_lora.json"]
sys.argv = _ARGS[:2]  # ref_llama.py top-level reads sys.argv[1] as MODEL only
exec(open("sglang-c/tests/ref_llama.py", encoding="utf-8").read().split("# tokenizer")[0])
LORA = _ARGS[2]
OUT = _ARGS[3]

acfg = json.load(open(f"{LORA}/adapter_config.json"))
R = acfg["r"]
ALPHA = acfg["lora_alpha"]
SCALE = ALPHA / R
import struct
with open(f"{LORA}/adapter.safetensors", "rb") as fh:
    hlen = struct.unpack("<Q", fh.read(8))[0]
    lhead = json.loads(fh.read(hlen))
    ldata = fh.read()
LT = {}
for k, meta in lhead.items():
    s, t = meta["data_offsets"]
    LT[k] = np.frombuffer(ldata[s:t], dtype=np.float32).reshape(meta["shape"])

merged = 0
for key in list(W.keys()):
    if ".lora_" in key or key not in W:
        continue
for l in range(L):
    for suf in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]:
        an = f"model.layers.{l}.{suf}.lora_A.weight"
        bn = f"model.layers.{l}.{suf}.lora_B.weight"
        if an in LT and bn in LT:
            W[f"model.layers.{l}.{suf}.weight"] = f32(
                W[f"model.layers.{l}.{suf}.weight"]
                + LT[bn] @ LT[an] * SCALE)
            merged += 1
print(f"merged {merged} projections, scale={SCALE}")

ids = [1, 15043, 3186]  # Hello world (llama bos + tokens, same as HF)
lg = forward(ids)
top5 = np.argsort(lg)[::-1][:5].tolist()

greedy6 = []
cur = int(np.argmax(lg))
for s in range(6):
    greedy6.append(cur)
    ids = ids + [cur]
    lg = forward(ids)
    cur = int(np.argmax(lg))

out = {"ids": ids, "top5": top5,
       "logits5": [float(lg[i]) for i in top5], "greedy6": greedy6}
json.dump(out, open(OUT, "w"), indent=1)
print(f"saved {OUT}: top5={top5} greedy6={greedy6}")
