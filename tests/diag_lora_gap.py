"""diag_lora_gap.py - measure top1/top2 logit gap at each greedy step for
model+lora. If the gap at the divergence step is tiny (<1e-3), the CPU/GPU
mismatch is benign FP sensitivity, not a logic bug."""
import sys
import numpy as np

_ARGS = [sys.argv[0], sys.argv[1] if len(sys.argv) > 1 else "models/qwen05",
         sys.argv[2] if len(sys.argv) > 2 else "models/test-lora-q2"]
sys.argv = _ARGS[:2]
exec(open("sglang-c/tests/ref_llama.py", encoding="utf-8").read().split("# tokenizer")[0])
LORA = _ARGS[2]

from safetensors.numpy import load_file  # noqa: E402

# merge lora into W (same math as C: W += B@A * alpha/r)
import json as _json
cfg = _json.load(open(f"{LORA}/adapter_config.json"))
scale = cfg.get("lora_alpha", 16) / cfg.get("r", 8)
LA = load_file(f"{LORA}/adapter.safetensors")
n = 0
for k in list(LA.keys()):
    if ".lora_A" not in k:
        continue
    base = k[: -len(".lora_A.weight")] + ".weight"
    other = f"{k[: -len('.lora_A.weight')]}.lora_B.weight"
    if other not in LA or base not in W:
        continue
    A = LA[k].astype(np.float32)
    B = LA[other].astype(np.float32)
    W[base] = W[base] + (B @ A) * scale
    n += 1
print(f"merged {n} tensors, scale={scale}")

ids = [9707, 1879]  # 'Hello world'
for step in range(6):
    lg = forward(ids)
    order = np.argsort(-lg)
    t1, t2 = order[0], order[1]
    gap = lg[t1] - lg[t2]
    print(f"step {step}: top1={t1} top2={t2} gap={gap:.6g}")
    ids.append(int(t1))
print("greedy6:", ids[2:])
