"""用 HF 加载真实 27B 的前 N 层（fp8 手动反量化），dump 第 0 层 linear_attn 输出，
与 sglang-c 的 [gpu linattn] 对比。只占几 GB 显存。
用法: hf27_partial.py <model_dir> [n_layers] [prompt]
"""
import glob
import json
import sys

import torch
from safetensors import safe_open
from transformers import AutoTokenizer, Qwen3_5TextConfig, Qwen3_5TextModel

mdir = sys.argv[1]
NLAYER = int(sys.argv[2]) if len(sys.argv) > 2 else 2
prompt = sys.argv[3] if len(sys.argv) > 3 else "The capital of France is"

cfg_raw = json.load(open(mdir + "/config.json"))
tcfg = dict(cfg_raw.get("text_config", cfg_raw))
tcfg["num_hidden_layers"] = NLAYER
tcfg["layer_types"] = tcfg["layer_types"][:NLAYER]
cfg = Qwen3_5TextConfig(**{k: v for k, v in tcfg.items() if not isinstance(v, dict)})
model = Qwen3_5TextModel(cfg).eval()
print("layers:", NLAYER, "types:", tcfg["layer_types"])

# 收集需要的张量：embed + layers.0..N-1 + final norm
want = {}
prefix = "model.language_model."
for f in sorted(glob.glob(mdir + "/*.safetensors")):
    with safe_open(f, framework="pt") as fh:
        for k in fh.keys():
            if not k.startswith(prefix):
                continue
            short = k[len(prefix):]
            if short.startswith("layers."):
                li = int(short.split(".")[1])
                if li >= NLAYER:
                    continue
            want[k] = f

# fp8 反量化后填入模型
def deq(f, key):
    with safe_open(f, framework="pt") as fh:
        t = fh.get_tensor(key)
        hdr = fh.get_slice(key).get_shape()
    if t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        sk = key.replace(".weight", ".weight_scale_inv")
        with safe_open(f, framework="pt") as fh:
            if sk in fh.keys():
                s = fh.get_tensor(sk).float()
                n, k = t.shape
                sn, skk = s.shape
                tt = t.to(torch.float32).reshape(sn, n // sn, skk, k // skk)
                tt = tt * s[:, None, :, None]
                return tt.reshape(n, k)
    return t.float()

sd = {}
for k, f in want.items():
    short = k[len(prefix):]
    sd[short] = deq(f, k)
missing, unexpected = model.load_state_dict(sd, strict=False)
print("missing:", len(missing), list(missing)[:3])
print("loaded tensors:", len(sd))

outs = []
def hook(mod, inp, out):
    h = out[0] if isinstance(out, tuple) else out
    v = h[0, -1, :2].float().detach().cpu().numpy()
    outs.append(v)

for l in model.layers:
    if hasattr(l, "linear_attn"):
        l.linear_attn.register_forward_hook(hook)

tok = AutoTokenizer.from_pretrained(mdir, trust_remote_code=True)
ids = tok(prompt, return_tensors="pt").input_ids
print("tokens:", ids.tolist()[0])
with torch.no_grad():
    o = model(ids, use_cache=False)
for i, v in enumerate(outs):
    print(f"[hf linattn L{i}] out=[{v[0]:.7g},{v[1]:.7g}]")