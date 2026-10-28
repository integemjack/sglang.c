"""导出 HF 每层 hidden state（对比 sglang-c 用）。
用法: hf_layer_dump.py <model_dir> [prompt]
"""
import sys, json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

mdir = sys.argv[1]
prompt = sys.argv[2] if len(sys.argv) > 2 else "hello world"

tok = AutoTokenizer.from_pretrained(mdir, trust_remote_code=True)
ids = tok(prompt, return_tensors="pt").input_ids
cfg = json.load(open(mdir + "/config.json"))
V = cfg.get("text_config", cfg).get("vocab_size", 0)
model = AutoModelForCausalLM.from_pretrained(mdir, trust_remote_code=True,
                                            dtype=torch.float32, low_cpu_mem_usage=True).eval()
lim = int(model.get_input_embeddings().weight.shape[0])
if int(ids.max()) >= lim:
    ids = ids % lim
print("tokens:", ids.tolist())

outs = []
def hook(mod, inp, out):
    h = out[0] if isinstance(out, tuple) else out
    outs.append(h[0, -1].float().detach().cpu().numpy())

hs = model.model.layers if hasattr(model.model, "layers") else model.model.language_model.layers
for l in hs:
    l.register_forward_hook(hook)
with torch.no_grad():
    o = model(ids, use_cache=False)
for i, h in enumerate(outs):
    print(f"layer {i}: x=[{h[0]:.7g},{h[1]:.7g}]  (n={len(h)})")
lg = o.logits[0, -1].float().numpy()
print("logits[0:6]:", np.round(lg[:6], 6))