"""dump HF 的 linear_attn 模块输出（out_proj 后），与 sglang-c 的 h 行对比。"""
import sys, json
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

mdir = sys.argv[1]
prompt = sys.argv[2] if len(sys.argv) > 2 else "hello world"
tok = AutoTokenizer.from_pretrained(mdir, trust_remote_code=True)
ids = tok(prompt, return_tensors="pt").input_ids
model = AutoModelForCausalLM.from_pretrained(mdir, trust_remote_code=True,
                                            dtype=torch.float32, low_cpu_mem_usage=True).eval()
lim = int(model.get_input_embeddings().weight.shape[0])
if int(ids.max()) >= lim:
    ids = ids % lim
print("tokens:", ids.tolist())

def hook(mod, inp, out):
    h = out[0] if isinstance(out, tuple) else out
    v = h[0, -1, :2].float().detach().cpu().numpy()
    print(f"[hf linattn] out=[{v[0]:.7g},{v[1]:.7g}]")

hs = model.model.layers if hasattr(model.model, "layers") else model.model.language_model.layers
for i, l in enumerate(hs):
    if hasattr(l, "linear_attn"):
        l.linear_attn.register_forward_hook(hook)
with torch.no_grad():
    o = model(ids, use_cache=False)