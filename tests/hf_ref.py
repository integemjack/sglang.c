"""用 HF transformers 作金标准：加载模型 → prefill → 导出 logits，供 sglang-c 对比。
用法: hf_ref.py <model_dir> <out_file> [prompt]
"""
import sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

mdir = sys.argv[1]
out = sys.argv[2]
prompt = sys.argv[3] if len(sys.argv) > 3 else "hello world"

tok = AutoTokenizer.from_pretrained(mdir, trust_remote_code=True)
ids = tok(prompt, return_tensors="pt").input_ids

# mini 模型的词表可能远小于 tokenizer 词表：把 id 折进范围（对比两个实现时
# 只关心“相同输入下输出一致”，id 具体是多少无所谓）
import json as _json
_cfg = _json.load(open(mdir + "/config.json"))
_V = _cfg.get("text_config", _cfg).get("vocab_size", 0)
if _V and int(ids.max()) >= _V:
    ids = ids % _V
    print(f"remapped token ids into vocab[{_V}]")
print("tokens:", ids.tolist())

model = AutoModelForCausalLM.from_pretrained(
    mdir, trust_remote_code=True, dtype=torch.float32, low_cpu_mem_usage=True
).eval()
print("model:", type(model).__name__)
_emb = model.get_input_embeddings().weight
print("embed shape", tuple(_emb.shape), "cfg vocab", model.config.vocab_size)
lim = int(_emb.shape[0])
if int(ids.max()) >= lim:
    ids = ids % lim
    print("remapped ids into", lim, "->", ids.tolist())

with torch.no_grad():
    o = model(ids)
lg = o.logits[0, -1].float().numpy()
print("logits shape", lg.shape, "top5", np.argsort(-lg)[:5].tolist())
print("logits[0:5]", lg[:5])
np.savetxt(out, lg)
print("wrote", out)

# 贪心生成 8 token 作对照
cur = ids.clone()
gen = []
with torch.no_grad():
    for _ in range(8):
        o = model(cur)
        nxt = int(o.logits[0, -1].argmax())
        gen.append(nxt)
        cur = torch.cat([cur, torch.tensor([[nxt]])], dim=1)
print("hf greedy 8:", gen)