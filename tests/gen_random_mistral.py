"""t9: 生成随机权重的 Mistral 结构小模型（sliding window 测试用）。
结构 = llama + config.sliding_window；权重与 qwen2 同名（q/k/v/o + qk 无 norm 无 bias）。
用法: python gen_random_mistral.py <out_dir>"""
import json, os, sys
import numpy as np

out = sys.argv[1] if len(sys.argv) > 1 else "models/mistral-rand"

def _args(argv, default_out):
    """G19: --out/-o 与 --tokenizer/-t 可传（CI 用仓内 tokenizer），位置参数仍兼容。"""
    out, tok, pos = default_out, "models/qwen3-06", []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--out", "-o") and i + 1 < len(argv):
            out = argv[i + 1]; i += 2; continue
        if a in ("--tokenizer", "-t") and i + 1 < len(argv):
            tok = argv[i + 1]; i += 2; continue
        pos.append(a); i += 1
    if pos:
        out = pos[0]
    return out, tok, pos

out, TOKDIR, _pos = _args(sys.argv[1:], "models/mistral-rand")
L, H, KVH, E, I, V, SW = 2, 4, 2, 64, 128, 151936, 64
D = E // H
rng = np.random.default_rng(42)
os.makedirs(out, exist_ok=True)

cfg = {
    "architectures": ["MistralForCausalLM"], "model_type": "mistral",
    "hidden_size": E, "intermediate_size": I, "num_hidden_layers": L,
    "num_attention_heads": H, "num_key_value_heads": KVH,
    "head_dim": D, "vocab_size": V, "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0, "sliding_window": SW, "max_position_embeddings": 512,
    "attention_bias": False, "mlp_bias": False, "tie_word_embeddings": False,
}
json.dump(cfg, open(f"{out}/config.json", "w"), indent=1)

W = {"model.embed_tokens.weight": rng.normal(0, 0.02, (V, E)).astype(np.float32),
     "model.norm.weight": np.ones(E, dtype=np.float32),
     "lm_head.weight": rng.normal(0, 0.02, (V, E)).astype(np.float32)}
for l in range(L):
    p = f"model.layers.{l}"
    W[f"{p}.input_layernorm.weight"] = np.ones(E, dtype=np.float32)
    W[f"{p}.post_attention_layernorm.weight"] = np.ones(E, dtype=np.float32)
    W[f"{p}.self_attn.q_proj.weight"] = rng.normal(0, 0.02, (H * D, E)).astype(np.float32)
    W[f"{p}.self_attn.k_proj.weight"] = rng.normal(0, 0.02, (KVH * D, E)).astype(np.float32)
    W[f"{p}.self_attn.v_proj.weight"] = rng.normal(0, 0.02, (KVH * D, E)).astype(np.float32)
    W[f"{p}.self_attn.o_proj.weight"] = rng.normal(0, 0.02, (E, H * D)).astype(np.float32)
    W[f"{p}.mlp.gate_proj.weight"] = rng.normal(0, 0.02, (I, E)).astype(np.float32)
    W[f"{p}.mlp.up_proj.weight"] = rng.normal(0, 0.02, (I, E)).astype(np.float32)
    W[f"{p}.mlp.down_proj.weight"] = rng.normal(0, 0.02, (E, I)).astype(np.float32)

import struct
def save_safetensors(path, tensors):
    hdr = {}
    off = 0
    for k, v in tensors.items():
        b = v.tobytes()
        hdr[k] = {"dtype": "F32", "shape": list(v.shape), "data_offsets": [off, off + len(b)]}
        off += len(b)
    hb = json.dumps(hdr).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb))); f.write(hb)
        for v in tensors.values():
            f.write(v.tobytes())

save_safetensors(f"{out}/model.safetensors", W)
import shutil
src_tok = f"{TOKDIR}/tokenizer.json"
if os.path.exists(src_tok):
    shutil.copy(src_tok, f"{out}/tokenizer.json")
    for f2 in ("vocab.json", "merges.txt", "tokenizer_config.json"):
        p2 = f"{TOKDIR}/{f2}"
        if os.path.exists(p2):
            shutil.copy(p2, f"{out}/{f2}")
else:
    print(f"WARN: no tokenizer at {src_tok} (--tokenizer <dir> to point at one)")
print(f"saved random mistral -> {out} (L={L} H={H} KVH={KVH} E={E} window={SW})")
