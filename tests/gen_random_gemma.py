"""t9: 生成随机权重的 Mistral 结构小模型（sliding window 测试用）。
结构 = llama + config.sliding_window；权重与 qwen2 同名（q/k/v/o + qk 无 norm 无 bias）。
用法: python gen_random_mistral.py <out_dir>"""
import json, os, sys
import numpy as np

out = sys.argv[1] if len(sys.argv) > 1 else "models/gemma-rand"
L, H, KVH, E, I, V, SW = 2, 4, 2, 64, 128, 151936, 0
D = E // H
rng = np.random.default_rng(42)
os.makedirs(out, exist_ok=True)

cfg = {
    "architectures": ["GemmaForCausalLM"], "model_type": "gemma",
    "hidden_size": E, "intermediate_size": I, "num_hidden_layers": L,
    "num_attention_heads": H, "num_key_value_heads": KVH,
    "head_dim": D, "vocab_size": V, "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0, "max_position_embeddings": 512,
    "attention_bias": False, "mlp_bias": False, "tie_word_embeddings": True,
}
json.dump(cfg, open(f"{out}/config.json", "w"), indent=1)

W = {"model.embed_tokens.weight": rng.normal(0, 0.02, (V, E)).astype(np.float32),
     "model.norm.weight": np.ones(E, dtype=np.float32)}
# gemma: tied embeddings (lm_head == wte)
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
src_tok = "models/qwen3-06/tokenizer.json"
if os.path.exists(src_tok):
    shutil.copy(src_tok, f"{out}/tokenizer.json")
    for f2 in ("vocab.json", "merges.txt", "tokenizer_config.json"):
        p2 = f"models/qwen3-06/{f2}"
        if os.path.exists(p2):
            shutil.copy(p2, f"{out}/{f2}")
print(f"saved random gemma -> {out} (L={L} H={H} KVH={KVH} E={E})")
