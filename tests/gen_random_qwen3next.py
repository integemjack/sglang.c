"""G9: 生成随机权重的 Qwen3-Next 结构小模型（layer_types 混合层：linear_attention + full_attention）。
结构：L=4，前 3 层 linear_attention（gated deltanet，权重名对齐 HF Qwen3Next 检查点：
linear_attn.in_proj_qkvz / in_proj_ba / conv1d / norm / out_proj / A_log / dt_bias），
第 4 层 full_attention（qwen3 风格 q/k/v/o + qk-norm）。
in_proj_qkvz 按 k-head 组交错布局 [q kd | k kd | v ratio*vd | z ratio*vd]（同 HF
fix_query_key_value_ordering）；in_proj_ba 每组 [b ratio | a ratio]。
用法: python gen_random_qwen3next.py <out_dir>"""
import json, os, sys
import numpy as np

out = sys.argv[1] if len(sys.argv) > 1 else "models/qwen3next-rand"
L, H, KVH, E, I, V = 4, 4, 2, 64, 128, 151936
D = E // H
NK, NH, KD, VD, CONV = 2, 4, 16, 16, 4  # linear attn：k 头 2、v 头 4、头维 16、conv 核 4
RATIO = NH // NK
KEYD, VALD, CONVD = NK * KD, NH * VD, 2 * NK * KD + NH * VD
rng = np.random.default_rng(7)
os.makedirs(out, exist_ok=True)

cfg = {
    "architectures": ["Qwen3NextForCausalLM"], "model_type": "qwen3_next",
    "hidden_size": E, "intermediate_size": I, "num_hidden_layers": L,
    "num_attention_heads": H, "num_key_value_heads": KVH,
    "head_dim": D, "vocab_size": V, "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0, "max_position_embeddings": 512,
    "tie_word_embeddings": True,
    "layer_types": ["linear_attention", "linear_attention",
                    "linear_attention", "full_attention"],
    "linear_num_key_heads": NK, "linear_num_value_heads": NH,
    "linear_key_head_dim": KD, "linear_value_head_dim": VD,
    "linear_conv_kernel_dim": CONV,
}
json.dump(cfg, open(f"{out}/config.json", "w"), indent=1)

W = {"model.embed_tokens.weight": rng.normal(0, 0.02, (V, E)).astype(np.float32),
     "model.norm.weight": np.ones(E, dtype=np.float32)}
for l in range(L):
    p = f"model.layers.{l}"
    W[f"{p}.input_layernorm.weight"] = np.ones(E, dtype=np.float32)
    W[f"{p}.post_attention_layernorm.weight"] = np.ones(E, dtype=np.float32)
    if cfg["layer_types"][l] == "linear_attention":
        W[f"{p}.linear_attn.in_proj_qkvz.weight"] = rng.normal(0, 0.02, (2 * KEYD + 2 * VALD, E)).astype(np.float32)
        W[f"{p}.linear_attn.in_proj_ba.weight"] = rng.normal(0, 0.02, (2 * NH, E)).astype(np.float32)
        W[f"{p}.linear_attn.conv1d.weight"] = rng.normal(0, 0.02, (CONVD, 1, CONV)).astype(np.float32)
        W[f"{p}.linear_attn.norm.weight"] = np.ones(VD, dtype=np.float32)
        W[f"{p}.linear_attn.out_proj.weight"] = rng.normal(0, 0.02, (E, VALD)).astype(np.float32)
        # 与 HF 初始化一致：A ~ U(0.01,16) 取 log；dt_bias=1
        W[f"{p}.linear_attn.A_log"] = np.log(rng.uniform(0.01, 16, NH)).astype(np.float32)
        W[f"{p}.linear_attn.dt_bias"] = np.ones(NH, dtype=np.float32)
    else:
        W[f"{p}.self_attn.q_proj.weight"] = rng.normal(0, 0.02, (H * D, E)).astype(np.float32)
        W[f"{p}.self_attn.k_proj.weight"] = rng.normal(0, 0.02, (KVH * D, E)).astype(np.float32)
        W[f"{p}.self_attn.v_proj.weight"] = rng.normal(0, 0.02, (KVH * D, E)).astype(np.float32)
        W[f"{p}.self_attn.o_proj.weight"] = rng.normal(0, 0.02, (E, H * D)).astype(np.float32)
        W[f"{p}.self_attn.q_norm.weight"] = np.ones(D, dtype=np.float32)
        W[f"{p}.self_attn.k_norm.weight"] = np.ones(D, dtype=np.float32)
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
for f2 in ("tokenizer.json", "vocab.json", "merges.txt", "tokenizer_config.json"):
    p2 = f"models/qwen3-06/{f2}"
    if os.path.exists(p2):
        shutil.copy(p2, f"{out}/{f2}")
print(f"saved random qwen3_next -> {out} (L={L} E={E} lin: nk={NK} nh={NH} kd={KD} vd={VD} conv={CONV})")
