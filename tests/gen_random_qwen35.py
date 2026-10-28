"""生成迷你 qwen3_5 模型（dense 版，对齐 Qwen3.8-27B 的结构，尺寸缩小）。

与 gen_random_qwen3next.py 的差异：
- model_type=qwen3_5 + text_config 嵌套（VLM 包装，走 model.language_model.* 前缀）
- 无 MoE：每层都是标准 SwiGLU MLP
- attn_output_gate=true（q_proj 输出 2*H*D，逐头交错 [q|gate]）
- layer_types 周期 4（3 个 linear_attention + 1 个 full_attention）
"""
import json, os, struct, sys
import numpy as np

out = sys.argv[1] if len(sys.argv) > 1 else "models/qwen35-mini"
TOKDIR = sys.argv[2] if len(sys.argv) > 2 else "models/smollm2-135m"

L, H, KVH, E, I, V = 8, 4, 1, 256, 512, 1024
D = 64                      # = E/H? 不：显式 head_dim（真实模型 D=256≠E/H=213）
NK, NH, KD, VD, CONV = 2, 4, 32, 32, 4
import os as _os
PARTIAL = float(_os.environ.get("SGL_GEN_PARTIAL", "0.25"))
QGATE = _os.environ.get("SGL_GEN_QGATE", "1") == "1"
EPS = 1e-6
ROPE_THETA = 10000000.0
# 调试用：SGL_GEN_L / SGL_GEN_KINDS 可只生成单层单类型模型做隔离对比
import os as _os
if _os.environ.get("SGL_GEN_L"):
    L = int(_os.environ["SGL_GEN_L"])

RATIO = NH // NK
KEYD, VALD = NK * KD, NH * VD
ROW_N = 2 * KEYD + 2 * VALD
CONVD = 2 * KEYD + NH * VD
if _os.environ.get("SGL_GEN_KINDS"):
    LAYER_TYPES = _os.environ["SGL_GEN_KINDS"].split(",")
    L = len(LAYER_TYPES)
else:
    LAYER_TYPES = ["linear_attention" if (i + 1) % 4 else "full_attention" for i in range(L)]

rng = np.random.default_rng(11)
os.makedirs(out, exist_ok=True)

text_cfg = {
    "hidden_size": E, "intermediate_size": I, "num_hidden_layers": L,
    "num_attention_heads": H, "num_key_value_heads": KVH, "head_dim": D,
    "vocab_size": V, "rms_norm_eps": EPS, "hidden_act": "silu",
    "max_position_embeddings": 512,
    "rope_parameters": {"rope_theta": ROPE_THETA, "rope_type": "default",
                        "partial_rotary_factor": PARTIAL,
                        "mrope_interleaved": True, "mrope_section": [11, 11, 10]},
    "attention_bias": False, "attention_dropout": 0.0,
    "attn_output_gate": QGATE, "output_gate_type": "swish",
    "tie_word_embeddings": False, "use_cache": True,
    "layer_types": LAYER_TYPES, "full_attention_interval": 4,
    "linear_num_key_heads": NK, "linear_num_value_heads": NH,
    "linear_key_head_dim": KD, "linear_value_head_dim": VD,
    "linear_conv_kernel_dim": CONV,
    "bos_token_id": 1, "eos_token_id": 2,
}
cfg = {"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
       "text_config": text_cfg}
json.dump(cfg, open(f"{out}/config.json", "w"), indent=1)

PFX = "model.language_model."
def norm_w(n):   # qwen3_5 RMSNorm：(1+w) 语义
    return rng.normal(0, 0.3, n).astype(np.float32)

W = {PFX + "embed_tokens.weight": rng.normal(0, 0.02, (V, E)).astype(np.float32),
     PFX + "norm.weight": norm_w(E),
     PFX + "lm_head.weight": rng.normal(0, 0.02, (V, E)).astype(np.float32)}

for l in range(L):
    p = f"{PFX}layers.{l}"
    W[f"{p}.input_layernorm.weight"] = norm_w(E)
    W[f"{p}.post_attention_layernorm.weight"] = norm_w(E)
    if LAYER_TYPES[l] == "linear_attention":
        # qwen3_5：in_proj 拆成四份 —— qkvz / z / b / a
        W[f"{p}.linear_attn.in_proj_qkv.weight"] = rng.normal(0, 0.02, (2 * KEYD + VALD, E)).astype(np.float32)
        W[f"{p}.linear_attn.in_proj_z.weight"] = rng.normal(0, 0.02, (VALD, E)).astype(np.float32)
        W[f"{p}.linear_attn.in_proj_b.weight"] = rng.normal(0, 0.02, (NH, E)).astype(np.float32)
        W[f"{p}.linear_attn.in_proj_a.weight"] = rng.normal(0, 0.02, (NH, E)).astype(np.float32)
        W[f"{p}.linear_attn.conv1d.weight"] = rng.normal(0, 0.02, (CONVD, 1, CONV)).astype(np.float32)
        # 真实 checkpoint 没有 conv1d.bias；但 HF 的 nn.Conv1d 带 bias，缺 key 时会用
        # 随机初始化 → 必须显式写成 0，否则 HF 与 sglang-c 来不及
        W[f"{p}.linear_attn.conv1d.bias"] = np.zeros(CONVD, dtype=np.float32)
        W[f"{p}.linear_attn.norm.weight"] = np.ones(VD, dtype=np.float32)
        W[f"{p}.linear_attn.out_proj.weight"] = rng.normal(0, 0.02, (E, VALD)).astype(np.float32)
        W[f"{p}.linear_attn.A_log"] = np.log(rng.uniform(0.01, 16, NH)).astype(np.float32)
        W[f"{p}.linear_attn.dt_bias"] = np.ones(NH, dtype=np.float32)
    else:
        # q_proj 输出 2*H*D，逐头交错：[h0_q(D) h0_gate(D) h1_q(D) h1_gate(D) ...]
        W[f"{p}.self_attn.q_proj.weight"] = rng.normal(0, 0.02, (2 * H * D, E)).astype(np.float32)
        W[f"{p}.self_attn.k_proj.weight"] = rng.normal(0, 0.02, (KVH * D, E)).astype(np.float32)
        W[f"{p}.self_attn.v_proj.weight"] = rng.normal(0, 0.02, (KVH * D, E)).astype(np.float32)
        W[f"{p}.self_attn.o_proj.weight"] = rng.normal(0, 0.02, (E, H * D)).astype(np.float32)
        W[f"{p}.self_attn.q_norm.weight"] = norm_w(D)
        W[f"{p}.self_attn.k_norm.weight"] = norm_w(D)
    W[f"{p}.mlp.gate_proj.weight"] = rng.normal(0, 0.02, (I, E)).astype(np.float32)
    W[f"{p}.mlp.up_proj.weight"] = rng.normal(0, 0.02, (I, E)).astype(np.float32)
    W[f"{p}.mlp.down_proj.weight"] = rng.normal(0, 0.02, (E, I)).astype(np.float32)

def save_safetensors(path, tensors):
    hdr, off = {}, 0
    for k, v in tensors.items():
        b = np.ascontiguousarray(v, dtype=np.float32).tobytes()
        hdr[k] = {"dtype": "F32", "shape": list(np.shape(v)), "data_offsets": [off, off + len(b)]}
        off += len(b)
    hb = json.dumps(hdr).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb))); f.write(hb)
        for v in tensors.values():
            f.write(np.ascontiguousarray(v, dtype=np.float32).tobytes())

save_safetensors(f"{out}/model.safetensors", W)
import shutil
for f2 in ("tokenizer.json", "vocab.json", "merges.txt", "tokenizer_config.json"):
    p2 = f"{TOKDIR}/{f2}"
    if os.path.exists(p2):
        shutil.copy(p2, f"{out}/{f2}")
print(f"saved {out}: L={L} E={E} H={H} D={D} I={I} V={V} lin(nk={NK},nh={NH},kd={KD},vd={VD},conv={CONV}) "
      f"q_gate+partial_rope({PARTIAL}) dense_mlp layer_types={LAYER_TYPES}")