"""G9/G13: 生成随机权重的 Qwen3-Next 结构小模型，含真实 full_attention 的全部特性。

结构（对齐 HF transformers `modeling_qwen3_next.py`）：
- layer_types = ["linear_attention", "full_attention", "linear_attention", "full_attention"]
  （1 个 dense linear 层 + 1 个 MoE linear 层 + 2 个 full 层，其中 1 个 MoE）
- linear_attention：gated deltanet（G9 已实现），权重名 linear_attn.in_proj_qkvz /
  in_proj_ba / conv1d / norm / out_proj / A_log / dt_bias
- full_attention（G13 补齐）：
    a) q_proj 输出 2*H*D，半幅 sigmoid 门控 attn 输出（Qwen3NextAttention.forward）
    b) partial rotary：rope_parameters.partial_rotary_factor=0.25，只旋转 head_dim 的前 25%
    c) 所有 RMSNorm 是 (1+w)*x/sqrt(var+eps)（Qwen3NextRMSNorm，w 初值 0）
    d) MoE：router（softmax + top-k + norm_topk_prob）+ 专家 gate_up/down + 共享专家
       + shared_expert_gate（sigmoid）+ routed_scaling_factor
- decoder_sparse_step=1 且 mlp_only_layers=[0]：第 0 层 dense MLP，其余层都是 MoE
  （所以同一个模型里两条 MLP 路径都被覆盖）

norm 权重取 normal(0, 0.3)（不是 HF 的零初值）：这样「(1+w) 还是 w」写错会被立刻
放大成 O(1) 的误差，比零初值更有判别力；参考脚本按同一份权重算。

用法: python gen_random_qwen3next.py <out_dir>
     python gen_random_qwen3next.py --out models/qwen3next-rand --tokenizer models/qwen3-06
"""
import json, os, sys
import numpy as np

def _args(argv, default_out):
    """G19: --out/-o 与 --tokenizer/-t 可传，位置参数仍兼容。"""
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

out, TOKDIR, _pos = _args(sys.argv[1:], "models/qwen3next-rand")
L, H, KVH, E, I, V = 4, 4, 2, 64, 128, 151936
D = E // H
NK, NH, KD, VD, CONV = 2, 4, 16, 16, 4  # linear attn：k 头 2、v 头 4、头维 16、conv 核 4
RATIO = NH // NK
KEYD, VALD, CONVD = NK * KD, NH * VD, 2 * NK * KD + NH * VD
LAYER_TYPES = ["linear_attention", "full_attention", "linear_attention", "full_attention"]
NEXP, TOPK, MOE_I, SH_I = 4, 2, 32, 32          # G13: MoE（4 专家、top-2）+ 共享专家
SPARSE_STEP, MLP_ONLY = 1, [0]                  # 第 0 层 dense，其余 MoE
PARTIAL = 0.25                                  # G13: partial rotary
ROUTED_SCALE = 1.5                              # G13: routed scaling factor
rng = np.random.default_rng(7)
os.makedirs(out, exist_ok=True)

cfg = {
    "architectures": ["Qwen3NextForCausalLM"], "model_type": "qwen3_next",
    "hidden_size": E, "intermediate_size": I, "num_hidden_layers": L,
    "num_attention_heads": H, "num_key_value_heads": KVH,
    "head_dim": D, "vocab_size": V, "rms_norm_eps": 1e-6,
    # 字段布局与真实 80B config.json 一致：rope_theta 与 partial_rotary_factor 在顶层
    "rope_theta": 1000000.0, "partial_rotary_factor": PARTIAL,
    "max_position_embeddings": 512,
    "tie_word_embeddings": True, "attention_bias": False,
    "layer_types": LAYER_TYPES,
    "linear_num_key_heads": NK, "linear_num_value_heads": NH,
    "linear_key_head_dim": KD, "linear_value_head_dim": VD,
    "linear_conv_kernel_dim": CONV,
    "num_experts": NEXP, "num_experts_per_tok": TOPK, "moe_intermediate_size": MOE_I,
    "shared_expert_intermediate_size": SH_I, "decoder_sparse_step": SPARSE_STEP,
    "mlp_only_layers": MLP_ONLY, "norm_topk_prob": True,
    "routed_scaling_factor": ROUTED_SCALE, "hidden_act": "silu",
}
json.dump(cfg, open(f"{out}/config.json", "w"), indent=1)

def norm_w(n):
    """Qwen3NextRMSNorm：前向是 (1+w)*x/sqrt(var+eps)，w 由模型学出（HF 初值 0）。"""
    return rng.normal(0, 0.3, n).astype(np.float32)

W = {"model.embed_tokens.weight": rng.normal(0, 0.02, (V, E)).astype(np.float32),
     "model.norm.weight": norm_w(E)}
for l in range(L):
    p = f"model.layers.{l}"
    W[f"{p}.input_layernorm.weight"] = norm_w(E)
    W[f"{p}.post_attention_layernorm.weight"] = norm_w(E)
    if LAYER_TYPES[l] == "linear_attention":
        W[f"{p}.linear_attn.in_proj_qkvz.weight"] = rng.normal(0, 0.02, (2 * KEYD + 2 * VALD, E)).astype(np.float32)
        W[f"{p}.linear_attn.in_proj_ba.weight"] = rng.normal(0, 0.02, (2 * NH, E)).astype(np.float32)
        W[f"{p}.linear_attn.conv1d.weight"] = rng.normal(0, 0.02, (CONVD, 1, CONV)).astype(np.float32)
        W[f"{p}.linear_attn.norm.weight"] = np.ones(VD, dtype=np.float32)  # gated norm：普通权重，无 (1+w)
        W[f"{p}.linear_attn.out_proj.weight"] = rng.normal(0, 0.02, (E, VALD)).astype(np.float32)
        W[f"{p}.linear_attn.A_log"] = np.log(rng.uniform(0.01, 16, NH)).astype(np.float32)
        W[f"{p}.linear_attn.dt_bias"] = np.ones(NH, dtype=np.float32)
    else:
        # G13(a): q_proj 输出 2*H*D（前 H*D 是 q，后 H*D 是 gate，逐头交错）
        W[f"{p}.self_attn.q_proj.weight"] = rng.normal(0, 0.02, (2 * H * D, E)).astype(np.float32)
        W[f"{p}.self_attn.k_proj.weight"] = rng.normal(0, 0.02, (KVH * D, E)).astype(np.float32)
        W[f"{p}.self_attn.v_proj.weight"] = rng.normal(0, 0.02, (KVH * D, E)).astype(np.float32)
        W[f"{p}.self_attn.o_proj.weight"] = rng.normal(0, 0.02, (E, H * D)).astype(np.float32)
        W[f"{p}.self_attn.q_norm.weight"] = norm_w(D)   # G13(c)
        W[f"{p}.self_attn.k_norm.weight"] = norm_w(D)
    if l in MLP_ONLY:
        W[f"{p}.mlp.gate_proj.weight"] = rng.normal(0, 0.02, (I, E)).astype(np.float32)
        W[f"{p}.mlp.up_proj.weight"] = rng.normal(0, 0.02, (I, E)).astype(np.float32)
        W[f"{p}.mlp.down_proj.weight"] = rng.normal(0, 0.02, (E, I)).astype(np.float32)
    else:
        # G13(d): MoE —— router + 专家（gate_up 融合）+ 共享专家 + 共享门控
        W[f"{p}.mlp.gate.weight"] = rng.normal(0, 0.3, (NEXP, E)).astype(np.float32)
        W[f"{p}.mlp.experts.gate_up_proj"] = rng.normal(0, 0.02, (NEXP, 2 * MOE_I, E)).astype(np.float32)
        W[f"{p}.mlp.experts.down_proj"] = rng.normal(0, 0.02, (NEXP, E, MOE_I)).astype(np.float32)
        W[f"{p}.mlp.shared_expert.gate_proj.weight"] = rng.normal(0, 0.02, (SH_I, E)).astype(np.float32)
        W[f"{p}.mlp.shared_expert.up_proj.weight"] = rng.normal(0, 0.02, (SH_I, E)).astype(np.float32)
        W[f"{p}.mlp.shared_expert.down_proj.weight"] = rng.normal(0, 0.02, (E, SH_I)).astype(np.float32)
        W[f"{p}.mlp.shared_expert_gate.weight"] = rng.normal(0, 0.3, (1, E)).astype(np.float32)

import struct
def save_safetensors(path, tensors):
    hdr = {}
    off = 0
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
if not os.path.exists(f"{TOKDIR}/tokenizer.json"):
    print(f"WARN: no tokenizer at {TOKDIR}/tokenizer.json (--tokenizer <dir> to point at one)")
print(f"saved random qwen3_next -> {out} (L={L} E={E} D={D} lin: nk={NK} nh={NH} kd={KD} vd={VD} conv={CONV}; "
      f"full: q_gate+partial_rotary({PARTIAL})+norm(1+w); moe: {NEXP} experts top{TOPK} "
      f"moe_i={MOE_I} shared_i={SH_I} routed_scale={ROUTED_SCALE} sparse_step={SPARSE_STEP} mlp_only={MLP_ONLY})")