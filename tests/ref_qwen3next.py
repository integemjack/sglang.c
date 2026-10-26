"""G9 独立参考：numpy Qwen3-Next 混合层（gated deltanet linear_attention + full_attention）前向。
公式来源：HF transformers modeling_qwen3_next.py 的 Qwen3NextGatedDeltaNet /
torch_recurrent_gated_delta_rule（与 C 代码零共享，全程 float64）。
用法：python ref_qwen3next.py <model_dir> <out.json>"""
import json
import struct
import sys
import numpy as np
from tokenizers import Tokenizer

D = sys.argv[1] if len(sys.argv) > 1 else "models/qwen3next-rand"
OUT = sys.argv[2] if len(sys.argv) > 2 else "sglang-c/tests/_ref_qwen3next.json"
cfg = json.load(open(f"{D}/config.json"))


def load_st(path):
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen))
        data = f.read()
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        s, t = meta["data_offsets"]
        out[name] = np.frombuffer(data[s:t], dtype=np.float32).reshape(meta["shape"]).astype(np.float64)
    return out


W = load_st(f"{D}/model.safetensors")
E = cfg["hidden_size"]
H = cfg["num_attention_heads"]
KVH = cfg.get("num_key_value_heads", H)
L = cfg["num_hidden_layers"]
I = cfg["intermediate_size"]
V = cfg["vocab_size"]
Dh = cfg.get("head_dim", E // H)
eps = cfg["rms_norm_eps"]
theta = cfg.get("rope_theta", 10000000.0)
LT = cfg["layer_types"]
NK = cfg["linear_num_key_heads"]
NH = cfg["linear_num_value_heads"]
KD = cfg["linear_key_head_dim"]
VD = cfg["linear_value_head_dim"]
KCONV = cfg["linear_conv_kernel_dim"]
RATIO = NH // NK
KEYD, VALD = NK * KD, NH * VD
CONVD = 2 * KEYD + VALD

# ---- G13: full_attention 的真实特性（对齐 HF modeling_qwen3_next.py）----
RP = cfg.get("rope_parameters") or {}
theta = RP.get("rope_theta", cfg.get("rope_theta", 10000000.0))
PARTIAL = RP.get("partial_rotary_factor", cfg.get("partial_rotary_factor", 1.0))
ROT = int(Dh * PARTIAL)          # partial rotary：只旋转头维的前 ROT 维
NEXP = cfg.get("num_experts", 0)
TOPK = cfg.get("num_experts_per_tok", 0)
MOE_I = cfg.get("moe_intermediate_size", 0)
SH_I = cfg.get("shared_expert_intermediate_size", 0)
SPARSE = cfg.get("decoder_sparse_step", 1)
MLP_ONLY = cfg.get("mlp_only_layers", []) or []
NORM_TOPK = cfg.get("norm_topk_prob", False)
ROUTED = cfg.get("routed_scaling_factor", 1.0)


def is_moe(l):
    return NEXP > 0 and (l not in MLP_ONLY) and ((l + 1) % SPARSE == 0)


def rms(x, w, e=None):
    """Qwen3NextRMSNorm：(1+w) * x / sqrt(var+eps)（HF：w 初值 0，前向用 1+w）"""
    e = eps if e is None else e
    return (1.0 + w) * x / np.sqrt((x * x).mean(-1, keepdims=True) + e)


def rms_plain(x, w, e=None):
    """普通 RMSNorm（linear_attn 的 Qwen3NextRMSNormGated 用：w 是正向权重，无 1+）"""
    e = eps if e is None else e
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + e) * w


def l2norm(x, e=1e-6):
    return x / np.sqrt((x * x).sum(-1, keepdims=True) + e)


def rope(x, pos):  # x: [n, heads, Dh]；只旋转前 ROT 维（partial rotary）
    dim = ROT if ROT > 0 else Dh
    half = dim // 2
    inv = theta ** (-np.arange(0, half) * 2 / dim)
    ang = np.arange(pos, pos + x.shape[0])[:, None] * inv[None, :]
    cos, sin = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    xr, xp = x[..., :dim], x[..., dim:]
    x0, x1 = xr[..., :half], xr[..., half:]
    rot = np.concatenate([x0 * cos - x1 * sin, x0 * sin + x1 * cos], -1)
    return np.concatenate([rot, xp], -1)


def silu(x):
    return x / (1.0 + np.exp(-x))


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def linear_attn(h, l):
    """h: [n, E] -> out [n, E]。按 HF Qwen3NextGatedDeltaNet 公式。"""
    p = f"model.layers.{l}.linear_attn"
    qkvz = h @ W[f"{p}.in_proj_qkvz.weight"].T          # [n, 2*KEYD+2*VALD]
    ba = h @ W[f"{p}.in_proj_ba.weight"].T              # [n, 2*NH]
    gw = 2 * KD + 2 * RATIO * VD                        # 每 k-head 组宽
    # 拼 mixed_qkv [q|k|v]（非交错），做因果 depthwise conv + silu
    mixed = np.zeros((h.shape[0], CONVD))
    for g in range(NK):
        base = qkvz[:, g * gw:(g + 1) * gw]
        mixed[:, g * KD:(g + 1) * KD] = base[:, 0:KD]
        mixed[:, KEYD + g * KD:KEYD + (g + 1) * KD] = base[:, KD:2 * KD]
        mixed[:, 2 * KEYD + g * RATIO * VD:2 * KEYD + (g + 1) * RATIO * VD] = base[:, 2 * KD:2 * KD + RATIO * VD]
    cw = W[f"{p}.conv1d.weight"].reshape(CONVD, KCONV)  # [convd, 1, K]
    n = h.shape[0]
    conv_out = np.zeros((n, CONVD))
    for t in range(n):
        for j in range(KCONV):
            src = t - KCONV + 1 + j
            if src >= 0:
                conv_out[t] += cw[:, j] * mixed[src]
    conv_out = silu(conv_out)
    qc = conv_out[:, :KEYD]
    kc = conv_out[:, KEYD:2 * KEYD]
    vc = conv_out[:, 2 * KEYD:]
    # beta / decay（b、a 按组交错：[b RATIO | a RATIO]）
    b_idx = [(h2 // RATIO) * 2 * RATIO + h2 % RATIO for h2 in range(NH)]
    a_idx = [i + RATIO for i in b_idx]
    beta = 1.0 / (1.0 + np.exp(-ba[:, b_idx]))          # [n, NH]
    gdec = -np.exp(W[f"{p}.A_log"]) * softplus(ba[:, a_idx] + W[f"{p}.dt_bias"])  # [n, NH]
    # 逐 token 递归（gated delta rule）
    qn = l2norm(qc.reshape(n, NK, KD)) / np.sqrt(KD)
    kn = l2norm(kc.reshape(n, NK, KD))
    vn = vc.reshape(n, NH, VD)
    zn = np.zeros((n, NH, VD))
    for g in range(NK):
        base = qkvz[:, g * gw:(g + 1) * gw]
        zn[:, g * RATIO:(g + 1) * RATIO] = base[:, 2 * KD + RATIO * VD:].reshape(n, RATIO, VD)
    S = np.zeros((NH, KD, VD))
    out = np.zeros((n, NH, VD))
    for t in range(n):
        for hh in range(NH):
            g = hh // RATIO
            qh, kh = qn[t, g], kn[t, g]
            S[hh] *= np.exp(gdec[t, hh])
            mem = S[hh].T @ kh                          # [VD]
            delta = (vn[t, hh] - mem) * beta[t, hh]
            S[hh] += np.outer(kh, delta)
            out[t, hh] = S[hh].T @ qh
    # gated RMSNorm（先 norm 后 * silu(z)）+ out_proj（普通权重，无 (1+w)）
    var = (out * out).mean(-1, keepdims=True)
    out = rms_plain(out, W[f"{p}.norm.weight"]) * silu(zn)
    return out.reshape(n, VALD) @ W[f"{p}.out_proj.weight"].T


def full_attn(h, l):
    p = f"model.layers.{l}"
    n = h.shape[0]
    qg = h @ W[f"{p}.self_attn.q_proj.weight"].T          # [n, H*Dh*2]
    # G13(a): q_proj 输出按头交错的 [q | gate]，chunk 在最后一维上切
    qg = qg.reshape(n, H, Dh * 2)
    q, gate = qg[..., :Dh], qg[..., Dh:]
    k = (h @ W[f"{p}.self_attn.k_proj.weight"].T).reshape(n, KVH, Dh)
    v = (h @ W[f"{p}.self_attn.v_proj.weight"].T).reshape(n, KVH, Dh)
    if f"{p}.self_attn.q_norm.weight" in W:
        q = rms(q, W[f"{p}.self_attn.q_norm.weight"], 1e-6)   # G13(c): (1+w)
        k = rms(k, W[f"{p}.self_attn.k_norm.weight"], 1e-6)
    q, k = rope(q, 0), rope(k, 0)                              # G13(b): partial rotary
    k = np.repeat(k, H // KVH, axis=1)
    v = np.repeat(v, H // KVH, axis=1)
    mask = np.triu(np.full((n, n), -np.inf), 1)
    att = np.einsum("qhd,khd->hqk", q, k) / np.sqrt(Dh) + mask
    att = att - att.max(-1, keepdims=True)
    att = np.exp(att)
    att = att / att.sum(-1, keepdims=True)
    o = np.einsum("hqk,khd->qhd", att, v)
    o = o * (1.0 / (1.0 + np.exp(-gate)))                      # G13(a): sigmoid 门控
    return o.reshape(n, H * Dh) @ W[f"{p}.self_attn.o_proj.weight"].T


def dense_mlp(h, l):
    p = f"model.layers.{l}"
    return silu(h @ W[f"{p}.mlp.gate_proj.weight"].T) * (h @ W[f"{p}.mlp.up_proj.weight"].T) \
        @ W[f"{p}.mlp.down_proj.weight"].T


def moe_mlp(h, l):
    """G13(d): Qwen3NextSparseMoeBlock（router softmax+topk+norm、专家 gate_up/down、
    共享专家 + sigmoid 共享门控、routed_scaling_factor）。"""
    p = f"model.layers.{l}.mlp"
    n = h.shape[0]
    logits = h @ W[f"{p}.gate.weight"].T                       # [n, NEXP]
    e = np.exp(logits - logits.max(-1, keepdims=True))
    probs = e / e.sum(-1, keepdims=True)
    idx = np.argsort(-probs, axis=-1)[:, :TOPK]
    wgt = np.take_along_axis(probs, idx, axis=-1)
    if NORM_TOPK:
        wgt = wgt / wgt.sum(-1, keepdims=True)
    out = np.zeros((n, E))
    for t in range(n):
        for j in range(TOPK):
            ex = idx[t, j]
            gu = h[t] @ W[f"{p}.experts.gate_up_proj"][ex].T   # [2*MOE_I]
            g2, u2 = gu[:MOE_I], gu[MOE_I:]
            out[t] += (silu(g2) * u2) @ W[f"{p}.experts.down_proj"][ex].T * wgt[t, j] * ROUTED
    sh = silu(h @ W[f"{p}.shared_expert.gate_proj.weight"].T) * \
        (h @ W[f"{p}.shared_expert.up_proj.weight"].T)
    sh = sh @ W[f"{p}.shared_expert.down_proj.weight"].T
    sg = 1.0 / (1.0 + np.exp(-(h @ W[f"{p}.shared_expert_gate.weight"].T)))  # [n,1]
    return out + sh * sg


def forward(ids):
    n = len(ids)
    x = W["model.embed_tokens.weight"][ids]
    for l in range(L):
        p = f"model.layers.{l}"
        x = x + (linear_attn(rms(x, W[f"{p}.input_layernorm.weight"]), l)
                 if LT[l] == "linear_attention"
                 else full_attn(rms(x, W[f"{p}.input_layernorm.weight"]), l))
        h = rms(x, W[f"{p}.post_attention_layernorm.weight"])
        mx = moe_mlp(h, l) if is_moe(l) else dense_mlp(h, l)
        x = x + mx
    x = rms(x, W["model.norm.weight"])
    lm = W["lm_head.weight"] if "lm_head.weight" in W else W["model.embed_tokens.weight"]
    return x[-1] @ lm.T


tok = Tokenizer.from_file(f"{D}/tokenizer.json")
out = {"config": {"E": E, "H": H, "L": L, "KVH": KVH, "V": V}, "cases": []}
for text in ["Hello world", "The meaning of life is", "Paris is the capital of",
             " ".join(["word"] * 90)]:
    ids = tok.encode(text).ids
    logits = forward(ids)
    top = np.argsort(-logits)[:5]
    greedy = []
    cur = list(ids)
    for _ in range(6):
        lg = forward(cur)
        t = int(np.argmax(lg))
        greedy.append(t)
        cur.append(t)
    out["cases"].append({
        "text": text, "ids": ids,
        "top5": [int(t) for t in top],
        "top5_logits": [float(logits[t]) for t in top],
        "greedy6": greedy,
    })
    print(f"{text!r} -> ids={ids} top5={list(top)} greedy6={greedy}")

json.dump(out, open(OUT, "w"), indent=1)
print(f"saved {OUT}")
