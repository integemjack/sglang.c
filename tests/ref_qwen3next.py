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


def rms(x, w, e=None):
    e = eps if e is None else e
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + e) * w


def l2norm(x, e=1e-6):
    return x / np.sqrt((x * x).sum(-1, keepdims=True) + e)


def rope(x, pos):  # x: [n, heads, Dh]
    half = Dh // 2
    inv = theta ** (-np.arange(0, half) * 2 / Dh)
    ang = np.arange(pos, pos + x.shape[0])[:, None] * inv[None, :]
    cos, sin = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    x0, x1 = x[..., :half], x[..., half:]
    return np.concatenate([x0 * cos - x1 * sin, x0 * sin + x1 * cos], -1)


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
    # gated RMSNorm（先 norm 后 * silu(z)）+ out_proj
    var = (out * out).mean(-1, keepdims=True)
    out = out / np.sqrt(var + eps) * W[f"{p}.norm.weight"] * silu(zn)
    return out.reshape(n, VALD) @ W[f"{p}.out_proj.weight"].T


def full_attn(h, l):
    p = f"model.layers.{l}"
    n = h.shape[0]
    q = (h @ W[f"{p}.self_attn.q_proj.weight"].T).reshape(n, H, Dh)
    k = (h @ W[f"{p}.self_attn.k_proj.weight"].T).reshape(n, KVH, Dh)
    v = (h @ W[f"{p}.self_attn.v_proj.weight"].T).reshape(n, KVH, Dh)
    if f"{p}.self_attn.q_norm.weight" in W:
        q = rms(q, W[f"{p}.self_attn.q_norm.weight"], 1e-6)
        k = rms(k, W[f"{p}.self_attn.k_norm.weight"], 1e-6)
    q, k = rope(q, 0), rope(k, 0)
    k = np.repeat(k, H // KVH, axis=1)
    v = np.repeat(v, H // KVH, axis=1)
    mask = np.triu(np.full((n, n), -np.inf), 1)
    att = np.einsum("qhd,khd->hqk", q, k) / np.sqrt(Dh) + mask
    att = att - att.max(-1, keepdims=True)
    att = np.exp(att)
    att = att / att.sum(-1, keepdims=True)
    o = np.einsum("hqk,khd->qhd", att, v).reshape(n, H * Dh)
    return o @ W[f"{p}.self_attn.o_proj.weight"].T


def forward(ids):
    n = len(ids)
    x = W["model.embed_tokens.weight"][ids]
    for l in range(L):
        p = f"model.layers.{l}"
        x = x + (linear_attn(rms(x, W[f"{p}.input_layernorm.weight"]), l)
                 if LT[l] == "linear_attention"
                 else full_attn(rms(x, W[f"{p}.input_layernorm.weight"]), l))
        h = rms(x, W[f"{p}.post_attention_layernorm.weight"])
        g = h @ W[f"{p}.mlp.gate_proj.weight"].T
        u = h @ W[f"{p}.mlp.up_proj.weight"].T
        x = x + silu(g) * u @ W[f"{p}.mlp.down_proj.weight"].T
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
