"""独立参考：numpy Llama/Qwen2 forward，生成 ref JSON 给 C 对照。
用法：python ref_llama.py <model_dir> <out.json>；与 C 代码零共享。"""
import json
import sys
import numpy as np
from tokenizers import Tokenizer

D = sys.argv[1] if len(sys.argv) > 1 else "models/tiny-llama"
OUT = sys.argv[2] if len(sys.argv) > 2 else "sglang-c/tests/ref_tinyllama.json"
cfg = json.load(open(f"{D}/config.json"))


def load_st(path):
    """轻量 safetensors 读入（支持 bf16：u16 << 16 视为 f32）"""
    import struct
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen))
        data = f.read()
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        s, t = meta["data_offsets"]
        raw = data[s:t]
        if meta["dtype"] == "BF16":
            u16 = np.frombuffer(raw, dtype=np.uint16)
            arr = (u16.astype(np.uint32) << 16).view(np.float32)
        elif meta["dtype"] == "F16":
            arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
        else:
            arr = np.frombuffer(raw, dtype=np.float32)
        out[name] = arr.reshape(meta["shape"])
    return out


W = load_st(f"{D}/model.safetensors")

E, H = cfg["hidden_size"], cfg["num_attention_heads"]
L = cfg["num_hidden_layers"]
KVH = cfg.get("num_key_value_heads", H)
I = cfg.get("intermediate_size", 4 * E)
V = cfg["vocab_size"]
Dh = cfg.get("head_dim", E // H)
KD = Dh * KVH
IS_QWEN = "model.layers.0.self_attn.q_proj.bias" in W or \
    any(".q_norm.weight" in k for k in W)
eps = cfg["rms_norm_eps"]
theta = cfg.get("rope_theta", 10000.0)
# G9 gemma-3: embedding tanh 软上限（在 sqrt(E) scale 前）
SOFTCAP = float(cfg.get("hidden_softcap", 0) or 0)
# G9: rope_scaling（linear：角度除以 factor；dynamic：NTK 基频缩放 theta*factor^(D/(D-2))）
#     + layer_types（sliding_attention 层用 rope_local_base_freq，其余用
#       rope_scaling.global_base_freq；无 layer_types 时保持旧行为：全局 sliding_window）
RS = cfg.get("rope_scaling") or {}
RTYPE = RS.get("rope_type", "default")
RFACTOR = float(RS.get("factor", 1.0) or 1.0)
POS_SCALE = 1.0 / RFACTOR if RTYPE == "linear" else 1.0
LOCAL_BASE = float(cfg.get("rope_local_base_freq", 0) or 0)
GLOBAL_BASE = float(RS.get("global_base_freq", 0) or 0)
LAYER_TYPES = cfg.get("layer_types") or None


def layer_rope(l):
    """G9：第 l 层的 (rope 基频, sliding 窗口)。无 layer_types 时同旧行为。"""
    if LAYER_TYPES:
        t = LAYER_TYPES[l]
        kind = "sliding" if t == "sliding_attention" else "full"
    else:
        kind = "sliding" if cfg.get("sliding_window") else "full"
    base = theta
    if LAYER_TYPES and kind == "sliding" and LOCAL_BASE > 0:
        base = LOCAL_BASE
    elif LAYER_TYPES and kind == "full" and GLOBAL_BASE > 0:
        base = GLOBAL_BASE
    if RTYPE == "dynamic":
        base = base * RFACTOR ** (Dh / (Dh - 2))
    win = (cfg.get("sliding_window") or 0) if kind == "sliding" else 0
    return base, win


def f32(a):
    return a.astype(np.float32)


def rms(x, w):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def rope(x, pos, base=None):  # x: [n, heads, Dh]；G9：基频/位置缩放可按层不传 base 用全局 theta
    half = Dh // 2
    if base is None:
        base = theta
    inv = base ** (-np.arange(0, half, dtype=np.float64) * 2 / Dh)
    ang = (np.arange(pos, pos + x.shape[0], dtype=np.float64) * POS_SCALE)[:, None] * inv[None, :]  # [n, half]
    cos, sin = np.cos(ang).astype(np.float32)[:, None, :], np.sin(ang).astype(np.float32)[:, None, :]  # [n,1,half]
    x0, x1 = x[..., :half], x[..., half:]
    # HF llama: rotate_half: cat(-x1, x0)；q_embed = q*cos + rotate_half(q)*sin
    return np.concatenate([x0 * cos - x1 * sin, x0 * sin + x1 * cos], -1)


def forward(ids):
    n = len(ids)
    x = f32(W["model.embed_tokens.weight"][ids])
    if SOFTCAP:  # G9 gemma-3: emb = cap*tanh(emb/cap)，在 sqrt(E) scale 前
        x = f32(np.float32(SOFTCAP) * np.tanh(x / np.float32(SOFTCAP)))
    if cfg.get("model_type") == "gemma":  # t9: hidden normalizer
        x = x * np.sqrt(E, dtype=np.float32)
    for l in range(L):
        p = f"model.layers.{l}"
        base_l, win_l = layer_rope(l)
        mask = np.triu(np.full((n, n), -1e9, dtype=np.float32), 1)
        if win_l:  # t9 mistral / G9 sliding 层：带状 mask（attend p in [q-win+1, q]）
            band = np.abs(np.subtract.outer(np.arange(n), np.arange(n))) >= win_l
            mask = mask + np.where(band, -1e9, 0).astype(np.float32)
        h = rms(x, f32(W[f"{p}.input_layernorm.weight"]))
        q = h @ f32(W[f"{p}.self_attn.q_proj.weight"]).T
        k = h @ f32(W[f"{p}.self_attn.k_proj.weight"]).T
        v = h @ f32(W[f"{p}.self_attn.v_proj.weight"]).T
        q = q.reshape(n, H, Dh)
        k = k.reshape(n, KVH, Dh)
        if IS_QWEN:  # Qwen2：qkv bias（存在时加）
            if f"{p}.self_attn.q_proj.bias" in W:
                q = (q.reshape(n, E) + f32(W[f"{p}.self_attn.q_proj.bias"])).reshape(n, H, Dh)
                k = (k.reshape(n, KD) + f32(W[f"{p}.self_attn.k_proj.bias"])).reshape(n, KVH, Dh)
                v = v + f32(W[f"{p}.self_attn.v_proj.bias"])
            if f"{p}.self_attn.q_norm.weight" in W:  # QK-Norm（RoPE 前，head_dim 维，eps 1e-6）
                qn = f32(W[f"{p}.self_attn.q_norm.weight"])
                kn = f32(W[f"{p}.self_attn.k_norm.weight"])
                q = q / np.sqrt((q * q).mean(-1, keepdims=True) + 1e-6) * qn
                k = k / np.sqrt((k * k).mean(-1, keepdims=True) + 1e-6) * kn
        q = rope(q, 0, base_l)
        k = rope(k, 0, base_l)
        rep = H // KVH
        k = np.repeat(k, rep, axis=1)
        v = np.repeat(v.reshape(n, KVH, Dh), rep, axis=1)
        att = np.einsum("qhd,khd->hqk", q, k) / np.sqrt(Dh) + mask
        att = att - att.max(-1, keepdims=True)
        att = np.exp(att)
        att = att / att.sum(-1, keepdims=True)
        o = np.einsum("hqk,khd->qhd", att, v).reshape(n, H * Dh)  # H*D rows (qwen3: != E)
        x = x + o @ f32(W[f"{p}.self_attn.o_proj.weight"]).T
        h = rms(x, f32(W[f"{p}.post_attention_layernorm.weight"]))
        g = h @ f32(W[f"{p}.mlp.gate_proj.weight"]).T
        u = h @ f32(W[f"{p}.mlp.up_proj.weight"]).T
        act = g / (1 + np.exp(-g)) * u
        x = x + act @ f32(W[f"{p}.mlp.down_proj.weight"]).T
    x = rms(x, f32(W["model.norm.weight"]))
    logits = x[-1] @ f32(W["lm_head.weight"].T if "lm_head.weight" in W else W["model.embed_tokens.weight"].T)
    return logits


# tokenizer 参考（用 tiktoken 不行——SP 模型。手写与 C 相同思路会共错；
# 用 HuggingFace tokenizers 库（Rust 实现，权威））
from tokenizers import Tokenizer
tok = Tokenizer.from_file(f"{D}/tokenizer.json")

out = {"config": {"E": E, "H": H, "L": L, "KVH": KVH, "V": V}, "cases": []}
for text in ["Hello world", "The meaning of life is", "Paris is the capital of",
             " ".join(["word"] * 90)]:  # t9: >sliding_window(64) tokens to exercise the band
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
