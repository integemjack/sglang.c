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


def f32(a):
    return a.astype(np.float32)


def rms(x, w):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def rope(x, pos):  # x: [n, heads, Dh]
    half = Dh // 2
    inv = theta ** (-np.arange(0, half, dtype=np.float32) * 2 / Dh)
    ang = np.arange(pos, pos + x.shape[0], dtype=np.float32)[:, None] * inv[None, :]  # [n, half]
    cos, sin = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]  # [n,1,half]
    x0, x1 = x[..., :half], x[..., half:]
    # HF llama: rotate_half: cat(-x1, x0)；q_embed = q*cos + rotate_half(q)*sin
    return np.concatenate([x0 * cos - x1 * sin, x0 * sin + x1 * cos], -1)


def forward(ids):
    n = len(ids)
    x = f32(W["model.embed_tokens.weight"][ids])
    mask = np.triu(np.full((n, n), -1e9, dtype=np.float32), 1)
    for l in range(L):
        p = f"model.layers.{l}"
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
        q = rope(q, 0)
        k = rope(k, 0)
        rep = H // KVH
        k = np.repeat(k, rep, axis=1)
        v = np.repeat(v.reshape(n, KVH, Dh), rep, axis=1)
        att = np.einsum("qhd,khd->hqk", q, k) / np.sqrt(Dh) + mask
        att = att - att.max(-1, keepdims=True)
        att = np.exp(att)
        att = att / att.sum(-1, keepdims=True)
        o = np.einsum("hqk,khd->qhd", att, v).reshape(n, E)
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
for text in ["Hello world", "The meaning of life is", "Paris is the capital of"]:
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
