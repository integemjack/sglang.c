"""tests/gen_random_mla.py — 生成随机 DeepSeek MLA 小模型 + numpy 参考输出（tests/test_mla.c 用）

结构照 HF modeling_deepseek_v2.py 的 DeepseekV2Attention：
  q  = q_b_proj(q_a_layernorm(q_a_proj(x)))          （q_lora_rank 为 null 时 = q_proj(x)）
  kv = kv_a_proj_with_mqa(x) → [c | k_pe]，c = kv_a_layernorm(c)
  k_nope / v = split(kv_b_proj(c)) 按 head 拆
  RoPE（非交错 rotate_half，只打在 q_pe 每头、k_pe 单份共享）
  score = (q_nope·k_nope + q_pe·k_pe) * q_head_dim**-0.5，causal softmax，乘 v
  out = o_proj(concat_h(attn_h))
参考用「非吸收式」算（C 侧是吸收式），两者数学等价，差异仅 fp32 舍入。

输出：
  <out>/config.json         DeepSeek 风格 config
  <out>/model.safetensors   权重（F32，HF DeepSeek 命名 model.layers.0.self_attn.*）
  <out>/mla_ref.json        n_pos / x[n_pos][hidden] / out[n_pos][hidden] / cache[n_pos][stride]

用法：
  python tests/gen_random_mla.py --out models/mla-rand --seed 7 --q-lora 12
  python tests/gen_random_mla.py --out models/mla-rand-q0 --seed 11 --q-lora 0
"""
import json
import os
import struct
import sys

import numpy as np


def save_safetensors(path, tensors):
    hdr = {}
    off = 0
    for k, v in tensors.items():
        b = v.tobytes()
        hdr[k] = {"dtype": "F32", "shape": list(v.shape), "data_offsets": [off, off + len(b)]}
        off += len(b)
    hb = json.dumps(hdr).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for v in tensors.values():
            f.write(v.tobytes())


def rmsnorm(x, w, eps):
    """HF DeepseekV2RMSNorm：x / sqrt(mean(x^2)+eps) * w（float64 计算）"""
    x = np.asarray(x, dtype=np.float64)
    v = (x * x).mean(axis=-1, keepdims=True)
    return x / np.sqrt(v + eps) * np.asarray(w, dtype=np.float64)


def rope_rotate_half(x, pos, theta, dim):
    """HF 默认的非交错 rotate_half：对最后 dim 维，配对 (i, i+dim/2)。

    x: [n, dim]；pos: [n]（每个 token 的位置）。inv_freq_i = theta^(-2i/dim)。
    """
    n = x.shape[0]
    half = dim // 2
    i = np.arange(half, dtype=np.float64)
    inv = theta ** (-2.0 * i / dim)
    ang = np.asarray(pos, dtype=np.float64)[:, None] * inv[None, :]
    c, s = np.cos(ang), np.sin(ang)
    x1, x2 = x[:, :half], x[:, half:]
    return np.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], axis=-1)


def ref_forward(x, W, cfg):
    """非吸收式参考前向（全序列 prefill = 每个位置的 causal 输出）。"""
    n = x.shape[0]
    H = cfg["num_attention_heads"]
    Dn = cfg["qk_nope_head_dim"]
    Dr = cfg["qk_rope_head_dim"]
    Dc = cfg["kv_lora_rank"]
    Dv = cfg["v_head_dim"]
    eps = cfg["rms_norm_eps"]
    theta = cfg["rope_theta"]
    qlr = cfg["q_lora_rank"]
    pos = np.arange(n)

    x = np.asarray(x, dtype=np.float64)
    if qlr:
        q = rmsnorm(x @ W["q_a"].T, W["q_an"], eps) @ W["q_b"].T
    else:
        q = x @ W["q"].T
    q = q.reshape(n, H, Dn + Dr)
    q_nope, q_pe = q[..., :Dn], q[..., Dn:]

    kva = x @ W["kv_a"].T                      # [n, Dc+Dr]
    c = rmsnorm(kva[:, :Dc], W["kv_an"], eps)  # [n, Dc]
    k_pe = kva[:, Dc:]                         # [n, Dr]，全 head 共享
    kv = c @ W["kv_b"].T                       # [n, H*(Dn+Dv)]
    kv = kv.reshape(n, H, Dn + Dv)
    k_nope, v = kv[..., :Dn], kv[..., Dn:]

    # RoPE：q_pe 每头一份（同一位置 n*H 行），k_pe 每位置一份
    q_pe_r = rope_rotate_half(q_pe.reshape(n * H, Dr), np.repeat(pos, H), theta, Dr)
    q_pe_r = q_pe_r.reshape(n, H, Dr)
    k_pe_r = rope_rotate_half(k_pe, pos, theta, Dr)

    scale = (Dn + Dr) ** -0.5
    mask = np.triu(np.ones((n, n), dtype=bool), 1)  # causal：j > i 屏蔽
    out = np.zeros((n, H, Dv), dtype=np.float64)
    for h in range(H):
        s = q_nope[:, h, :] @ k_nope[:, h, :].T + q_pe_r[:, h, :] @ k_pe_r.T
        s = s * scale
        s = np.where(mask, -np.inf, s)
        s = s - s.max(axis=-1, keepdims=True)
        a = np.exp(s)
        a = a / a.sum(axis=-1, keepdims=True)
        out[:, h, :] = a @ v[:, h, :]
    out = out.reshape(n, H * Dv) @ W["o"].T

    cache = np.concatenate([c, k_pe_r], axis=-1)  # [n, Dc+Dr]：MLA 压缩 KV 行
    return out, cache


def f32list(a):
    """f32 精确回读的 JSON 列表（str(float(f32)) 是 short roundtrip，无损）"""
    flat = np.asarray(a, dtype=np.float32).reshape(-1)
    return "[" + ",".join(repr(float(v)) for v in flat) + "]"


def mat_json(a):
    a = np.asarray(a, dtype=np.float32)
    rows = [f32list(r) for r in a.reshape(-1, a.shape[-1])]
    return "[\n" + ",\n".join(rows) + "\n]"


def main(argv):
    dims = {"--hidden": 32, "--heads": 4, "--kv-lora": 16, "--nope": 8, "--rope": 4,
            "--v-head": 6}
    out = "models/mla-rand"
    seed = 7
    q_lora = 12
    hidden, heads, kv_lora, d_nope, d_rope, v_head = (dims[k] for k in
        ("--hidden", "--heads", "--kv-lora", "--nope", "--rope", "--v-head"))
    n_pos, ctx = 9, 64
    i = 0
    while i < len(argv):
        a = argv[i]
        if i + 1 < len(argv) and a in ("--out", "-o"):
            out = argv[i + 1]; i += 2; continue
        if i + 1 < len(argv) and a == "--seed":
            seed = int(argv[i + 1]); i += 2; continue
        if i + 1 < len(argv) and a == "--q-lora":
            q_lora = int(argv[i + 1]); i += 2; continue
        if i + 1 < len(argv) and a == "--n-pos":
            n_pos = int(argv[i + 1]); i += 2; continue
        if i + 1 < len(argv) and a in dims:
            dims[a] = int(argv[i + 1]); i += 2; continue
        raise SystemExit(f"unknown arg: {a}")
    hidden, heads, kv_lora = dims["--hidden"], dims["--heads"], dims["--kv-lora"]
    d_nope, d_rope, v_head = dims["--nope"], dims["--rope"], dims["--v-head"]

    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(seed)

    cfg = {
        "architectures": ["DeepseekV2ForCausalLM"],
        "model_type": "deepseek_v2",
        "hidden_size": hidden,
        "num_hidden_layers": 1,
        "num_attention_heads": heads,
        "q_lora_rank": (q_lora if q_lora > 0 else None),
        "kv_lora_rank": kv_lora,
        "qk_nope_head_dim": d_nope,
        "qk_rope_head_dim": d_rope,
        "v_head_dim": v_head,
        "rope_theta": 10000.0,
        "rope_scaling": None,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": ctx,
        "attention_bias": False,
        "torch_dtype": "float32",
    }

    def w(rows, cols):
        """投影权重：std = 1/sqrt(fan_in)，激活量级 O(1)，softmax 不退化"""
        return rng.normal(0.0, 1.0 / np.sqrt(cols), (rows, cols)).astype(np.float32)

    q_dim = heads * (d_nope + d_rope)
    W = {}
    if q_lora:
        W["q_a"] = w(q_lora, hidden)
        W["q_an"] = (1.0 + 0.1 * rng.standard_normal(q_lora)).astype(np.float32)  # 非平凡 RMSNorm
        W["q_b"] = w(q_dim, q_lora)
        names = {"q_a": "q_a_proj.weight", "q_an": "q_a_layernorm.weight", "q_b": "q_b_proj.weight"}
    else:
        W["q"] = w(q_dim, hidden)
        names = {"q": "q_proj.weight"}
    W["kv_a"] = w(kv_lora + d_rope, hidden)
    W["kv_an"] = (1.0 + 0.1 * rng.standard_normal(kv_lora)).astype(np.float32)
    W["kv_b"] = w(heads * (d_nope + v_head), kv_lora)
    W["o"] = w(hidden, heads * v_head)
    names.update({"kv_a": "kv_a_proj_with_mqa.weight", "kv_an": "kv_a_layernorm.weight",
                  "kv_b": "kv_b_proj.weight", "o": "o_proj.weight"})

    tensors = {}
    for k, nm in names.items():
        tensors[f"model.layers.0.self_attn.{nm}"] = W[k]
    save_safetensors(os.path.join(out, "model.safetensors"), tensors)
    with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1)

    x = rng.normal(0.0, 1.0, (n_pos, hidden)).astype(np.float32)
    out_ref, cache_ref = ref_forward(x, W, cfg)

    with open(os.path.join(out, "mla_ref.json"), "w", encoding="utf-8") as f:
        f.write("{\n")
        f.write(f' "n_pos": {n_pos},\n')
        f.write(f' "hidden": {hidden},\n')
        f.write(f' "stride": {kv_lora + d_rope},\n')
        f.write(f' "q_lora_rank": {q_lora},\n')
        f.write(f' "x": {mat_json(x)},\n')
        f.write(f' "out": {mat_json(out_ref)},\n')
        f.write(f' "cache": {mat_json(cache_ref)}\n')
        f.write("}\n")

    print(f"saved random MLA -> {out} (hidden={hidden} heads={heads} q_lora={q_lora} "
          f"kv_lora={kv_lora} nope={d_nope} rope={d_rope} v={v_head} n_pos={n_pos} seed={seed})")


if __name__ == "__main__":
    main(sys.argv[1:])