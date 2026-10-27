"""tests/gen_random_deepseek.py — 生成随机 DeepSeek-V2/V3 小模型 + numpy 参考前向
（tests/test_deepseek.c 用；参考与 C 零共享代码路径：参考在 numpy 里自己算，
 C 读 safetensors 走自己的 C 前向）。

公式照 HF modeling_deepseek_v2.py / modeling_deepseek_v3.py：
  embed → L × DecoderLayer → model.norm → lm_head
  DecoderLayer: x += Attn(RMSNorm_in(x)); x += MLP(RMSNorm_post(x))
  Attn = MLA（非吸收式：显式算 q_nope/q_pe/k_nope/k_pe/v；C 侧是吸收式，数学等价）：
    q = q_b_proj(q_a_layernorm(q_a_proj(x)))            （q_lora_rank 为 null 时 = q_proj(x)）
    kv_a = kv_a_proj_with_mqa(x) → [c | k_pe]；c = kv_a_layernorm(c)
    kv = kv_b_proj(c) → 每头 [k_nope | v]
    RoPE（非交错 rotate_half）只打在 q_pe（每头）与 k_pe（单份共享）
    score = (q_nope·k_nope + q_pe·k_pe) * (qk_nope+qk_rope)**-0.5，causal softmax，乘 v
    out = o_proj(concat_h(...))
  MLP = dense（layer < first_k_dense_replace）或 MoE：
    gate = x @ router_w.T；scores = softmax(gate) 或 sigmoid(gate)
    scores_for_choice = scores + e_score_correction_bias（有则加）
    group_limited_greedy：每 group 取 top-2 求和为组分数 → 选 topk_group 个组 → 组外 -inf
    top-k 专家 = topk(scores_for_choice)；权重 = scores[topk]（未修正，HF/官方写法），
    norm_topk_prob 时 /(sum+1e-20)，再 × routed_scaling_factor
    out = Σ w_e · expert_e(x) + shared_experts(x)（共享专家无门控）
参考全程 float64（C 是 f32），所以差异只剩 C 的舍入。

输出（<out>/）：
  config.json / model.safetensors（F32，HF DeepSeek 命名）
  deepseek_ref.json：
    tokens[n]                测试 prompt
    logits_all[n][V]         全序列 prefill 的每个位置分布
    logits_last[V]           末位分布
    cache_l0[n][stride]      layer0 的压缩 KV 行（[latent | k_pe]，逐位置）

用法：
  python tests/gen_random_deepseek.py --out ../models/deepseek-rand --seed 7
  python tests/gen_random_deepseek.py --out ../models/deepseek-rand-v3 --style v3 --seed 11
    --style v2（默认）：softmax 打分 + greedy + norm_topk_prob=false
    --style v3：sigmoid 打分 + group_limited_greedy + e_score_correction_bias +
                norm_topk_prob=true + routed_scaling_factor=2.5
"""
import json
import os
import struct
import sys

import numpy as np


def save_safetensors(path, tensors):
    hdr, off = {}, 0
    for k, v in tensors.items():
        b = np.ascontiguousarray(v, dtype=np.float32).tobytes()
        hdr[k] = {"dtype": "F32", "shape": list(np.shape(v)), "data_offsets": [off, off + len(b)]}
        off += len(b)
    hb = json.dumps(hdr).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for v in tensors.values():
            f.write(np.ascontiguousarray(v, dtype=np.float32).tobytes())


def rmsnorm(x, w, eps):
    """HF DeepseekV2RMSNorm：x / sqrt(mean(x^2)+eps) * w"""
    x = np.asarray(x, dtype=np.float64)
    return x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps) * np.asarray(w, dtype=np.float64)


def rope_rotate_half(x, pos, theta, dim, inv=None, scale=1.0):
    """HF 非交错 rotate_half：最后 dim 维配对 (i, i+dim/2)。x: [n, dim]
    inv / scale 非空时用于 YaRN（逆频率表 + mscale 乘到 cos/sin 上）。"""
    half = dim // 2
    if inv is None:
        inv = theta ** (-2.0 * np.arange(half, dtype=np.float64) / dim)
    ang = np.asarray(pos, dtype=np.float64)[:, None] * np.asarray(inv, dtype=np.float64)[None, :]
    c, s = np.cos(ang) * scale, np.sin(ang) * scale
    x1, x2 = x[:, :half], x[:, half:]
    return np.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], axis=-1)


def rope_interleaved(x, pos, theta, dim, inv=None, scale=1.0):
    """HF V3 config.rope_interleave：最后 dim 维配对 (2i, 2i+1)"""
    half = dim // 2
    if inv is None:
        inv = theta ** (-2.0 * np.arange(half, dtype=np.float64) / dim)
    ang = np.asarray(pos, dtype=np.float64)[:, None] * np.asarray(inv, dtype=np.float64)[None, :]
    c, s = np.cos(ang) * scale, np.sin(ang) * scale
    x1, x2 = x[:, 0::2], x[:, 1::2]
    return np.stack([x1 * c - x2 * s, x1 * s + x2 * c], axis=-1).reshape(x.shape)


def silu(x):
    return x / (1.0 + np.exp(-x))


def route(x, moe, cfg):
    """HF DeepseekV2/V3 TopkRouter.forward 的 numpy 版 → ids[n,k], w[n,k]"""
    n = x.shape[0]
    NE, k = cfg["n_routed_experts"], cfg["num_experts_per_tok"]
    logits = x @ np.asarray(moe["router"], dtype=np.float64).T
    if cfg["scoring_func"] == "sigmoid":
        scores = 1.0 / (1.0 + np.exp(-logits))
    else:
        z = logits - logits.max(axis=-1, keepdims=True)
        e = np.exp(z)
        scores = e / e.sum(axis=-1, keepdims=True)
    bias = moe.get("ec_bias")
    choice = scores + (np.asarray(bias, dtype=np.float64)[None, :] if bias is not None else 0.0)
    if cfg["topk_method"] == "group_limited_greedy":
        ng, tg = cfg["n_group"], cfg["topk_group"]
        epg = NE // ng
        gs = np.sort(choice.reshape(n, ng, epg), axis=-1)[..., -2:].sum(axis=-1)  # 组内 top-2 之和
        keep = np.zeros((n, ng), dtype=bool)
        np.put_along_axis(keep, np.argsort(-gs, axis=-1)[:, :tg], True, axis=-1)
        choice = np.where(np.repeat(keep, epg, axis=-1), choice, -np.inf)
    ids = np.argsort(-choice, axis=-1)[:, :k]
    w = np.take_along_axis(scores, ids, axis=-1)  # 未修正打分（HF: scores.gather）
    if k > 1 and cfg["norm_topk_prob"]:
        w = w / (w.sum(axis=-1, keepdims=True) + 1e-20)
    return ids, w * cfg["routed_scaling_factor"]


def mlp(x, p):
    return silu(x @ np.asarray(p["gate"], dtype=np.float64).T) * (x @ np.asarray(p["up"], dtype=np.float64).T) @ np.asarray(p["down"], dtype=np.float64).T


def yarn_table(cfg):
    """YaRN 逆频率表 + attention factor，直接取 transformers 的
    `_compute_yarn_parameters`（金标准，不自己重写公式，避免「参考和被测抄同一份错」。
    返回 (inv_freq[Dr/2], attn_factor) 或 (None, 1.0)。"""
    rs = cfg.get("rope_scaling")
    if not rs or (rs.get("type") or rs.get("rope_type")) != "yarn":
        return None, 1.0
    import math
    from types import SimpleNamespace
    from transformers.modeling_rope_utils import _compute_yarn_parameters

    Dr = cfg["qk_rope_head_dim"]
    rp = dict(rs)
    rp["rope_theta"] = cfg["rope_theta"]
    rp["rope_type"] = "yarn"
    orig = rp.get("original_max_position_embeddings") or cfg["max_position_embeddings"]
    rp["original_max_position_embeddings"] = orig
    if "factor" not in rp:
        rp["factor"] = cfg["max_position_embeddings"] / orig
    o = SimpleNamespace(
        head_dim=Dr, hidden_size=Dr, num_attention_heads=1,
        max_position_embeddings=cfg["max_position_embeddings"],
        rope_parameters=rp, rope_scaling=None,
        standardize_rope_params=lambda: None, per_layer_config={},
    )
    inv, attn = _compute_yarn_parameters(o)
    return np.asarray(inv.tolist(), dtype=np.float64), float(attn)


def ref_forward(tokens, P, cfg):
    L = cfg["num_hidden_layers"]
    H = cfg["num_attention_heads"]
    Dn, Dr, Dc, Dv = (cfg["qk_nope_head_dim"], cfg["qk_rope_head_dim"], cfg["kv_lora_rank"],
                      cfg["v_head_dim"])
    eps, theta, qlr = cfg["rms_norm_eps"], cfg["rope_theta"], cfg["q_lora_rank"]
    n = len(tokens)
    pos = np.arange(n)
    scale = (Dn + Dr) ** -0.5
    mask = np.triu(np.ones((n, n), dtype=bool), 1)
    rope_fn = rope_interleaved if cfg.get("rope_interleave") else rope_rotate_half
    yinv, yscale = yarn_table(cfg)
    # DeepSeek 的 mscale 乘在 softmax_scale 上（HF DeepseekV2Attention.scaling），
    # 不是乘到 cos/sin；yscale 只给 llama 系用，这里丢掉
    rs = cfg.get("rope_scaling") or {}
    mscale_all = rs.get("mscale_all_dim") or 0.0
    if (rs.get("type") or rs.get("rope_type")) == "yarn" and mscale_all:
        mc = 0.1 * mscale_all * np.log(rs.get("factor", 1.0)) + 1.0
        scale = scale * mc * mc

    def rope(xx, pp):
        return rope_fn(xx, pp, theta, Dr, yinv, 1.0)

    x = np.asarray(P["embed"][tokens], dtype=np.float64)
    cache_l0 = None
    for l in range(L):
        pl = P["layers"][l]
        h = rmsnorm(x, pl["in_norm"], eps)                       # MLA
        if qlr:
            q = rmsnorm(h @ pl["q_a"].T, pl["q_a_norm"], eps) @ pl["q_b"].T
        else:
            q = h @ pl["q_b"].T
        q = q.reshape(n, H, Dn + Dr)
        q_nope, q_pe = q[..., :Dn], q[..., Dn:]
        kva = h @ pl["kv_a"].T
        c = rmsnorm(kva[:, :Dc], pl["kv_a_norm"], eps)
        k_pe = kva[:, Dc:]
        kv = (c @ pl["kv_b"].T).reshape(n, H, Dn + Dv)
        k_nope, v = kv[..., :Dn], kv[..., Dn:]
        q_pe_r = rope(q_pe.reshape(n * H, Dr), np.repeat(pos, H)).reshape(n, H, Dr)
        k_pe_r = rope(k_pe, pos)
        if l == 0:
            cache_l0 = np.concatenate([c, k_pe_r], axis=-1)
        out = np.empty((n, H, Dv), dtype=np.float64)
        for hh in range(H):
            s = (q_nope[:, hh, :] @ k_nope[:, hh, :].T + q_pe_r[:, hh, :] @ k_pe_r.T) * scale
            s = np.where(mask, -np.inf, s)
            s = s - s.max(axis=-1, keepdims=True)
            a = np.exp(s)
            out[:, hh, :] = (a / a.sum(axis=-1, keepdims=True)) @ v[:, hh, :]
        x = x + out.reshape(n, H * Dv) @ pl["o"].T
        h = rmsnorm(x, pl["post_norm"], eps)                     # MLP
        if cfg["n_routed_experts"] and l >= cfg["first_k_dense_replace"] and l % cfg.get("moe_layer_freq", 1) == 0:
            moe = pl["moe"]
            ids, w = route(h, moe, cfg)
            acc = np.zeros_like(h)
            for e in range(cfg["n_routed_experts"]):
                for j in range(cfg["num_experts_per_tok"]):
                    sel = ids[:, j] == e
                    if sel.any():
                        acc[sel] += w[sel, j][:, None] * mlp(h[sel], moe["experts"][e])
            x = x + acc + mlp(h, moe["shared"])
        else:
            x = x + mlp(h, pl["dense"])
    return rmsnorm(x, P["final_norm"], eps) @ P["lm_head"].T, cache_l0


def main(argv):
    out, seed, style, n_pos = "../models/deepseek-rand", 7, "v2", 9
    hidden, heads, kv_lora, d_nope, d_rope, v_head = 32, 4, 16, 8, 4, 6
    q_lora, n_layers, n_experts, topk, moe_inter, n_shared = 12, 3, 4, 2, 12, 1
    first_k_dense, vocab, ctx, n_inter = 1, 24, 64, 24
    # DeepSeek 的 RoPE 默认是【交错】配对 (2i, 2i+1)（HF apply_rotary_emb 的 view_as_complex），
    # 与官方 inference 代码一致；--interleave 0 生成非交错 rotate_half 的兼容性用例。
    stacked, interleave = False, True
    yarn = False
    yarn_factor, yarn_orig = 4.0, 16
    i = 0
    while i < len(argv):
        a, v = argv[i], argv[i + 1]
        if a in ("--out", "-o"):
            out = v
        elif a == "--seed":
            seed = int(v)
        elif a == "--style":
            style = v
        elif a == "--n-pos":
            n_pos = int(v)
        elif a == "--q-lora":
            q_lora = int(v)
        elif a == "--layers":
            n_layers = int(v)
        elif a == "--hidden":
            hidden = int(v)
        elif a == "--heads":
            heads = int(v)
        elif a == "--experts":
            stacked = (v == "stacked")   # HF V3 三维融合布局 gate_up_proj / down_proj
        elif a == "--interleave":
            interleave = (v == "1")      # 1 = 交错（DeepSeek 默认），0 = 非交错 rotate_half
        elif a == "--yarn":
            yarn = (v == "1")            # config.rope_scaling = yarn（DeepSeek-V3）
        elif a == "--yarn-factor":
            yarn_factor = float(v)
        elif a == "--yarn-orig":
            yarn_orig = int(v)
        else:
            raise SystemExit("unknown arg: " + a)
        i += 2

    print(f"style={style} seed={seed} out={out} layers={n_layers} experts={n_experts} q_lora={q_lora}")
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(seed)
    if style == "v3":
        scoring_func, topk_method, n_group, topk_group = "sigmoid", "group_limited_greedy", 2, 1
        norm_topk_prob, routed_scale = True, 2.5
    else:
        scoring_func, topk_method, n_group, topk_group = "softmax", "greedy", 0, 0
        norm_topk_prob, routed_scale = False, 1.0
    cfg = {
        "architectures": ["DeepseekV2ForCausalLM" if style == "v2" else "DeepseekV3ForCausalLM"],
        "model_type": "deepseek_v2" if style == "v2" else "deepseek_v3",
        "hidden_size": hidden, "num_hidden_layers": n_layers, "num_attention_heads": heads,
        "q_lora_rank": (q_lora if q_lora > 0 else None), "kv_lora_rank": kv_lora,
        "qk_nope_head_dim": d_nope, "qk_rope_head_dim": d_rope, "v_head_dim": v_head,
        "rope_theta": 10000.0,
        "rope_scaling": ({"type": "yarn", "factor": yarn_factor,
                          "original_max_position_embeddings": yarn_orig,
                          "beta_fast": 32, "beta_slow": 1,
                          "mscale": 1.0, "mscale_all_dim": 1.0} if yarn else None),
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": ctx, "vocab_size": vocab, "intermediate_size": n_inter,
        "moe_intermediate_size": moe_inter, "n_routed_experts": n_experts,
        "n_shared_experts": n_shared, "num_experts_per_tok": topk,
        "first_k_dense_replace": first_k_dense, "moe_layer_freq": 1,
        "norm_topk_prob": norm_topk_prob, "routed_scaling_factor": routed_scale,
        "scoring_func": scoring_func, "topk_method": topk_method,
        "n_group": n_group, "topk_group": topk_group,
        "attention_bias": False, "tie_word_embeddings": False, "hidden_act": "silu",
        "torch_dtype": "float32",
    }
    # 总是显式写出（True = 交错，DeepSeek 默认；False = 非交错旧测试模型），
    # 这样 C 侧不会因为缺字段而套默认值
    cfg["rope_interleave"] = bool(interleave)

    def w(rows, cols):
        return rng.normal(0.0, 1.0 / np.sqrt(cols), (rows, cols)).astype(np.float32)

    def nrm(n):
        return (1.0 + 0.1 * rng.standard_normal(n)).astype(np.float32)

    qhd = d_nope + d_rope
    tensors, P = {}, {}
    tensors["model.embed_tokens.weight"] = rng.normal(0.0, 0.3, (vocab, hidden)).astype(np.float32)
    tensors["model.norm.weight"] = nrm(hidden)
    tensors["lm_head.weight"] = w(vocab, hidden)
    P["embed"] = tensors["model.embed_tokens.weight"]
    P["final_norm"] = tensors["model.norm.weight"]
    P["lm_head"] = tensors["lm_head.weight"]
    P["layers"] = []
    for l in range(n_layers):
        pfx = f"model.layers.{l}."
        ap = pfx + "self_attn."
        pl = {"in_norm": nrm(hidden), "post_norm": nrm(hidden)}
        tensors[pfx + "input_layernorm.weight"] = pl["in_norm"]
        tensors[pfx + "post_attention_layernorm.weight"] = pl["post_norm"]
        if q_lora:
            pl["q_a"] = w(q_lora, hidden)
            pl["q_a_norm"] = nrm(q_lora)
            pl["q_b"] = w(heads * qhd, q_lora)
            tensors[ap + "q_a_proj.weight"] = pl["q_a"]
            tensors[ap + "q_a_layernorm.weight"] = pl["q_a_norm"]
            tensors[ap + "q_b_proj.weight"] = pl["q_b"]
        else:
            pl["q_b"] = w(heads * qhd, hidden)
            tensors[ap + "q_proj.weight"] = pl["q_b"]
        pl["kv_a"] = w(kv_lora + d_rope, hidden)
        pl["kv_a_norm"] = nrm(kv_lora)
        pl["kv_b"] = w(heads * (d_nope + v_head), kv_lora)
        pl["o"] = w(hidden, heads * v_head)
        tensors[ap + "kv_a_proj_with_mqa.weight"] = pl["kv_a"]
        tensors[ap + "kv_a_layernorm.weight"] = pl["kv_a_norm"]
        tensors[ap + "kv_b_proj.weight"] = pl["kv_b"]
        tensors[ap + "o_proj.weight"] = pl["o"]
        if n_experts and l >= first_k_dense and l % cfg["moe_layer_freq"] == 0:
            moe = {"experts": []}
            moe["router"] = w(n_experts, hidden)
            tensors[pfx + "mlp.gate.weight"] = moe["router"]
            if style == "v3":
                moe["ec_bias"] = (0.1 * rng.standard_normal(n_experts)).astype(np.float32)
                tensors[pfx + "mlp.gate.e_score_correction_bias"] = moe["ec_bias"]
            for e in range(n_experts):
                ex = {"gate": w(moe_inter, hidden), "up": w(moe_inter, hidden),
                      "down": w(hidden, moe_inter)}
                moe["experts"].append(ex)
            if stacked:
                tensors[pfx + "mlp.experts.gate_up_proj.weight"] = np.stack(
                    [np.concatenate([ex["gate"], ex["up"]], axis=0) for ex in moe["experts"]])
                tensors[pfx + "mlp.experts.down_proj.weight"] = np.stack(
                    [ex["down"] for ex in moe["experts"]])
            else:
                for e, ex in enumerate(moe["experts"]):
                    for nm, k in (("gate_proj", "gate"), ("up_proj", "up"), ("down_proj", "down")):
                        tensors[f"{pfx}mlp.experts.{e}.{nm}.weight"] = ex[k]
            sh = moe_inter * n_shared
            moe["shared"] = {"gate": w(sh, hidden), "up": w(sh, hidden), "down": w(hidden, sh)}
            for nm, k in (("gate_proj", "gate"), ("up_proj", "up"), ("down_proj", "down")):
                tensors[pfx + f"mlp.shared_experts.{nm}.weight"] = moe["shared"][k]
            pl["moe"] = moe
        else:
            pl["dense"] = {"gate": w(n_inter, hidden), "up": w(n_inter, hidden),
                           "down": w(hidden, n_inter)}
            for nm in ("gate_proj", "up_proj", "down_proj"):
                tensors[pfx + f"mlp.{nm}.weight"] = pl["dense"][nm.split("_")[0]]
        P["layers"].append(pl)
    # numpy 参考要把 float32 权重转成 float64 用；上面 P 里存的就是 float32 张量，参考里显式转
    save_safetensors(os.path.join(out, "model.safetensors"), tensors)
    with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    tokens = [int(t) for t in rng.integers(1, vocab, n_pos)]
    logits, cache_l0 = ref_forward(tokens, P, cfg)
    ref = {"meta": {"n_pos": n_pos, "vocab": vocab, "hidden": hidden,
                    "stride": kv_lora + d_rope, "n_layer": n_layers, "style": style},
           "tokens": tokens, "logits_all": logits.tolist(), "logits_last": logits[-1].tolist(),
           "cache_l0": np.asarray(cache_l0).tolist()}
    with open(os.path.join(out, "deepseek_ref.json"), "w", encoding="utf-8") as f:
        json.dump(ref, f)
    print("wrote", out, "tokens=", tokens)
    print("logits_last[0:4]=", [round(float(v), 6) for v in logits[-1][:4]])


if __name__ == "__main__":
    main(sys.argv[1:])