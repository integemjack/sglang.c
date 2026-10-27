"""MLX 版 CLIP+projector 参考（在 Apple Silicon 上用 MLX/Metal 计算）。

与 tests/ref_llava.py 数学等价、实现独立：前者用 numpy（CPU fp32/fp64 混合），这里全部
用 mlx.core（默认 GPU）。产出同名 `_ref_proj.npy` / `_ref_feats.npy` 覆盖掉 numpy 版，
于是 `test_llava` 的 maxerr 就变成「C 实现 vs **MLX** 参考」——这就是「Mac 上用 MLX 测」。

用法: python ref_llava_mlx.py <llava_dir> <out.json>
"""
import json
import sys

import mlx.core as mx
import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else "models/llava-rand"
OUT = sys.argv[2] if len(sys.argv) > 2 else "ref_llava_mlx.json"
cfg = json.load(open(f"{D}/clip_config.json"))
E, H, L, P, IMG, I, PROJ = (cfg[k] for k in
                            ("hidden_size", "num_heads", "num_hidden_layers", "patch_size",
                             "image_size", "intermediate_size", "projection_dim"))
EPS = float(cfg.get("layer_norm_eps", 1e-5))
ACT = cfg.get("hidden_act", "quick_gelu")
NP = (IMG // P) ** 2


def load_st(path):
    """safetensors → {name: np.ndarray}（只读文件用 numpy，计算不经过它）"""
    b = open(path, "rb").read()
    n = int.from_bytes(b[:8], "little")
    hdr = json.loads(b[8:8 + n])
    data = b[8 + n:]
    out = {}
    for k, m in hdr.items():
        s, e = m["data_offsets"]
        out[k] = np.frombuffer(data[s:e], dtype=np.float32).reshape(m["shape"]).copy()
    return out


Wnp = load_st(f"{D}/clip_model.safetensors")
PJnp = load_st(f"{D}/projector.safetensors")
W = {k: mx.array(v) for k, v in Wnp.items()}
PJ = {k: mx.array(v) for k, v in PJnp.items()}
img = mx.array(np.fromfile(f"{D}/image.raw", dtype=np.float32).reshape(3, IMG, IMG))


def ln(x, w, b):
    m = mx.mean(x, axis=-1, keepdims=True)
    d = x - m
    v = mx.mean(d * d, axis=-1, keepdims=True)
    return d / mx.sqrt(v + EPS) * w + b


def act_fn(x):
    if ACT == "quick_gelu":
        return x * (1.0 / (1.0 + mx.exp(-1.702 * x)))
    # gelu（erf 版），与 torch.nn.functional.gelu 默认一致
    return 0.5 * x * (1.0 + mx.erf(x / mx.sqrt(mx.array(2.0))))


def attn(q, k, v):  # [n,H,D] 双向注意力
    n, Hd, d = q.shape
    s = mx.matmul(mx.transpose(q, (1, 0, 2)), mx.transpose(k, (1, 2, 0))) / mx.sqrt(mx.array(float(d)))
    s = s - mx.max(s, axis=-1, keepdims=True)
    a = mx.exp(s)
    a = a / mx.sum(a, axis=-1, keepdims=True)
    o = mx.matmul(a, mx.transpose(v, (1, 0, 2)))       # [H,n,D]
    return mx.reshape(mx.transpose(o, (1, 0, 2)), (n, Hd * d))


def lin(x, w, b):
    return mx.matmul(x, mx.transpose(w)) + b


# patch embedding（conv 按 patch 摊成一次 matmul）
side = IMG // P
pw = mx.reshape(W["vision_model.embeddings.patch_embedding.weight"], (E, 3 * P * P))
pb = W["vision_model.embeddings.patch_embedding.bias"]
rows = []
for py in range(side):
    for px in range(side):
        patch = mx.reshape(img[:, py * P:py * P + P, px * P:px * P + P], (3 * P * P,))
        rows.append(lin(patch, pw, pb)[None, :])
x = mx.concatenate(rows, axis=0)                        # [NP, E]
cls = W["vision_model.embeddings.class_embedding"]
x = mx.concatenate([mx.reshape(cls, (1, E)), x], axis=0)  # [NP+1, E]
pos = W.get("vision_model.embeddings.position_embedding.weight",
            W.get("vision_model.embeddings.position_embedding"))
x = x + pos
pre_w = W.get("vision_model.pre_layrnorm.weight",
              W.get("vision_model.embeddings.pre_layernorm.weight"))
pre_b = W.get("vision_model.pre_layrnorm.bias",
              W.get("vision_model.embeddings.pre_layernorm.bias"))
x = ln(x, pre_w, pre_b)

for l in range(L):
    p = f"vision_model.encoder.layers.{l}"
    h = ln(x, W[f"{p}.layer_norm1.weight"], W[f"{p}.layer_norm1.bias"])
    q = mx.reshape(lin(h, W[f"{p}.self_attn.q_proj.weight"], W[f"{p}.self_attn.q_proj.bias"]), (NP + 1, H, -1))
    k = mx.reshape(lin(h, W[f"{p}.self_attn.k_proj.weight"], W[f"{p}.self_attn.k_proj.bias"]), (NP + 1, H, -1))
    v = mx.reshape(lin(h, W[f"{p}.self_attn.v_proj.weight"], W[f"{p}.self_attn.v_proj.bias"]), (NP + 1, H, -1))
    o = lin(attn(q, k, v), W[f"{p}.self_attn.out_proj.weight"], W[f"{p}.self_attn.out_proj.bias"])
    x = x + o
    h = ln(x, W[f"{p}.layer_norm2.weight"], W[f"{p}.layer_norm2.bias"])
    mid = act_fn(lin(h, W[f"{p}.mlp.fc1.weight"], W[f"{p}.mlp.fc1.bias"]))
    x = x + lin(mid, W[f"{p}.mlp.fc2.weight"], W[f"{p}.mlp.fc2.bias"])

feats = x[1:]                                            # 不过 post_layernorm
pooled = ln(x[0], W["vision_model.post_layernorm.weight"],
            W["vision_model.post_layernorm.bias"])
img_emb = mx.matmul(pooled, mx.transpose(W["visual_projection.weight"]))
proj = lin(feats, PJ["linear_1.weight"], PJ["linear_1.bias"])
if "linear_2.weight" in PJ:
    proj = lin(act_fn(proj) if ACT != "quick_gelu" else
               0.5 * proj * (1.0 + mx.erf(proj / mx.sqrt(mx.array(2.0)))),
               PJ["linear_2.weight"], PJ["linear_2.bias"])

feats_np = np.array(feats, dtype=np.float32)
proj_np = np.array(proj, dtype=np.float32)
out = {"n_patch": NP, "feat_dim": E, "proj_dim": E,
       "backend": f"mlx ({mx.default_device()})",
       "projector": ("mlp" if "linear_2.weight" in PJ else "linear"),
       "feats_mean": float(feats_np.mean()),
       "proj_first": proj_np[0][:8].tolist(), "proj_last": proj_np[-1][:8].tolist(),
       "proj_checksum": float(proj_np.astype(np.float64).sum())}
json.dump(out, open(OUT, "w"), indent=1)
np.save(f"{D}/_ref_proj.npy", proj_np)
np.save(f"{D}/_ref_feats.npy", feats_np)
np.save(f"{D}/_ref_imgemb.npy", np.array(img_emb, dtype=np.float32))
print(f"MLX({mx.default_device()}) -> {OUT}: NP={NP} feat_dim={E} "
      f"proj_checksum={out['proj_checksum']:.6f}")