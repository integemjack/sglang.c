"""t10 numpy 参考：CLIP encode + projector，与 clip.c 逐位对齐的独立实现。
用法: python ref_llava.py <llava_dir> <out.json>"""
import json, sys
import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else "models/llava-rand"
OUT = sys.argv[2] if len(sys.argv) > 2 else "ref_llava.json"
cfg = json.load(open(f"{D}/clip_config.json"))
E, H, L, P, IMG, I, PROJ = (cfg[k] for k in
    ("hidden_size", "num_heads", "num_hidden_layers", "patch_size",
     "image_size", "intermediate_size", "projection_dim"))
NP = (IMG // P) ** 2

def load_st(path):
    b = open(path, "rb").read()
    n = int.from_bytes(b[:8], "little")
    hdr = json.loads(b[8:8 + n])
    data = b[8 + n:]
    out = {}
    for k, m in hdr.items():
        s, e = m["data_offsets"]
        out[k] = np.frombuffer(data[s:e], dtype=np.float32).reshape(m["shape"]).copy()
    return out

W = load_st(f"{D}/clip_model.safetensors")
PJ = load_st(f"{D}/projector.safetensors")
img = np.fromfile(f"{D}/image.raw", dtype=np.float32).reshape(3, IMG, IMG)

def ln(x, w, b):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + 1e-5) * w + b

def gelu(x):
    from math import erf
    from scipy.special import erf as serf
    return 0.5 * x * (1.0 + serf(x / np.sqrt(2.0)))

def attn(q, k, v):  # [n,H,D]
    n, Hd, d = q.shape
    s = np.einsum("qhd,khd->hqk", q, k) / np.sqrt(d)
    s = s - s.max(-1, keepdims=True)
    a = np.exp(s); a = a / a.sum(-1, keepdims=True)
    return np.einsum("hqk,khd->qhd", a, v).reshape(n, Hd * d)

x = np.zeros((NP + 1, E), dtype=np.float32)
pw = W["vision_model.embeddings.patch_embedding.weight"].reshape(E, 3 * P * P)
pb = W["vision_model.embeddings.patch_embedding.bias"]
side = IMG // P
for py in range(side):
    for px in range(side):
        pi = py * side + px
        patch = img[:, py * P:py * P + P, px * P:px * P + P].reshape(3 * P * P)
        x[pi + 1] = patch @ pw.T + pb
x[0] = W["vision_model.embeddings.class_embedding"]
x = x + W["vision_model.embeddings.position_embedding"]
x = ln(x, W["vision_model.embeddings.pre_layernorm.weight"],
       W["vision_model.embeddings.pre_layernorm.bias"])
for l in range(L):
    p = f"vision_model.encoder.layers.{l}"
    h = ln(x, W[f"{p}.layer_norm1.weight"], W[f"{p}.layer_norm1.bias"])
    q = (h @ W[f"{p}.self_attn.q_proj.weight"].T + W[f"{p}.self_attn.q_proj.bias"]).reshape(NP + 1, H, -1)
    k = (h @ W[f"{p}.self_attn.k_proj.weight"].T + W[f"{p}.self_attn.k_proj.bias"]).reshape(NP + 1, H, -1)
    v = (h @ W[f"{p}.self_attn.v_proj.weight"].T + W[f"{p}.self_attn.v_proj.bias"]).reshape(NP + 1, H, -1)
    o = attn(q, k, v) @ W[f"{p}.self_attn.out_proj.weight"].T + W[f"{p}.self_attn.out_proj.bias"]
    x = x + o
    h = ln(x, W[f"{p}.layer_norm2.weight"], W[f"{p}.layer_norm2.bias"])
    mid = gelu(h @ W[f"{p}.mlp.fc1.weight"].T + W[f"{p}.mlp.fc1.bias"])
    x = x + (mid @ W[f"{p}.mlp.fc2.weight"].T + W[f"{p}.mlp.fc2.bias"])
x = ln(x, W["vision_model.post_layernorm.weight"], W["vision_model.post_layernorm.bias"])
feats = x[1:] @ W["visual_projection.weight"].T          # [NP, PROJ]
proj = feats @ PJ["linear_1.weight"].T + PJ["linear_1.bias"]  # [NP, LM_E]

out = {"n_patch": NP, "proj_dim": PROJ,
       "feats_mean": float(feats.mean()), "feats_maxerr_base": float(np.abs(feats).mean()),
       "proj_first": proj[0][:8].tolist(), "proj_last": proj[-1][:8].tolist(),
       "proj_checksum": float(proj.sum())}
json.dump(out, open(OUT, "w"), indent=1)
np.save(f"{D}/_ref_proj.npy", proj.astype(np.float32))  # scipy erf 升维 f8，拉回 f32 与 C 对齐
print(f"saved {OUT}: NP={NP} proj_checksum={out['proj_checksum']:.6f}")
