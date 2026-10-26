"""t10: 随机权重 LLaVA 结构模型（小 CLIP + 单层 projector），验证多模态融合路径。
权重布局/语义与真实 HF CLIPVisionModel 一致（G8）：双向 attention、learnable pos emb、
pre_layrnorm 带仿射、quick_gelu MLP、post_layernorm 只作用 pooled cls、encode 输出 patch 行。
用法: python gen_random_llava.py <out_dir>
产物: clip_config.json + clip_model.safetensors + projector.safetensors + image.raw"""
import json, os, struct, sys
import numpy as np

out = sys.argv[1] if len(sys.argv) > 1 else "models/llava-rand"
E, H, L, P, IMG, I, PROJ = 64, 4, 2, 8, 32, 128, 32   # n_patch=16
rng = np.random.default_rng(7)
os.makedirs(out, exist_ok=True)

json.dump({"hidden_size": E, "num_heads": H, "num_hidden_layers": L, "patch_size": P,
           "image_size": IMG, "intermediate_size": I, "projection_dim": PROJ,
           "layer_norm_eps": 1e-5, "hidden_act": "quick_gelu"},
          open(f"{out}/clip_config.json", "w"), indent=1)

def r(*shape, s=0.02):
    return rng.normal(0, s, shape).astype(np.float32)

NP = (IMG // P) ** 2
W = {
    "vision_model.embeddings.class_embedding": r(E),
    "vision_model.embeddings.position_embedding": r(NP + 1, E),
    "vision_model.embeddings.patch_embedding.weight": r(E, 3, P, P),
    "vision_model.embeddings.patch_embedding.bias": r(E, s=0.001),
    "vision_model.embeddings.pre_layernorm.weight": np.ones(E, dtype=np.float32),
    "vision_model.embeddings.pre_layernorm.bias": np.zeros(E, dtype=np.float32),
    "vision_model.post_layernorm.weight": np.ones(E, dtype=np.float32),
    "vision_model.post_layernorm.bias": np.zeros(E, dtype=np.float32),
    "visual_projection.weight": r(PROJ, E),
}
for l in range(L):
    p = f"vision_model.encoder.layers.{l}"
    W[f"{p}.layer_norm1.weight"] = np.ones(E, dtype=np.float32)
    W[f"{p}.layer_norm1.bias"] = np.zeros(E, dtype=np.float32)
    W[f"{p}.layer_norm2.weight"] = np.ones(E, dtype=np.float32)
    W[f"{p}.layer_norm2.bias"] = np.zeros(E, dtype=np.float32)
    W[f"{p}.self_attn.q_proj.weight"] = r(E, E); W[f"{p}.self_attn.q_proj.bias"] = r(E, s=0.001)
    W[f"{p}.self_attn.k_proj.weight"] = r(E, E); W[f"{p}.self_attn.k_proj.bias"] = r(E, s=0.001)
    W[f"{p}.self_attn.v_proj.weight"] = r(E, E); W[f"{p}.self_attn.v_proj.bias"] = r(E, s=0.001)
    W[f"{p}.self_attn.out_proj.weight"] = r(E, E); W[f"{p}.self_attn.out_proj.bias"] = r(E, s=0.001)
    W[f"{p}.mlp.fc1.weight"] = r(I, E); W[f"{p}.mlp.fc1.bias"] = r(I, s=0.001)
    W[f"{p}.mlp.fc2.weight"] = r(E, I); W[f"{p}.mlp.fc2.bias"] = r(E, s=0.001)

def save_st(path, tensors):
    hdr, off = {}, 0
    for k, v in tensors.items():
        b = v.tobytes()
        hdr[k] = {"dtype": "F32", "shape": list(v.shape), "data_offsets": [off, off + len(b)]}
        off += len(b)
    hb = json.dumps(hdr).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb))); f.write(hb)
        for v in tensors.values():
            f.write(v.tobytes())

save_st(f"{out}/clip_model.safetensors", W)
# mm_projector: single linear E -> LM_E（真实语义：视觉塔输出 [n_patch][hidden_size]，
# visual_projection 只作用于 pooled cls，不作用于 patch 行；G8 对齐）
# 真实 llava-1.5 是 1024->4096 两层 MLP，这里单层。
LM_E = 64
save_st(f"{out}/projector.safetensors",
        {"linear_1.weight": r(LM_E, E), "linear_1.bias": r(LM_E, s=0.001)})

# 固定测试图（CHW）
img = rng.normal(0, 1, (3, IMG, IMG)).astype(np.float32)
img.tofile(f"{out}/image.raw")
print(f"saved random llava -> {out} (CLIP E={E} L={L} patch={P} img={IMG} NP={NP} proj={PROJ})")
