"""生成随机 MTP 草稿头（mtp.safetensors）——G22b 混跑投机的机制验证用。

权重随机（固定 seed）→ 草稿质量≈0 → 拒绝采样应拒掉一切草稿 →
投机输出必须与不投机完全一致（这是 tools/spec_serve_check.py 断言的不变量；
形状必须与 src/cuda_backend.cu 的 mtp_get_* 期望逐一对上，错了 mtp_load 会点名）。

用法：python tests/gen_mtp_head.py [model_dir=models/qwen3next-rand]
"""
import json
import struct
import sys
import os

import numpy as np


def load_dims(cfg):
    E = cfg["hidden_size"]
    I = cfg["intermediate_size"]
    H = cfg["num_attention_heads"]
    KVH = cfg.get("num_key_value_heads", H)
    D = cfg.get("head_dim", E // H)
    return E, I, H * D, KVH * D, D  # E, I, QD, KD, D


def write_st(path, tensors):
    header = {}
    off = 0
    for k, v in tensors.items():
        b = v.tobytes()
        header[k] = {"dtype": "F32", "shape": list(v.shape), "data_offsets": [off, off + len(b)]}
        off += len(b)
    hb = json.dumps(header).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for v in tensors.values():
            f.write(v.tobytes())


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "models", "qwen3next-rand")
    cfg = json.load(open(os.path.join(d, "config.json"), encoding="utf-8"))
    E, I, QD, KD, D = load_dims(cfg)
    rng = np.random.default_rng(20261105)
    t = {}
    t["mtp.fc.weight"] = rng.normal(0, 0.02, (E, 2 * E)).astype(np.float32)
    t["mtp.layers.0.self_attn.q_proj.weight"] = rng.normal(0, 0.02, (2 * QD, E)).astype(np.float32)
    t["mtp.layers.0.self_attn.k_proj.weight"] = rng.normal(0, 0.02, (KD, E)).astype(np.float32)
    t["mtp.layers.0.self_attn.v_proj.weight"] = rng.normal(0, 0.02, (KD, E)).astype(np.float32)
    t["mtp.layers.0.self_attn.o_proj.weight"] = rng.normal(0, 0.02, (E, QD)).astype(np.float32)
    t["mtp.layers.0.mlp.gate_proj.weight"] = rng.normal(0, 0.02, (I, E)).astype(np.float32)
    t["mtp.layers.0.mlp.up_proj.weight"] = rng.normal(0, 0.02, (I, E)).astype(np.float32)
    t["mtp.layers.0.mlp.down_proj.weight"] = rng.normal(0, 0.02, (E, I)).astype(np.float32)
    for n in ("mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight",
              "mtp.layers.0.input_layernorm.weight",
              "mtp.layers.0.post_attention_layernorm.weight", "mtp.norm.weight"):
        t[n] = (1.0 + 0.05 * rng.standard_normal(E)).astype(np.float32)
    for n in ("mtp.layers.0.self_attn.q_norm.weight", "mtp.layers.0.self_attn.k_norm.weight"):
        t[n] = (1.0 + 0.05 * rng.standard_normal(D)).astype(np.float32)
    out = os.path.join(d, "mtp.safetensors")
    write_st(out, t)
    print(f"mtp.safetensors -> {out}  (E={E} I={I} QD={QD} KD={KD} D={D}, {len(t)} tensors)")


if __name__ == "__main__":
    main()
