"""npz → safetensors + 二进制 IO 流（绕开 C++ 侧 host 内存造 array 的全部坑）。

用法：python npz_to_st.py mini_ref.npz out_dir
产物：
  mini_ref.st     —— 全部 w_* 权重 + engram_mult_0/primes + ids + 预计算 freqs（f32/i32）
  mini_ref_io.bin —— logits_pre(f32 B*V) + logits_i(f32 B*V) × n_dec
  mini_ref_tok.bin—— tok_i(u32 B) × n_dec
  mini_ref_io.txt —— 布局说明（人读）
"""
import json
import math
import sys

import numpy as np
from safetensors.numpy import save_file

npz = np.load(sys.argv[1])
out = {}
io_f = []
io_u = []
n_dec = 0
cfg = json.loads(npz["cfg"].tobytes().decode())

for k in npz.files:
    v = npz[k]
    if k.startswith("w_"):
        out[k[2:]] = np.ascontiguousarray(v, dtype=np.float32)
    elif k == "ids":
        out["ids"] = np.ascontiguousarray(v, dtype=np.int32)
    elif k in ("engram_mult_0", "engram_primes"):
        out[k] = np.ascontiguousarray(v, dtype=np.int32)
    elif k.startswith("logits_") or k == "logits_pre":
        io_f.append(v.astype(np.float32).ravel())
    elif k.startswith("tok_"):
        io_u.append(v.astype(np.uint32).ravel())
        n_dec += 1


def precompute_freqs(dim, seqlen, osl, base, factor, bf, bs):
    i = np.arange(0, dim, 2, dtype=np.float32)
    freqs = 1.0 / (np.float32(base) ** (i / np.float32(dim)))
    if osl > 0:
        def corrected(rot):
            return dim * math.log(osl / (rot * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected(bf)), 0)
        high = min(math.ceil(corrected(bs)), dim - 1)
        ramp = np.clip((np.arange(dim // 2, dtype=np.float32) - low) / max(high - low, 1e-3), 0, 1)
        smooth = (1 - ramp).astype(np.float32)
        freqs = freqs / np.float32(factor) * (1 - smooth) + freqs * smooth
    ang = np.outer(np.arange(seqlen, dtype=np.float32), freqs).astype(np.float32)
    return np.cos(ang), np.sin(ang)


M = cfg["max_seq_len"]
wc, ws = precompute_freqs(cfg["rope_head_dim"], M, cfg["original_seq_len"], cfg["rope_theta"],
                          cfg["rope_factor"], cfg["beta_fast"], cfg["beta_slow"])
cc, cs = precompute_freqs(cfg["rope_head_dim"], M, cfg["original_seq_len"], cfg["compress_rope_theta"],
                          cfg["rope_factor"], cfg["beta_fast"], cfg["beta_slow"])
out["f_win_cos"] = np.ascontiguousarray(wc, dtype=np.float32)
out["f_win_sin"] = np.ascontiguousarray(ws, dtype=np.float32)
out["f_cmp_cos"] = np.ascontiguousarray(cc, dtype=np.float32)
out["f_cmp_sin"] = np.ascontiguousarray(cs, dtype=np.float32)

save_file(out, sys.argv[2] + "/mini_ref.st")
blob = np.concatenate(io_f) if io_f else np.array([], dtype=np.float32)
blob.tofile(sys.argv[2] + "/mini_ref_io.bin")
toks = np.concatenate(io_u) if io_u else np.array([], dtype=np.uint32)
toks.tofile(sys.argv[2] + "/mini_ref_tok.bin")
with open(sys.argv[2] + "/mini_ref_io.txt", "w") as f:
    f.write(f"V={cfg['vocab']} B=2 n_dec={n_dec}\n")
    f.write("io.bin: logits_pre(f32 B*V) + logits_i(f32 B*V) * n_dec\n")
    f.write("tok.bin: tok_i(u32 B) * n_dec\n")
print(f"ok: {len(out)} st arrays, {n_dec} dec steps, logits blob {blob.size} f32, toks {toks.size}")
