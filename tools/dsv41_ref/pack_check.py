"""验证 ds41_conv.cpp 的 affine4 打包是否与 MLX 原生一致：同一矩阵、两种打包、逐字节对比。"""
import struct

import mlx.core as mx
import numpy as np

rng = np.random.default_rng(7)
K, N = 256, 16
w = (rng.standard_normal((N, K)) * 0.02).astype(np.float32)

# MLX 原生
wq, s, b = mx.quantize(mx.array(w), group_size=64, bits=4)
wq_np = np.array(wq.tolist()) if wq.dtype == mx.uint32 else np.frombuffer(bytes(wq), dtype=np.uint32)

# 复刻 ds41_conv.cpp 的 quantize_affine（LSB-first，step=(max-min)/15，bias=min）


def f32_to_f16_bits(v):
    x = struct.unpack("<I", struct.pack("<f", v))[0]
    sign = (x >> 16) & 0x8000
    e = ((x >> 23) & 0xFF) - 127 + 15
    man = x & 0x7FFFFF
    if ((x >> 23) & 0xFF) == 0xFF:
        return sign | 0x7C00
    if e >= 31:
        return sign | 0x7C00
    if e <= 0:
        return sign
    return sign | (e << 10) | (man >> 13)


def f16_bits_to_f32(h):
    return struct.unpack("<f", struct.pack("<I", (h << 16)))[0] if h & 0x7F80 not in (0,) else 0.0


def conv_pack(w, gs=64, bits=4, K=K, N=N):
    pf = 32 // bits
    words = K // pf
    q = np.zeros((N, words), dtype=np.uint32)
    scales = np.zeros((N, K // gs), dtype=np.uint16)
    biases = np.zeros((N, K // gs), dtype=np.uint16)
    qmax = (1 << bits) - 1
    for r in range(N):
        for g in range(K // gs):
            grp = w[r, g * gs : (g + 1) * gs]
            mn, mxx = grp.min(), grp.max()
            step = (mxx - mn) / qmax
            s16 = np.float16(step).view(np.uint16) if False else f32_to_f16_bits(step)
            b16 = f32_to_f16_bits(mn)
            sf, bf = f16_bits_to_f32(s16), f16_bits_to_f32(b16)
            scales[r, g], biases[r, g] = s16, b16
            for j in range(gs):
                v = grp[j]
                qv = 0 if sf == 0 else int(np.rint((v - bf) / sf))
                qv = max(0, min(qmax, qv))
                wi = g * gs + j
                q[r, wi // pf] |= np.uint32(qv) << np.uint32((wi % pf) * bits)
    return q, scales, biases


cq, cs, cb = conv_pack(w)
print("mlx wq dtype/shape:", wq.dtype, wq.shape, "ours:", cq.shape)
mlx_q = np.array(wq.tolist(), dtype=np.uint64).astype(np.uint32) if wq.dtype == mx.int32 else np.array(wq.tolist(), dtype=object)
try:
    mlx_q = np.array(wq.tolist())
except Exception as e:
    print("tolist fail", e)
    mlx_q = None
if mlx_q is not None:
    mlx_q = np.asarray(mlx_q).astype(np.uint64).astype(np.uint32)
    same_words = (mlx_q == cq).mean()
    print(f"逐 u32 相同比例: {same_words:.4f}")
    if same_words < 1:
        r = 0
        for col in range(cq.shape[1]):
            if mlx_q[r, col] != cq[r, col]:
                print(f"首个不同 word: col={col} mlx=0x{mlx_q[r,col]:08x} ours=0x{cq[r,col]:08x}")
                break
    # scales 对比
    mlx_s = np.array(s.tolist())
    print("scales match:", np.array_equal(mlx_s.astype(np.int64), cs.astype(np.int64)), "mlx_s[0,:4]=", mlx_s[0, :4].tolist(), "ours[0,:4]=", f16_bits_to_f32(cs[0, 0]), f16_bits_to_f32(cs[0, 1]))
