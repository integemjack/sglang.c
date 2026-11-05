"""GGUF(dsv41) -> MLX safetensors 转换器 v3。

v1/v2(python) Q5_K 反量化有 bug；v2.5(C++) 修了反量化但 affine4 打包自创（LSB-first
min/max 方案），与 MLX 原生语义不符（MLX 打包位序不同、scales 可为负的拟合式），
读回来权重 rms 膨胀 2.4x、直流 +0.028，真权重前向 NaN（2026-11-05 定位：
tools/dsv41_ref/{pack_check,q5k_verify,dc_probe,y2_forensic}.py）。
v3：Q5_K 反量化用独立验证过的实现（q5k_verify.py，GGUF 原版 mean=+1e-6）；
打包一律用 mx.quantize 原生（引擎 Wd()/qmm 与 python dequantize 三方一致）。
"""
import gc
import os
import struct
import sys

import mlx.core as mx
import numpy as np

GGUF_DIR = sys.argv[1] if len(sys.argv) > 1 else "./models/dsv41"
OUT = os.path.join(GGUF_DIR, "mlx-dense")
PREFIX = os.path.join(GGUF_DIR, "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-%05d-of-00006.gguf")
SHARDS = [PREFIX % i for i in range(1, 6)]
BIG_PARAMS = 8 * 1024 * 1024

QBLK = {0: (1, 4), 1: (1, 2), 30: (1, 2), 8: (32, 34), 13: (256, 176)}


def ggml_nbytes(ne, gt):
    nb = QBLK[gt]
    n = 1
    for d in ne[:-1]:
        n *= d
    return n * ne[-1] // nb[0] * nb[1]


VTS = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def index_shard(path):
    """与 q5k_verify.py 逐字一致（已在真实分片上验证）"""
    fh = open(path, "rb")
    assert fh.read(4) == b"GGUF"
    struct.unpack("<I", fh.read(4))
    n_ten, n_kv = struct.unpack("<QQ", fh.read(16))

    def rstr():
        n = struct.unpack("<Q", fh.read(8))[0]
        return fh.read(n)

    for _ in range(n_kv):
        rstr()
        t = struct.unpack("<I", fh.read(4))[0]
        if t == 8:
            rstr()
        elif t == 9:
            et = struct.unpack("<I", fh.read(4))[0]
            n = struct.unpack("<Q", fh.read(8))[0]
            if et == 8:
                for _ in range(n):
                    rstr()
            else:
                fh.seek(n * VTS[et], 1)
        else:
            fh.seek(VTS[t], 1)
    tens = {}
    for _ in range(n_ten):
        name = rstr().decode()
        nd = struct.unpack("<I", fh.read(4))[0]
        ne = struct.unpack("<" + "Q" * nd, fh.read(8 * nd))
        gtype = struct.unpack("<I", fh.read(4))[0]
        off = struct.unpack("<Q", fh.read(8))[0]
        tens[name] = (gtype, ne, off)
    idx_end = fh.tell()
    align = 32
    base = (idx_end + align - 1) // align * align
    return fh, tens, base


class Shards:
    def __init__(self, paths):
        self.fhs = []
        self.tensors = {}
        for fi, p in enumerate(paths):
            fh, tens, base = index_shard(p)
            self.fhs.append(fh)
            for name, (gtype, ne, off) in tens.items():
                if gtype in QBLK:
                    self.tensors[name] = (fi, base + off, gtype, ne, ggml_nbytes(ne, gtype))

    def read(self, name):
        fi, off, gt, ne, size = self.tensors[name]
        fh = self.fhs[fi]
        fh.seek(off)
        raw = fh.read(size)
        n0 = int(ne[0])
        n1 = int(ne[1]) if len(ne) > 1 else 1
        if gt == 0:
            return np.frombuffer(raw, dtype="<f4").reshape(n1, n0).astype(np.float32)
        if gt == 1:
            return np.frombuffer(raw, dtype="<f2").reshape(n1, n0).astype(np.float32)
        if gt == 30:
            return (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32).reshape(n1, n0).copy()
        if gt == 8:
            n = len(raw) // 34
            r = np.frombuffer(raw, dtype=np.uint8).reshape(n, 34)
            d = r[:, 0:2].copy().view(np.float16).astype(np.float32)
            q = r[:, 2:].copy().view(np.int8).astype(np.float32)
            return (q * d).reshape(n1, n0)
        if gt == 13:
            return deq_q5k(raw).reshape(n1, n0)
        raise RuntimeError(f"gtype {gt}")


def deq_q5k(raw):
    n = len(raw) // 176
    r = np.frombuffer(raw, dtype=np.uint8).reshape(n, 176)
    d = r[:, 0:2].copy().view(np.float16).astype(np.float32).ravel()
    dm = r[:, 2:4].copy().view(np.float16).astype(np.float32).ravel()
    scales = r[:, 4:16]
    qh = r[:, 16:48]
    ql = r[:, 48:176]
    out = np.empty((n, 256), dtype=np.float32)

    def k4(j):
        if j < 4:
            return scales[:, j] & 63, scales[:, j + 4] & 63
        return (
            (scales[:, j + 4] & 0xF) | ((scales[:, j - 4] >> 6) << 4),
            (scales[:, j + 4] >> 4) | ((scales[:, j] >> 6) << 4),
        )

    for j in range(4):
        sc1, m1 = k4(2 * j)
        sc2, m2 = k4(2 * j + 1)
        lo = (ql[:, 32 * j : 32 * j + 32] & 0xF) + ((qh & (1 << (2 * j))) != 0) * 16
        hi = (ql[:, 32 * j : 32 * j + 32] >> 4) + ((qh & (2 << (2 * j))) != 0) * 16
        out[:, 64 * j : 64 * j + 32] = lo * (d * sc1)[:, None] - (dm * m1)[:, None]
        out[:, 64 * j + 32 : 64 * j + 64] = hi * (d * sc2)[:, None] - (dm * m2)[:, None]
    return out.reshape(-1)


if __name__ == "__main__":
    shards = Shards(SHARDS)
    names = sorted(shards.tensors)
    keep = [nm for nm in names
            if not ("_exps" in nm and nm.endswith("weight"))
            and "per_layer_token_embd" not in nm
            and not nm.startswith("engram")]
    print(f"总 {len(names)} 转 {len(keep)}", flush=True)
    out = {}
    for i, nm in enumerate(keep):
        a = shards.read(nm)  # [ne1, ne0]
        p = a.size
        small = any(k in nm for k in ("norm", "bias", "sinks", "hc_", "_base", "_scale"))
        if small:
            v = mx.array(a.reshape(-1).astype(np.float32))
            out[nm] = v
            kind = "f32"
        elif p > BIG_PARAMS:
            wq, s, b = mx.quantize(mx.array(a.astype(np.float32)), group_size=64, bits=4)
            out[nm] = wq
            out[nm + ".scales"] = s
            out[nm + ".biases"] = b
            kind = "q4"
        else:
            out[nm] = mx.array(a.astype(np.float16))
            kind = "f16"
        del a
        if i % 20 == 0:
            print(f"  [{i}/{len(keep)}] {nm} -> {kind}", flush=True)
    tmp = os.path.join(OUT, "model.safetensors.new")
    mx.save_safetensors(tmp, out)
    os.replace(tmp, os.path.join(OUT, "model.safetensors"))
    print("written", os.path.getsize(os.path.join(OUT, "model.safetensors")), flush=True)

    # 写后自校验：blk.1 shexp 反量化统计必须与 GGUF 原版一致（mean~1e-6, rms~0.0194）
    del out
    gc.collect()
    W = mx.load(os.path.join(OUT, "model.safetensors"))
    k = "blk.1.ffn_gate_shexp.weight"
    dq = mx.dequantize(W[k], W[k + ".scales"], W[k + ".biases"], 64, 4).astype(mx.float32)
    mean = float(mx.mean(dq))
    rms = float(mx.sqrt(mx.mean(dq * dq)))
    print(f"VERIFY blk.1 shexp: mean={mean:+.6f} rms={rms:.6f} (GGUF原版 mean=+0.000001 rms=0.019431)", flush=True)
    ok = abs(mean) < 5e-4 and abs(rms - 0.019431) < 3e-3
    print("VERIFY_RESULT:", "PASS" if ok else "FAIL", flush=True)
