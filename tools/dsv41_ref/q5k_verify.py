"""独立 Q5_K 反量化，验证 GGUF 原始 shexp 权重是否自带 DC。不依赖转换器代码。"""
import struct
import sys

import numpy as np

GGUF_DIR = sys.argv[1] if len(sys.argv) > 1 else "./models/dsv41"
TARGET = sys.argv[2] if len(sys.argv) > 2 else "blk.1.ffn_gate_shexp.weight"


def index_shard(path):
    fh = open(path, "rb")
    assert fh.read(4) == b"GGUF"
    ver = struct.unpack("<I", fh.read(4))[0]
    n_ten, n_kv = struct.unpack("<QQ", fh.read(16))
    print(f"[dbg] ver={ver} n_ten={n_ten} n_kv={n_kv}")

    def rstr():
        n = struct.unpack("<Q", fh.read(8))[0]
        return fh.read(n)


    VTS = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
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
    align = 32
    tens = {}
    for _ in range(n_ten):
        name = rstr().decode()
        nd = struct.unpack("<I", fh.read(4))[0]
        ne = struct.unpack("<" + "Q" * nd, fh.read(8 * nd))
        gtype = struct.unpack("<I", fh.read(4))[0]
        off = struct.unpack("<Q", fh.read(8))[0]
        tens[name] = (gtype, ne, off, fh.tell())
    # 数据基址：索引结束对齐
    idx_end = max((v[3] for v in tens.values()), default=0)
    base = (idx_end + align - 1) // align * align
    return fh, tens, base


def deq_q5k(raw, n_el):
    n = len(raw) // 176
    assert n * 256 == n_el, (n, n_el)
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
        d1 = d * sc1
        mm1 = dm * m1
        d2 = d * sc2
        mm2 = dm * m2
        q = ql[:, 32 * j : 32 * j + 32]
        u1, u2 = 1 << (2 * j), 2 << (2 * j)
        lo = (q & 0xF) + ((qh[:, 0:32] & u1) != 0) * 16
        hi = (q >> 4) + ((qh[:, 0:32] & u2) != 0) * 16
        out[:, 64 * j : 64 * j + 32] = lo * d1[:, None] - mm1[:, None]
        out[:, 64 * j + 32 : 64 * j + 64] = hi * d2[:, None] - mm2[:, None]
    return out.reshape(-1)


for i in range(1, 6):
    path = f"{GGUF_DIR}/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-{i:05d}-of-00006.gguf"
    fh, tens, base = index_shard(path)
    if TARGET not in tens:
        continue
    gtype, ne, off, _ = tens[TARGET]
    K, N = ne[0], ne[1]
    fh.seek(base + off)
    raw = fh.read(N * (K // 256) * 176)
    a = deq_q5k(raw, K * N).reshape(N, K)
    print(f"{TARGET} gtype={gtype} ne={ne}")
    print(f"  GGUF原版: mean={a.mean():+.6f} row_mean_std={a.mean(axis=1).std():.6f} rms={np.sqrt((a**2).mean()):.6g} max={np.abs(a).max():.6g}")
    print(f"  前8值: {np.round(a[0,:8],4)}")
    break
else:
    print("tensor not found in shards 1-5")
