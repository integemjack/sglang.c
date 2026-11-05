"""GGUF(dsv41) -> MLX safetensors 转换器：只转稠密/注意力权重（专家和 Engram 表留在 GGUF）。

用法: python conv_dsv41.py <gguf前缀> <engram可省> <输出目录>
大张量(>big_thresh 参数) -> affine 4bit g64；小张量 -> f16；norm/sinks/bias/gate -> f32。
输出: model.safetensors + config.json
"""
import json, math, os, struct, sys
import numpy as np

GGUF_DIR = sys.argv[1] if len(sys.argv) > 1 else "/Volumes/9100 PRO/models/dsv41"
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "/Volumes/9100 PRO/models/dsv41/mlx-iq2"
PREFIX = os.path.join(GGUF_DIR, "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-%05d-of-00006.gguf")
BIG_PARAMS = 8 * 1024 * 1024  # >8M 参数的转 4bit
BIG_BITS = 4
SMALL_BITS = 8  # shexp 等中等张量 8bit

# ---------------- GGUF 读取（分片） ----------------
VTSZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

class Shards:
    def __init__(self, paths):
        self.fhs = [open(p, "rb") for p in paths]
        self.tensors = {}  # name -> (file_idx, data_off, gtype, ne[dim0,dim1,...] gguf 顺序=小端在前)
        for fi, fh in enumerate(self.fhs):
            self._index(fh, fi)

    def _index(self, fh, fi):
        assert fh.read(4) == b"GGUF"
        ver, n_ten, n_kv = struct.unpack("<IQQ", fh.read(20))
        for _ in range(n_kv):
            self._skip_str(fh)
            t = struct.unpack("<I", fh.read(4))[0]
            if t == 8:
                self._skip_str(fh)
            elif t == 9:
                et = struct.unpack("<I", fh.read(4))[0]
                n = struct.unpack("<Q", fh.read(8))[0]
                if et == 8:
                    for _ in range(n):
                        self._skip_str(fh)
                else:
                    fh.seek(n * VTSZ[et], 1)
            elif t in VTSZ:
                fh.seek(VTSZ[t], 1)
            else:
                raise RuntimeError(f"kv type {t}")
        for _ in range(n_ten):
            name = self._read_str(fh)
            nd = struct.unpack("<I", fh.read(4))[0]
            ne = struct.unpack("<" + "Q" * nd, fh.read(8 * nd))
            gtype = struct.unpack("<I", fh.read(4))[0]
            off = struct.unpack("<Q", fh.read(8))[0]
            fh.seek(0, 2)
            size = ggml_nbytes(ne, gtype)
            self.tensors[name] = (fi, off, gtype, ne, size)

    @staticmethod
    def _skip_str(fh):
        n = struct.unpack("<Q", fh.read(8))[0]
        fh.seek(n, 1)

    @staticmethod
    def _read_str(fh):
        n = struct.unpack("<Q", fh.read(8))[0]
        return fh.read(n).decode("utf-8")

    def read(self, name):
        fi, off, gtype, ne, size = self.tensors[name]
        fh = self.fhs[fi]
        fh.seek(off)
        raw = fh.read(size)
        return dequant(raw, gtype, ne)  # [d0, d1, ...] gguf 顺序

def ggml_nbytes(ne, gt):
    nb = QBLK.get(gt)
    if nb is None:
        raise RuntimeError(f"gtype {gt}")
    n = 1
    for d in ne[:-1]:
        n *= d
    return n * ne[-1] // nb[0] * nb[1]

# gtype: (块元素数, 块字节数)
QBLK = {
    0: (256, 4),     # F32
    1: (256, 2),     # F16
    30: (256, 256),  # BF16
    8: (32, 18),     # Q8_0: f16 d + 32 i8
    6: (256, 210),   # Q5_K
    10: (256, 84),   # Q5_K? no
    14: (256, 110),  # Q6_K
}
QBLK[6] = (256, 176)   # Q5_K: 4b q + scales
QBLK[10] = (256, 144)  # Q5_K_S? 修正见下
# llama.cpp ggml 类型: 6=Q4_K(144), 8?…用官方编号
QBLK = {
    0: (256, 4), 1: (256, 2), 30: (256, 256),
    8: (32, 18),      # Q8_0
    12: (256, 144),   # Q4_K
    13: (256, 168),   # Q4_K_S? -> 12/13 都按 q4k 处理风险，见 ggml.h
    14: (256, 210),   # Q5_K
    15: (256, 176),   # Q6_K
    16: (256, 84),    # Q8_K? no, 16=IQ2_XXS
    29: (256, 128),   # BF16? 编号校准见下
}
# —— 以 ggml.h 为准的类型编号：F32=0 F16=1 Q4_0=2 Q4_1=3 Q5_0=6 Q5_1=7 Q8_0=8 Q8_1=9
# Q2_K=10 Q3_K=11 Q4_K=12 Q5_K=13 Q6_K=14 Q8_K=15 IQ2_XXS=16 IQ2_XS=17 IQ3_XXS=18
# IQ1_S=19 IQ4_NL=20 IQ3_S=21 IQ2_S=22 IQ4_XS=23 I8=24 I16=25 I32=26 I64=27 F64=28 IQ1_M=29 BF16=30
QBLK = {
    0: (256, 4), 1: (256, 2), 30: (256, 256), 28: (256, 8),
    8: (32, 18),                                   # Q8_0
    12: (256, 144), 13: (256, 176), 14: (256, 210),  # Q4_K Q5_K Q6_K
    15: (256, 256),                                  # Q8_K
    16: (256, 66),   # IQ2_XXS: 2.0625 bpw -> 256*2.0625/8=66
    17: (256, 74),   # IQ2_XS  2.3125 -> 74
    19: (256, 50),   # IQ1_S   1.5625 -> 50
    20: (256, 136),  # IQ4_NL  4.5 -> 136
    23: (256, 136),  # IQ4_XS  4.5 -> 136
    29: (256, 58),   # IQ1_M   1.75 -> 56.25 -> 56? 见下
}
QBLK[29] = (256, 56)  # IQ1_M = 1.75 bpw = 56B/256elt

KB = 0.25

def dequant(raw, gt, ne):
    """返回 numpy 数组，形状按 ggml 语义 [ne[-1], ..., ne[0]] 的转置习惯——直接给 [ne[1], ne[0]]"""
    if gt == 0:
        a = np.frombuffer(raw, dtype=np.float32)
    elif gt == 1:
        a = np.frombuffer(raw, dtype=np.float16)
    elif gt == 30:
        a = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        a = a.view(np.float32) if a.dtype == np.float32 else (np.uint32(a) << np.uint32(16)).view(np.float32)
        a = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << np.uint32(16)).view(np.float32)
    elif gt == 8:  # Q8_0
        n = len(raw) // 18
        d = np.frombuffer(raw, dtype=np.float16).reshape(n, 1)
        q = np.frombuffer(raw, dtype=np.int8).reshape(n, 18)[:, 2:].astype(np.float32)
        a = (q * d)
    elif gt in (13,):  # Q5_K
        a = dequant_q5k(raw)
    elif gt in (16,):
        a = dequant_iq2_xxs(raw)
    elif gt in (17,):
        a = dequant_iq2_xs(raw)
    elif gt in (29,):
        a = dequant_iq1_m(raw)
    elif gt in (20, 23):
        a = dequant_iq4(raw, gt)
    else:
        raise RuntimeError(f"未实现的 gtype {gt}")
    n0, n1 = int(ne[0]), int(ne[1])
    if gt in (0, 1, 30):
        a = a.reshape(n1, n0)
    else:
        rows = n1 * n0 // 256
        a = a.reshape(rows, 256)
    return a.astype(np.float32)

def dequant_q5k(raw):
    """Q5_K: 每 256 elt 一块 176B = [4B d/q dmin(f32 pair)] + [32B ql] + [16B qh] + [16B scales(6bit each x8)]"""
    n = len(raw) // 176
    out = np.empty((n, 256), dtype=np.float32)
    r = np.frombuffer(raw, dtype=np.uint8).reshape(n, 176)
    d = r[:, 0:4].copy().view(np.float16).astype(np.float32).reshape(n, 1)
    dmin = r[:, 4:8].copy().view(np.float16).astype(np.float32).reshape(n, 1)
    ql = r[:, 8:40].astype(np.uint16)
    qh = r[:, 40:56]
    sc = r[:, 56:72]
    # scales: 8 个 6bit
    sc8 = np.empty((n, 8), dtype=np.uint8)
    sc8[:, 0] = sc[:, 0] & 63
    sc8[:, 1] = ((sc[:, 1] & 15) << 2) | (sc[:, 0] >> 6)
    sc8[:, 2] = ((sc[:, 2] & 3) << 4) | (sc[:, 1] >> 4)
    sc8[:, 3] = ((sc[:, 3] & 63) >> 2) | 0
    sc8[:, 3] = ((sc[:, 2] >> 6) | ((sc[:, 3] & 15) << 2)) & 63
    sc8[:, 4] = sc[:, 4] & 63
    sc8[:, 5] = ((sc[:, 5] & 15) << 2) | (sc[:, 4] >> 6)
    sc8[:, 6] = ((sc[:, 6] & 3) << 4) | (sc[:, 5] >> 4)
    sc8[:, 7] = (sc[:, 6] >> 6) | ((sc[:, 7] & 15) << 2)
    dsc = (sc8.astype(np.float32) * d)
    dm = (sc8.astype(np.float32) * dmin)
    # ql/qh -> 32 个 5bit 值 x8 子块
    qh_bits = np.unpackbits(qh.reshape(n, 16), axis=1, bitorder="little")  # [n,128]
    for ib in range(8):
        lo = ql[:, ib * 16:(ib + 1) * 16].copy().view(np.uint8) if False else None
    # 简化：按 llama.cpp get_color 逻辑逐 nibble 展开
    qlb = r[:, 8:40]  # 32B = 64 nibble
    nib = np.empty((n, 64), dtype=np.uint8)
    nib[:, 0::2] = qlb & 15
    nib[:, 1::2] = qlb >> 4
    q5 = nib.reshape(n, 8, 32).astype(np.uint16)
    q5 |= (qh_bits.reshape(n, 8, 32) << np.uint16(4))
    q5 = q5.astype(np.float32)
    for j in range(8):
        out[:, j * 32:(j + 1) * 32] = q5[:, j] * dsc[:, j] - dm[:, j]
    return out

def dequant_iq4(raw, gt):
    """IQ4_NL(20)/IQ4_XS(23)：kvalues 查表"""
    import gguf_quants as G
    if gt == 20:  # 每 32 elt: [2B f16 d][16B idx]
        n = len(raw) // 18
        r = np.frombuffer(raw, dtype=np.uint8).reshape(n, 18)
        d = r[:, 0:2].copy().view(np.float16).astype(np.float32)
        idx = r[:, 2:18]
    else:  # IQ4_XS 每 256: [2B d][2B scales][124B idx]
        n = len(raw) // 128
        r = np.frombuffer(raw, dtype=np.uint8).reshape(n, 128)
        d = r[:, 0:2].copy().view(np.float16).astype(np.float32)
        s = r[:, 2:4]
        idx = r[:, 4:128]
        # 略：XS 先不做
        raise RuntimeError("IQ4_XS 暂不支持")
    kv = np.array(G.kvalues_iq4nl, dtype=np.float32)
    return (kv[idx.reshape(-1)].reshape(n, 32) * d[:, None]).reshape(-1, 256)

def dequant_iq2_xxs(raw):
    import gguf_quants as G
    n = len(raw) // 66
    r = np.frombuffer(raw, dtype=np.uint8).reshape(n, 66)
    d = np.frombuffer(raw, dtype=np.uint16).reshape(n, 33)[:, 0].astype(np.uint16)
    d = d.copy().view(np.float16).astype(np.float32)
    qs = r[:, 2:].reshape(n, 64)  # 64B
    # 256 elt = 8 组 x 32；每组 8B：低 7B 索引 + 1B 符号/网格
    out = np.empty((n, 256), dtype=np.float32)
    CB2 = np.array(G.iq2xxs_grid, dtype=np.uint8).reshape(-1, 8)
    for g in range(8):
        blk = qs[:, g * 8:(g + 1) * 8]
        idx = (blk[:, 0].astype(np.uint32) | (blk[:, 1].astype(np.uint32) << 8) |
               (blk[:, 2].astype(np.uint32) << 16) | ((blk[:, 3].astype(np.uint32) & 3) << 24))
        grid = CB2[idx]  # [n,8] 8 elt
        sign = np.unpackbits(blk[:, 3:4][:, 0:1] >> 2, axis=0) if False else None
        # 符号位在 qs 的高 6bit + 后续，见 ggml dequantize_row_iq2_xxs
        out[:, g * 32:(g + 1) * 32] = grid.astype(np.float32)  # TODO scales/sign
    raise RuntimeError("IQ2_XXS 完整反量化需按 ggml-quants.c 逐位实现——见 build/gguf_dequant_ref.py")

def dequant_iq2_xs(raw):
    raise RuntimeError("IQ2_XS 见 ggml-quants.c")

def dequant_iq1_m(raw):
    raise RuntimeError("IQ1_M 见 ggml-quants.c")

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    shards = Shards([PREFIX.format(i) for i in range(1, 6)])
    names = sorted(shards.tensors)
    keep = [n for n in names if not ("_exps" in n and n.endswith("weight")) and "per_layer_token_embd" not in n]
    print(f"总张量 {len(names)}，转 {len(keep)}（专家/Engram 留 GGUF）")
    import mlx.core as mx
    out = {}
    for nm in keep:
        a = shards.read(nm)  # [rows(=ne[1]), cols(=ne[0])]
        p = a.size
        if "norm" in nm or "bias" in nm or "sinks" in nm:
            w = mx.array(a.reshape(-1))
        elif p > BIG_PARAMS:
            pass
        print(f"  {nm} {a.shape} {a.dtype}", flush=True)
        break  # 先冒烟
