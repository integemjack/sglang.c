"""GGUF(dsv41) -> MLX safetensors 转换器 v2（正确版）。

旧版 Q8_0 块大小写成 18（正确 34）、Q5_K 布局完全错误 -> 45% NaN/Inf。
本版逐条对照 llama.cpp ggml-common.h / ggml-quants.c：
  Q8_0: 34B/32el = [d f16][32 x i8]，val = d*q
  Q5_K: 176B/256el = [d f16][dmin f16][scales 12B][qh 32B][qs 128B]
        8 组 32el；sub j: sc1/m1=get_scale_min_k4(2j), sc2/m2=(2j+1)
        ql 低 4bit + qh bit(1<<2j) *16 -> d1*q - m1；高 4bit + bit(2<<2j) -> d2*q - m2
  F32/F16 原样。
大张量 -> mx.quantize(group=64, bits=4)；小张量 f16；hc/norm/sinks/bias f32。
只重写 model.safetensors，config.json 不动。
"""
import os, struct, sys
import numpy as np

GGUF_DIR = sys.argv[1] if len(sys.argv) > 1 else "/Volumes/9100 PRO/models/dsv41"
OUT = os.path.join(GGUF_DIR, "mlx-dense")
PREFIX = os.path.join(GGUF_DIR, "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-%05d-of-00006.gguf")
SHARDS = [PREFIX % i for i in range(1, 6)]
BIG_PARAMS = 8 * 1024 * 1024

QBLK = {0: (256, 1024), 1: (256, 512), 30: (256, 512),
        8: (32, 34),          # Q8_0
        12: (256, 144), 13: (256, 176), 14: (256, 210),  # Q4_K Q5_K Q6_K
        16: (256, 66), 17: (256, 74), 19: (256, 50), 20: (256, 136),
        23: (256, 136), 29: (256, 56)}

VTSZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 8: 1, 9: 1, 10: 8, 11: 8, 12: 8}


def ggml_nbytes(ne, gt):
    nb = QBLK[gt]
    n = 1
    for d in ne[:-1]:
        n *= d
    return n * ne[-1] // nb[0] * nb[1]


class Shards:
    def __init__(self, paths):
        self.fhs = [open(p, "rb") for p in paths]
        self.tensors = {}
        for fi, fh in enumerate(self.fhs):
            self._index(fh, fi)

    def _index(self, fh, fi):
        assert fh.read(4) == b"GGUF"
        _, n_ten, n_kv = struct.unpack("<IQQ", fh.read(20))
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
            self.tensors[name] = (fi, off, gtype, ne, ggml_nbytes(ne, gtype))

    @staticmethod
    def _skip_str(fh):
        n = struct.unpack("<Q", fh.read(8))[0]
        fh.seek(n, 1)

    @staticmethod
    def _read_str(fh):
        n = struct.unpack("<Q", fh.read(8))[0]
        return fh.read(n).decode("utf-8")

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
        if gt == 8:
            n = len(raw) // 34
            d = np.frombuffer(raw, dtype="<f2", count=n, offset=0)
            # 布局：每 34B 一块 [2B d][32B q]，d 数组需按块取
            d = np.frombuffer(raw, dtype=np.uint8).reshape(n, 34)[:, 0:2].copy().view(np.float16).astype(np.float32)
            q = np.frombuffer(raw, dtype=np.uint8).reshape(n, 34)[:, 2:].copy().view(np.int8).astype(np.float32)
            return (q * d).reshape(n1, n0)
        if gt == 13:
            return deq_q5k(raw).reshape(n1, n0)
        raise RuntimeError(f"gtype {gt} 未实现")


def get_scale_min_k4(j, q):
    """q: [n, 12] uint8，返回 sc/m [n]"""
    if j < 4:
        return q[:, j] & 63, q[:, j + 4] & 63
    return ((q[:, j + 4] & 0xF) | ((q[:, j - 4] >> 6) << 4)), ((q[:, j + 4] >> 4) | ((q[:, j] >> 6) << 4))


def deq_q5k(raw):
    n = len(raw) // 176
    r = np.frombuffer(raw, dtype=np.uint8).reshape(n, 176)
    d = r[:, 0:2].copy().view(np.float16).astype(np.float32)
    dmin = r[:, 2:4].copy().view(np.float16).astype(np.float32)
    scales = r[:, 4:16]
    qh = r[:, 16:48]
    qs = r[:, 48:176]
    out = np.empty((n, 256), dtype=np.float32)
    for j in range(4):  # 每 sub-block 64 elts
        sc1, m1 = get_scale_min_k4(2 * j, scales)
        sc2, m2 = get_scale_min_k4(2 * j + 1, scales)
        d1, d2 = d * sc1, d * sc2
        mm1, mm2 = dmin * m1, dmin * m2
        ql = qs[:, 32 * j:32 * j + 32]  # 32B：低 nibble 前 32 elts，高 nibble 后 32
        u1, u2 = 1 << (2 * j), 2 << (2 * j)
        h = qh[:, 0:32]
        lo = (ql & 0xF) + (((h & u1) != 0) * 16)
        hi = (ql >> 4) + (((h & u2) != 0) * 16)
        out[:, 64 * j:64 * j + 32] = lo * d1[:, None] - mm1[:, None]
        out[:, 64 * j + 32:64 * j + 64] = hi * d2[:, None] - mm2[:, None]
    return out


def q4_pack(w):
    """用 mlx 的原生量化（保证与 C++ mx::dequantize 语义一致）"""
    import mlx.core as mx
    x = mx.array(w)
    wq, s, b = mx.quantize(x, group_size=64, bits=4)
    return wq, s, b


if __name__ == "__main__":
    import mlx.core as mx

    shards = Shards(SHARDS)
    names = sorted(shards.tensors)
    keep = [nm for nm in names
            if not ("_exps" in nm and nm.endswith("weight"))
            and "per_layer_token_embd" not in nm
            and not nm.startswith("engram")]
    print(f"总 {len(names)} 转 {len(keep)}")
    out = {}
    for nm in keep:
        a = shards.read(nm)  # [ne1, ne0]
        p = a.size
        small = any(k in nm for k in ("norm", "bias", "sinks", "hc_", "_base", "_scale"))
        if small:
            v = mx.array(a.reshape(-1).astype(np.float32))
            out[nm] = v
        elif p > BIG_PARAMS:
            wq, s, b = q4_pack(a.astype(np.float32))
            out[nm] = wq
            out[nm + ".scales"] = s
            out[nm + ".biases"] = b
        else:
            out[nm] = mx.array(a.astype(np.float16))
        print(f"  {nm} {a.shape} -> {'q4' if (not small and p > BIG_PARAMS) else ('f32' if small else 'f16')}",
              flush=True)
    tmp = os.path.join(OUT, "model.safetensors.new")
    mx.save_safetensors(tmp, out)
    os.replace(tmp, os.path.join(OUT, "model.safetensors"))
    print("done", len(out))