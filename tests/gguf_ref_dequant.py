"""GGUF 反量化的独立参考实现（与 C 侧 gguf.c 对拍用）。

为什么要有它：C 侧手写了 ggml 的 K-quant 位布局，只跟"自己另写一份"比不够
（两边抄错同一处会误过）。这里的独立性来自：块结构用 ggml 的公开结构体定义成
numpy 结构化 dtype，由 numpy 负责字段切分，我只写算术部分；张量的形状直接取
GGUF 容器里的 ne（与 C 侧独立）。

用法：
    python gguf_ref_dequant.py <a.gguf> [b.gguf ...]

对每个 gguf 生成：
    <gguf>.refmeta   每行 "hf_name<TAB>ndim<TAB>d0,d1<TAB>byte_offset"
    <gguf>.refbin    依次堆放的 float32 小端（HF 行主序 [out, in]）

导出范围：全部 1D（norm/bias）+ 若干层的全部 2D。embed 不导出（端到端生成已覆盖）。
"""
import os
import sys

import numpy as np

try:
    import gguf
except ImportError:
    sys.exit("need: pip install gguf")

QK_K = 256

# ggml 的块结构（ggml-quants.h），itemsize 必须与 GGML_QUANT_SIZES 的字节数一致
BLK = {
    "F32": np.dtype([("f", np.float32)]),
    "F16": np.dtype([("f", np.float16)]),
    "BF16": np.dtype([("f", np.uint16)]),
    "Q4_0": np.dtype([("d", np.float16), ("qs", np.uint8, (16,))]),
    "Q4_1": np.dtype([("d", np.float16), ("m", np.float16), ("qs", np.uint8, (16,))]),
    "Q5_0": np.dtype([("d", np.float16), ("qh", np.uint32), ("qs", np.uint8, (16,))]),
    "Q5_1": np.dtype([("d", np.float16), ("m", np.float16), ("qh", np.uint32),
                      ("qs", np.uint8, (16,))]),
    "Q8_0": np.dtype([("d", np.float16), ("qs", np.int8, (32,))]),
    "IQ4_NL": np.dtype([("d", np.float16), ("qs", np.uint8, (16,))]),
    "Q2_K": np.dtype([("scales", np.uint8, (16,)), ("qs", np.uint8, (64,)),
                      ("d", np.float16), ("dmin", np.float16)]),
    "Q3_K": np.dtype([("hmask", np.uint8, (32,)), ("qs", np.uint8, (64,)),
                      ("scales", np.uint8, (12,)), ("d", np.float16)]),
    "Q4_K": np.dtype([("d", np.float16), ("dmin", np.float16), ("scales", np.uint8, (12,)),
                      ("qs", np.uint8, (128,))]),
    "Q5_K": np.dtype([("d", np.float16), ("dmin", np.float16), ("scales", np.uint8, (12,)),
                      ("qh", np.uint8, (32,)), ("qs", np.uint8, (128,))]),
    "Q6_K": np.dtype([("ql", np.uint8, (128,)), ("qh", np.uint8, (64,)),
                      ("scales", np.int8, (16,)), ("d", np.float16)]),
    "Q8_K": np.dtype([("d", np.float32), ("qs", np.int8, (256,)), ("bsums", np.int16, (16,))]),
}
BLOCK_ELEMS = {"Q4_0": 32, "Q4_1": 32, "Q5_0": 32, "Q5_1": 32, "Q8_0": 32, "IQ4_NL": 32,
               "Q2_K": QK_K, "Q3_K": QK_K, "Q4_K": QK_K, "Q5_K": QK_K, "Q6_K": QK_K,
               "Q8_K": QK_K, "F32": 1, "F16": 1, "BF16": 1}


def _f16(a):
    return np.asarray(a, dtype=np.float16).astype(np.float32)


def _deq_q4_0(b):
    d = _f16(b["d"])[:, None]
    q = b["qs"].astype(np.int16)
    return (np.concatenate([(q & 0x0F) - 8, (q >> 4) - 8], axis=1).astype(np.float32)) * d


def _deq_q4_1(b):
    d = _f16(b["d"])[:, None]
    m = _f16(b["m"])[:, None]
    q = b["qs"].astype(np.float32)
    return np.concatenate([q % 16.0, np.floor(q / 16.0)], axis=1) * d + m


def _deq_q5_0(b):
    d = _f16(b["d"])[:, None]
    q = b["qs"].astype(np.int32)
    qh = b["qh"].astype(np.uint32)
    bits = ((qh[:, None] >> np.arange(32, dtype=np.uint32)[None, :]) & 1).astype(np.int32)
    lo = (q & 0x0F) | (bits[:, 0:16] << 4)
    hi = (q >> 4) | (bits[:, 16:32] << 4)
    return (np.concatenate([lo, hi], axis=1) - 16).astype(np.float32) * d


def _deq_q5_1(b):
    d = _f16(b["d"])[:, None]
    m = _f16(b["m"])[:, None]
    q = b["qs"].astype(np.int32)
    qh = b["qh"].astype(np.uint32)
    bits = ((qh[:, None] >> np.arange(32, dtype=np.uint32)[None, :]) & 1).astype(np.int32)
    lo = (q & 0x0F) | (bits[:, 0:16] << 4)
    hi = (q >> 4) | (bits[:, 16:32] << 4)
    return (np.concatenate([lo, hi], axis=1)).astype(np.float32) * d + m


def _deq_q8_0(b):
    return b["qs"].astype(np.float32) * _f16(b["d"])[:, None]


# ggml 的 kvalues_iq4nl（16 项非线性码本）
KVALS_IQ4NL = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
                       dtype=np.float32)


def _deq_iq4_nl(b):
    d = _f16(b["d"])[:, None]
    q = b["qs"]
    lo = KVALS_IQ4NL[q & 0x0F]
    hi = KVALS_IQ4NL[q >> 4]
    return np.concatenate([lo, hi], axis=1).astype(np.float32) * d


def _deq_q2_K(b):
    nb = b.shape[0]
    out = np.zeros((nb, QK_K), dtype=np.float32)
    d = _f16(b["d"])[:, None]
    dmin = _f16(b["dmin"])[:, None]
    for i in range(nb):
        is_ = 0
        for half in range(2):
            off = half * 128
            q = b["qs"][i, half * 32:(half + 1) * 32].astype(np.int32)
            shift = 0
            for j in range(4):
                sc = int(b["scales"][i, is_]); is_ += 1
                dl, ml = d[i, 0] * (sc & 0xF), dmin[i, 0] * (sc >> 4)
                out[i, off + 32 * j:off + 32 * j + 16] = dl * ((q[0:16] >> shift) & 3) - ml
                sc = int(b["scales"][i, is_]); is_ += 1
                dl, ml = d[i, 0] * (sc & 0xF), dmin[i, 0] * (sc >> 4)
                out[i, off + 32 * j + 16:off + 32 * j + 32] = dl * ((q[16:32] >> shift) & 3) - ml
                shift += 2
    return out


def _deq_q3_K(b):
    nb = b.shape[0]
    out = np.zeros((nb, QK_K), dtype=np.float32)
    kmask1, kmask2 = 0x03030303, 0x0F0F0F0F
    for i in range(nb):
        d = float(_f16(b["d"][i]))
        a = np.frombuffer(b["scales"][i].tobytes(), dtype=np.uint32)
        tmp = int(a[2])
        aux = [
            (int(a[0]) & kmask2) | (((tmp >> 0) & kmask1) << 4),
            (int(a[1]) & kmask2) | (((tmp >> 2) & kmask1) << 4),
            ((int(a[0]) >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4),
            ((int(a[1]) >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4),
        ]
        sc = np.frombuffer(np.array(aux, dtype=np.uint32).tobytes(), dtype=np.int8)
        # 注意：hmask 是整块共用的 32 字节（不随 half 推进），靠 m 左移区分：
        # ggml 的 hm 指针在整个 block 内不变。
        h = b["hmask"][i].astype(np.int32)
        is_, m = 0, 1
        for half in range(2):
            off = half * 128
            q = b["qs"][i, half * 32:(half + 1) * 32].astype(np.int32)
            shift = 0
            for j in range(4):
                dl = d * (int(sc[is_]) - 32); is_ += 1
                sub = np.where((h[0:16] & m) != 0, 0, 4)
                out[i, off + 32 * j:off + 32 * j + 16] = dl * (((q[0:16] >> shift) & 3) - sub)
                dl = d * (int(sc[is_]) - 32); is_ += 1
                sub = np.where((h[16:32] & m) != 0, 0, 4)
                out[i, off + 32 * j + 16:off + 32 * j + 32] = dl * (((q[16:32] >> shift) & 3) - sub)
                shift += 2
                m <<= 1
    return out


def _get_scale_min_k4(j, sc):
    if j < 4:
        return sc[j] & 63, sc[j + 4] & 63
    return (((sc[j + 4] & 0x0F) | ((sc[j - 4] >> 6) << 4)),
            ((sc[j + 4] >> 4) | ((sc[j] >> 6) << 4)))


def _deq_q4_K(b):
    nb = b.shape[0]
    out = np.zeros((nb, QK_K), dtype=np.float32)
    for i in range(nb):
        d = float(_f16(b["d"][i]))
        dmin = float(_f16(b["dmin"][i]))
        sc = b["scales"][i]
        q = b["qs"][i].astype(np.int32)
        is_ = 0
        for k in range(0, QK_K, 64):
            s0, m0 = _get_scale_min_k4(is_ + 0, sc)
            s1, m1 = _get_scale_min_k4(is_ + 1, sc)
            bi = k // 64
            row = q[bi * 32:bi * 32 + 32]
            out[i, k:k + 32] = d * s0 * (row & 0x0F) - dmin * m0
            out[i, k + 32:k + 64] = d * s1 * (row >> 4) - dmin * m1
            is_ += 2
    return out


def _deq_q5_K(b):
    nb = b.shape[0]
    out = np.zeros((nb, QK_K), dtype=np.float32)
    for i in range(nb):
        d = float(_f16(b["d"][i]))
        dmin = float(_f16(b["dmin"][i]))
        sc = b["scales"][i]
        ql = b["qs"][i].astype(np.int32)
        # 注意：qh 是整块共用的 32 字节，靠 u1/u2 左移推进（ggml 里 qh 指针不变）
        qh = b["qh"][i].astype(np.int32)
        is_, u1, u2 = 0, 1, 2
        for k in range(0, QK_K, 64):
            s0, m0 = _get_scale_min_k4(is_ + 0, sc)
            s1, m1 = _get_scale_min_k4(is_ + 1, sc)
            bi = k // 64
            lo = ql[bi * 32:bi * 32 + 32]
            out[i, k:k + 32] = d * s0 * ((lo & 0x0F) + np.where((qh & u1) != 0, 16, 0)) - dmin * m0
            out[i, k + 32:k + 64] = d * s1 * ((lo >> 4) + np.where((qh & u2) != 0, 16, 0)) - dmin * m1
            is_ += 2
            u1 <<= 2
            u2 <<= 2
    return out


def _deq_q6_K(b):
    nb = b.shape[0]
    out = np.zeros((nb, QK_K), dtype=np.float32)
    for i in range(nb):
        d = float(_f16(b["d"][i]))
        for half in range(2):
            off = half * 128
            ql = b["ql"][i, half * 64:(half + 1) * 64].astype(np.int32)
            qh = b["qh"][i, half * 32:(half + 1) * 32].astype(np.int32)
            sc = b["scales"][i, half * 8:(half + 1) * 8].astype(np.int32)
            for l in range(32):
                isg = l // 16
                q1 = (ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)
                q2 = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)
                q3 = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)
                q4 = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)
                base = off + l
                out[i, base + 0] = d * sc[isg + 0] * (q1 - 32)
                out[i, base + 32] = d * sc[isg + 2] * (q2 - 32)
                out[i, base + 64] = d * sc[isg + 4] * (q3 - 32)
                out[i, base + 96] = d * sc[isg + 6] * (q4 - 32)
    return out


DEQ = {"Q4_0": _deq_q4_0, "Q4_1": _deq_q4_1, "Q5_0": _deq_q5_0, "Q5_1": _deq_q5_1,
       "Q8_0": _deq_q8_0, "IQ4_NL": _deq_iq4_nl, "Q2_K": _deq_q2_K, "Q3_K": _deq_q3_K,
       "Q4_K": _deq_q4_K, "Q5_K": _deq_q5_K, "Q6_K": _deq_q6_K}


def dequant_blocks(t):
    """把 tensor.data（uint8，形状 [rows, row_bytes]）解成 [rows, ne0] 的 f32。"""
    tname = t.tensor_type.name
    dt = BLK.get(tname)
    if dt is None:
        return None
    raw = np.ascontiguousarray(t.data)
    if tname in ("F32", "F16"):
        v = raw.view(dt).reshape(-1).astype(np.float32)
        return v.reshape(tuple(int(x) for x in reversed(t.shape)))
    if tname == "BF16":
        u = raw.view(np.uint16).astype(np.uint32) << 16
        return u.view(np.float32).reshape(tuple(int(x) for x in reversed(t.shape)))
    be = BLOCK_ELEMS[tname]
    nb_row = dt.itemsize
    if raw.ndim < 2:
        raw = raw.reshape(1, -1)
    rows = raw.shape[0]
    if raw.shape[1] % nb_row != 0:
        raise SystemExit(f"{t.name}: {tname} row_bytes {raw.shape[1]} % {nb_row} != 0")
    blk = raw.view(dt).reshape(rows, raw.shape[1] // nb_row)
    flat = DEQ[tname](blk.reshape(-1))
    return flat.reshape(rows, -1)


def to_hf(gname):
    """GGUF 名 → HF 名（与 C 侧同一套规则，这里独立写一遍）"""
    simple = {"token_embd.weight": "model.embed_tokens.weight",
              "output_norm.weight": "model.norm.weight",
              "output.weight": "lm_head.weight"}
    if gname in simple:
        return simple[gname]
    if not gname.startswith("blk."):
        return None
    rest = gname[4:]
    dot = rest.index(".")
    i, leaf = rest[:dot], rest[dot + 1:]
    blk = {"attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm",
           "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj",
           "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj",
           "attn_q_norm": "self_attn.q_norm", "attn_k_norm": "self_attn.k_norm",
           "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj",
           "ffn_down": "mlp.down_proj"}
    for g, h in blk.items():
        if leaf in (g + ".weight", g + ".bias"):
            suf = ".weight" if leaf.endswith(".weight") else ".bias"
            return "model.layers.%s.%s%s" % (i, h, suf)
    return None


def wanted(gname, ndim):
    if ndim == 1:
        return True
    if "token_embd" in gname or gname == "output.weight":
        return False
    for mark in ("blk.0.", "blk.1.", "blk.2.", "blk.15.", "blk.29."):
        if mark in gname:
            return True
    return False


def export(path):
    rd = gguf.GGUFReader(path)
    types = {}
    n_exp = 0
    with open(path + ".refbin", "wb") as fb, open(path + ".refmeta", "w", encoding="utf-8") as fm:
        off = 0
        for t in rd.tensors:
            hf = to_hf(t.name)
            if hf is None or not wanted(t.name, len(t.shape)):
                continue
            vals = dequant_blocks(t)
            if vals is None:
                sys.exit(f"unsupported type {t.tensor_type.name} in {t.name}")
            arr = np.ascontiguousarray(vals, dtype=np.float32)
            shape = arr.shape if arr.ndim > 1 else (arr.size,)
            fm.write("%s\t%d\t%s\t%d\n" % (hf, len(shape), ",".join(str(x) for x in shape), off))
            fb.write(arr.tobytes())
            off += arr.nbytes
            n_exp += 1
            types[t.tensor_type.name] = types.get(t.tensor_type.name, 0) + 1
    print("%s: %d tensors exported, bytes=%d, types=%s"
          % (os.path.basename(path), n_exp, off, types))


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for p in sys.argv[1:]:
        export(p)


if __name__ == "__main__":
    main()