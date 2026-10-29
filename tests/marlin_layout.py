"""marlin 布局（AWQ/GPTQ 权重的 marlin ABI）的参考实现与三方差分验证。

为什么要这份文件：`src/qweight.c` 要能解 marlin 布局，所以这套布局必须来自**权威实现**、
且用**两个互相独立的参考**交叉验证，否则很容易写出「自洽但错位」的解包（静默数值错误）。

三个来源：

  A. Marlin 作者官方 Python 打包
     https://github.com/IST-DASLab/marlin/blob/master/marlin/__init__.py
     `_get_perms()` + `Layer.pack()`（在 1024 元素的 16x64 tile 上做 _perm 重排，
      再 `q |= res[:, i::8] << 4*i`）—— 这是 kernel 真正吃的 B。

  B. vLLM 的 AWQ→marlin 重排 CUDA kernel
     https://github.com/vllm-project/vllm/blob/main/csrc/quantization/marlin/awq_marlin_repack.cu
     这里逐行移植到 numpy（4bit / 8bit，is_a_8bit=False）：从**标准 AWQ 落盘布局**
     （qweight [K, N/pack]，含 AWQ 的 8/4 元重排）重排成 marlin B。

  B'. vLLM 的 GPTQ→marlin 重排 CUDA kernel（只用于 8bit 的交叉对账）
     https://github.com/vllm-project/vllm/blob/main/csrc/quantization/marlin/gptq_marlin_repack.cu
     输入是 GPTQ 自然布局（qweight [K/pack, N]，沿 K 打包），shared memory 组织也不同，
     但输出必须与 B 逐字节相同 —— 两份独立实现。
     **注意：Marlin 作者的官方 Python `Layer.pack` 只做 4bit（maxq = 2^4-1），
     所以 8bit 只有 B、B' 两份参考 + C 的闭式索引。**

  C. 我们 C 里要用的闭式索引（storage word/nibble → (k, n)）
     从 B 的 kernel 逻辑化简出来的公式，见 `marlin_index_map()`。

A 与 B 是两拨人各写一遍的同一个布局；C 是 B 的代数化简。三者对同一份随机数据必须
产出**逐字节相同**的 B —— `python tests/marlin_layout.py` 就是在验这件事，
顺带验 scale 的 64 列置换、以及 AWQ 零点（awq_to_marlin_zero_points）的二次重排。

跑法：python tests/marlin_layout.py
"""
import numpy as np

# ---------------------------------------------------------------------------
# A. Marlin 官方 Python（IST-DASLab）——逐字移植 _get_perms / Layer.pack 的打包部分
# ---------------------------------------------------------------------------


def marlin_ref_perms():
    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in [0, 1]:
            for row in [2 * (i % 4), 2 * (i % 4) + 1,
                        2 * (i % 4 + 4), 2 * (i % 4 + 4) + 1]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm.extend([p + 256 * j for p in perm1])
    perm = np.array(perm)
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    perm = perm.reshape((-1, 8))[:, interleave].ravel()
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return perm, scale_perm, scale_perm_single


MARLIN_PERM, SCALE_PERM, SCALE_PERM_SINGLE = marlin_ref_perms()


def marlin_ref_pack(codes):
    """codes: uint8/int32 [K, N]（已经是含 +8 偏移的无符号码），返回 B [K/16, N*16/8]。

    对应 IST-DASLab/marlin `Layer.pack`（它自己也在里面量化：round(w/s)+8，
    这里直接喂码，等价）。**只做打包，不管 scale。**"""
    k, n = codes.shape
    tile = 16
    assert k % tile == 0 and n % tile == 0, (k, n)
    w = codes.astype(np.int32)
    w = w.reshape((k // tile, tile, n // tile, tile)).transpose(0, 2, 1, 3)
    w = w.reshape((k // tile, n * tile))
    res = w.reshape((-1, MARLIN_PERM.size))[:, MARLIN_PERM].reshape(w.shape)
    q = np.zeros((res.shape[0], res.shape[1] // 8), dtype=np.uint32)
    for i in range(8):
        q |= (res[:, i::8].astype(np.uint32) << np.uint32(4 * i))
    return q


# ---------------------------------------------------------------------------
# B. vLLM awq_marlin_repack 的 numpy 移植（4bit / 8bit，is_a_8bit=False）
# ---------------------------------------------------------------------------


def awq_pack_natural(codes, bits=4):
    """自然列序的码 → 标准 AWQ 落盘 qweight [K, N/pack]（含 AWQ 的组内重排）"""
    k, n = codes.shape
    pf = 32 // bits
    undo = [0, 4, 1, 5, 2, 6, 3, 7] if bits == 4 else [0, 2, 1, 3]
    qw = np.zeros((k, n // pf), dtype=np.uint32)
    for nn in range(n):
        qw[:, nn // pf] |= (codes[:, nn].astype(np.uint32) << np.uint32(bits * undo[nn % pf]))
    return qw.astype(np.int32)


def gptq_pack_natural(codes, bits=8):
    """自然序的码 → GPTQ 落盘 qweight [K/pack, N]（沿 K 打包，无重排）"""
    k, n = codes.shape
    pf = 32 // bits
    qw = np.zeros((k // pf, n), dtype=np.uint32)
    for kk in range(k):
        qw[kk // pf, :] |= (codes[kk, :].astype(np.uint32) << np.uint32(bits * (kk % pf)))
    return qw.astype(np.int32)


def vllm_awq_marlin_repack(qw, size_k, size_n, num_bits=4):
    """逐行移植 csrc/quantization/marlin/awq_marlin_repack.cu（is_a_8bit=False）"""
    pf = 32 // num_bits
    tile_n = 64
    tile_k = 16
    mask = (1 << num_bits) - 1
    k_tiles = size_k // tile_k
    n_tiles = size_n // tile_n
    qwcols = size_n // pf
    qwf = qw.ravel()
    tile_size = tile_k * tile_n // pf
    out = np.zeros(k_tiles * n_tiles * tile_size, dtype=np.uint32)
    offs = [0, 1, 8, 9]
    undo = [0, 4, 1, 5, 2, 6, 3, 7] if num_bits == 4 else [0, 2, 1, 3]
    for kt in range(k_tiles):
        for nt in range(n_tiles):
            base = (kt * n_tiles + nt) * tile_size
            first_n_packed = nt * tile_n // pf
            first_k = kt * tile_k
            for th in range(32):
                for wid in range(4):
                    tc_col = th // 4
                    tc_row = (th % 4) * 2
                    cur_n = wid * 16 + tc_col
                    cur_n_packed = cur_n // pf
                    pos = undo[cur_n % pf]
                    vals = [0] * 8
                    for i in range(4):
                        ce = tc_row + offs[i]
                        s0 = qwf[(first_k + ce) * qwcols + first_n_packed + cur_n_packed]
                        s1 = qwf[(first_k + ce) * qwcols + first_n_packed +
                                 (8 // pf) + cur_n_packed]
                        vals[i] = (int(s0) >> (pos * num_bits)) & mask
                        vals[4 + i] = (int(s1) >> (pos * num_bits)) & mask
                    if num_bits == 4:
                        pack_idx = [0, 2, 4, 6, 1, 3, 5, 7]
                        res = 0
                        for i in range(8):
                            res |= vals[pack_idx[i]] << (i * 4)
                        out[base + th * 4 + wid] = res
                    else:
                        pack_idx = [0, 2, 1, 3]
                        res1 = res2 = 0
                        for i in range(4):
                            res1 |= vals[pack_idx[i]] << (i * 8)
                            res2 |= vals[4 + pack_idx[i]] << (i * 8)
                        out[base + th * 8 + wid * 2 + 0] = res1
                        out[base + th * 8 + wid * 2 + 1] = res2
    return out


def vllm_gptq_marlin_repack(qw, size_k, size_n, num_bits=8):
    """逐行移植 csrc/quantization/marlin/gptq_marlin_repack.cu（is_a_8bit=False）

    输入 qw 是 GPTQ 自然布局 [K/pack, N]（沿 K 打包）。输出应与 B 相同。"""
    pf = 32 // num_bits
    tile_n = 64
    tile_k = 16
    mask = (1 << num_bits) - 1
    k_tiles = size_k // tile_k
    n_tiles = size_n // tile_n
    tile_ints = tile_k // pf
    tile_size = tile_k * tile_n // pf
    out = np.zeros(k_tiles * n_tiles * tile_size, dtype=np.uint32)
    offs = [0, 1, 8, 9]
    for kt in range(k_tiles):
        for nt in range(n_tiles):
            base = (kt * n_tiles + nt) * tile_size
            first_kpf = kt * tile_k // pf
            first_n = nt * tile_n
            for th in range(32):
                for wid in range(4):
                    tc_col = th // 4
                    tc_row = (th % 4) * 2
                    cur_n = wid * 16 + tc_col
                    b1 = [int(qw[first_kpf + i, first_n + cur_n]) for i in range(tile_ints)]
                    b2 = [int(qw[first_kpf + i, first_n + cur_n + 8]) for i in range(tile_ints)]
                    vals = [0] * 8
                    for i in range(4):
                        ce = tc_row + offs[i]
                        ci, cp = ce // pf, ce % pf
                        vals[i] = (b1[ci] >> (cp * num_bits)) & mask
                        vals[4 + i] = (b2[ci] >> (cp * num_bits)) & mask
                    if num_bits == 4:
                        pack_idx = [0, 2, 4, 6, 1, 3, 5, 7]
                        res = 0
                        for i in range(8):
                            res |= vals[pack_idx[i]] << (i * 4)
                        out[base + th * 4 + wid] = res
                    else:
                        pack_idx = [0, 2, 1, 3]
                        res1 = res2 = 0
                        for i in range(4):
                            res1 |= vals[pack_idx[i]] << (i * 8)
                            res2 |= vals[4 + pack_idx[i]] << (i * 8)
                        out[base + th * 8 + wid * 2 + 0] = res1
                        out[base + th * 8 + wid * 2 + 1] = res2
    return out


# ---------------------------------------------------------------------------
# C. 闭式索引：marlin storage → 自然 (k, n)
# ---------------------------------------------------------------------------


def marlin_index_map(bits=4, n_tile=64):
    """对**一个 16x64 的 tile** 返回 (word, 码位) → (k_off, n_off)。

    由 B 的 kernel 化简：
      4bit（`p = th_id*4 + warp_id`，j = 第几个 nibble）：
        k_off = 2*((p>>2)&3) + (8 if j&1 else 0) + (1 if j>=4 else 0)
        n_off = 16*(p&3) + (p>>4) + (8 if j&2 else 0)
      8bit（每个线程写两个 word：`p = th_id*8 + warp_id*2 + half`，j = 第几个 byte）：
        k_off = 2*((p>>3)&3) + [0,8,1,9][j]
        n_off = 16*((p>>1)&3) + (p>>5) + (8 if p&1 else 0)
    """
    pf = 32 // bits
    ret = {}
    words = 16 * n_tile // pf
    off8 = [0, 8, 1, 9]
    for p in range(words):
        for j in range(pf):
            if bits == 4:
                k_off = 2 * ((p >> 2) & 3) + (8 if (j & 1) else 0) + (1 if j >= 4 else 0)
                n_off = 16 * (p & 3) + (p >> 4) + (8 if (j & 2) else 0)
            else:
                k_off = 2 * ((p >> 3) & 3) + off8[j]
                n_off = 16 * ((p >> 1) & 3) + (p >> 5) + (8 if (p & 1) else 0)
            ret[(p, j)] = (k_off, n_off)
    return ret


def closed_form_pack(codes, bits=4):
    """用闭式索引（纯 scatter，不碰 _perm / kernel 细节）打包一份 B，用来和 A、B、B' 对账"""
    k, n = codes.shape
    pf = 32 // bits
    assert k % 16 == 0 and n % 64 == 0, (k, n)
    tw = 16 * 64 // pf
    b = np.zeros((k // 16, n * 16 // pf), dtype=np.uint32)
    imap = marlin_index_map(bits)
    for kt in range(k // 16):
        for nt in range(n // 64):
            for p in range(tw):
                for j in range(pf):
                    ko, no = imap[(p, j)]
                    b[kt, nt * tw + p] |= (np.uint32(codes[kt * 16 + ko, nt * 64 + no])
                                            << np.uint32(bits * j))
    return b


# ---------------------------------------------------------------------------
# scale / zero-point 的 64 列置换（vLLM marlin_utils.py 的移植）
# ---------------------------------------------------------------------------


def marlin_permute_scales(s, size_k, size_n, group_size):
    """[K/G, N] 自然列序 → marlin（vLLM 加载后的内存布局）"""
    if group_size < size_k and group_size != -1:
        s = s.reshape((-1, len(SCALE_PERM)))[:, SCALE_PERM]
    else:
        s = s.reshape((-1, len(SCALE_PERM_SINGLE)))[:, SCALE_PERM_SINGLE]
    return s.reshape((-1, size_n)).copy()


def unpermute_scales(sp, size_n):
    """marlin 布局 → 自然列序（C 里要做的逆操作；纯 scatter）"""
    sp = sp.reshape((-1, size_n))
    out = np.empty_like(sp)
    for r in range(sp.shape[0]):
        blk = 0
        while blk < size_n:
            for p in range(64):
                out[r, blk + SCALE_PERM[p]] = sp[r, blk + p]
            blk += 64
    return out


def pack_cols(q_w, num_bits):
    pf = 32 // num_bits
    s_k, s_n = q_w.shape
    assert s_n % pf == 0
    q = np.zeros((s_k, s_n // pf), dtype=np.uint32)
    for i in range(pf):
        q |= (q_w[:, i::pf].astype(np.uint32) << np.uint32(num_bits * i))
    return q


def marlin_zero_points(zp, size_k, size_n, num_bits=4):
    """vLLM marlin_utils.marlin_zero_points 的移植（zp 已是 [G, N] 自然列序）"""
    zp = zp.reshape((-1, len(SCALE_PERM)))[:, SCALE_PERM]
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7]) if num_bits == 4 else np.array([0, 2, 1, 3])
    zp = zp.reshape((-1, len(interleave)))[:, interleave].ravel()
    zp = zp.reshape((-1, size_n)).copy()
    return pack_cols(zp, num_bits)


def unpermute_zero_points(zp_m, groups, size_n, num_bits=4):
    """marlin 零点（[G, N/pack]）→ 自然列序 [G, N]；纯 scatter，C 里照这个做。

    正向链：n → p = 8*(n%8) + (n/8)（scale_perm 的位置）→ 组内 8 元 interleave
    → pack_cols 每 8 列一 word。这里反向做一遍，只为验证 C 的公式。
    """
    pf = 32 // num_bits
    interleave = [0, 2, 4, 6, 1, 3, 5, 7] if num_bits == 4 else [0, 2, 1, 3]
    gsz = len(interleave)
    out = np.zeros((groups, size_n), dtype=np.int32)
    for g in range(groups):
        for n in range(size_n):
            blk = (n // 64) * 64
            p_local = 8 * (n % 8) + ((n % 64) // 8)   # zp1 里的位置（scale_perm 的逆）
            f = blk + p_local                          # 该行里的 flat 位置
            j = f % gsz
            pos = interleave.index(j)                  # 组内 interleave 后的槽位
            f2 = gsz * (f // gsz) + pos
            col, nib = f2 // pf, f2 % pf
            out[g, n] = (int(zp_m[g, col]) >> (nib * num_bits)) & ((1 << num_bits) - 1)
    return out


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------


def _codes(k, n, seed=0, bits=4):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 1 << bits, size=(k, n), dtype=np.uint8)


def emit_unit(path):
    """写一份给 C 单测（tests/test_qweight.c）用的定值用例：
    同一份数据三种打包（marlin 权重 / marlin scales / marlin 零点）+ 期望的 [N,K] f32。
    期望值在 Python 里按自然列序算（(q - z) * s），C 那边必须一模一样。"""
    import json

    cases = []
    rng = np.random.default_rng(2024)
    K, N, G = 32, 64, 8
    for bits in (4, 8):
        pf = 32 // bits
        for name, has_z in (("sym", 0), ("asym", 1)):
            codes = rng.integers(0, 1 << bits, size=(K, N), dtype=np.uint8)   # [k, n]
            sc = (rng.random((K // G, N)).astype(np.float32) * 0.05 + 0.01)
            zp = (rng.integers(0, 1 << bits, size=(K // G, N)).astype(np.int32) if has_z
                  else None)

            if bits == 4:
                qw = marlin_ref_pack(codes)                        # [K/16, N*2]
            else:
                # marlin 官方 Python 只做 4bit，8bit 用两份 vLLM kernel 互相对账
                qw = closed_form_pack(codes, 8)
            qw_v = vllm_awq_marlin_repack(awq_pack_natural(codes, bits), K, N, bits) \
                .reshape(K // 16, N * 16 // pf)
            qw_g = vllm_gptq_marlin_repack(gptq_pack_natural(codes, bits), K, N, bits) \
                .reshape(K // 16, N * 16 // pf)
            assert np.array_equal(qw, qw_v), (bits, name, "awq repack")
            assert np.array_equal(qw, qw_g), (bits, name, "gptq repack")
            sc_m = marlin_permute_scales(sc, K, N, G)
            zp_m = marlin_zero_points(zp, K // G, N, bits) if has_z else None
            # 反向确认：C 要能用 sc_m / zp_m 恢复出自然列序
            assert np.array_equal(unpermute_scales(sc_m, N), sc), (bits, name, "scales")
            if has_z:
                assert np.array_equal(unpermute_zero_points(zp_m, K // G, N, bits), zp), \
                    (bits, name, "zp")

            exp = np.zeros((N, K), dtype=np.float32)                   # 行 = out_features
            for k in range(K):
                for n in range(N):
                    q = int(codes[k, n]) - (int(zp[k // G, n]) if has_z else 1 << (bits - 1))
                    exp[n, k] = np.float32(q) * sc[k // G, n]
            cases.append({
                "name": "%s%d" % (name, bits), "K": K, "N": N, "G": G, "bits": bits,
                "has_z": int(has_z),
                "qw_rows": K // 16, "qw_cols": qw.shape[1],
                "qweight": [int(x) for x in qw.view(np.int32).ravel()],
                "qz_rows": (K // G) if has_z else 0,
                "qz_cols": (zp_m.shape[1] if has_z else 0),
                "qzeros": ([int(x) for x in zp_m.view(np.int32).ravel()] if has_z else []),
                "sc_rows": K // G, "sc_cols": N,
                "scales": [float(x) for x in sc_m.ravel()],
                "expected": [float(x) for x in exp.ravel()],
            })
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"cases": cases}, f)
    print("marlin 单测定值 -> %s（%d 个用例）" % (path, len(cases)))


def main():
    fails = 0
    def check(cond, msg):
        nonlocal fails
        print(("PASS " if cond else "FAIL ") + msg)
        if not cond:
            fails += 1

    # 1) 三方差分（4bit：Marlin 官方 pack / vLLM AWQ kernel / 闭式索引）
    for (k, n) in [(32, 64), (64, 128), (128, 256), (32, 128)]:
        codes = _codes(k, n)
        a = marlin_ref_pack(codes)
        b = vllm_awq_marlin_repack(awq_pack_natural(codes), k, n, 4).reshape(k // 16, n * 2)
        c = closed_form_pack(codes)
        check(a.shape == b.shape == c.shape,
              "4bit 形状 K=%d N=%d: ref %s / vllm %s / closed %s" % (k, n, a.shape, b.shape, c.shape))
        check(np.array_equal(a, b), "4bit K=%d N=%d: Marlin 官方 pack == vLLM repack kernel" % (k, n))
        check(np.array_equal(a, c), "4bit K=%d N=%d: Marlin 官方 pack == 我们的闭式索引" % (k, n))

    # 1') 8bit：Marlin 官方 Python 只做 4bit，所以用 vLLM 的两份独立 kernel 互相对账
    #     （AWQ 的从 [K, N/pack] 重排、GPTQ 的从 [K/pack, N] 重排，sh 组织也不同）
    for (k, n) in [(32, 64), (64, 128), (128, 256), (32, 128)]:
        pw = 4
        codes = _codes(k, n, seed=7, bits=8)
        b = vllm_awq_marlin_repack(awq_pack_natural(codes, 8), k, n, 8).reshape(k // 16, n * 16 // pw)
        g = vllm_gptq_marlin_repack(gptq_pack_natural(codes, 8), k, n, 8).reshape(k // 16,
                                                                                n * 16 // pw)
        c = closed_form_pack(codes, 8)
        check(b.shape == g.shape == c.shape,
              "8bit 形状 K=%d N=%d: awq %s / gptq %s / closed %s" % (k, n, b.shape, g.shape, c.shape))
        check(np.array_equal(b, g), "8bit K=%d N=%d: vLLM AWQ repack == vLLM GPTQ repack" % (k, n))
        check(np.array_equal(b, c), "8bit K=%d N=%d: vLLM repack == 我们的闭式索引" % (k, n))

    # 2) scale 置换可逆
    for n in (64, 128):
        s = np.arange(2 * n, dtype=np.float32).reshape(2, n)
        sp = marlin_permute_scales(s, 64, n, 8)
        back = unpermute_scales(sp, n)
        check(np.array_equal(back, s), "scale 置换可逆 N=%d" % n)

    # 3) AWQ 零点置换可逆（4bit / 8bit）
    for bits in (4, 8):
        for n in (64, 128):
            zp = np.arange(8 * n, dtype=np.int32).reshape(8, n) % (1 << bits)
            zpm = marlin_zero_points(zp, 8, n, bits)
            back = unpermute_zero_points(zpm, 8, n, bits)
            check(np.array_equal(back, zp),
                  "AWQ 零点置换可逆 %dbit N=%d（marlin 形状 %s）" % (bits, n, zpm.shape))

    print()
    if fails:
        print("%d 项失败" % fails)
        return 1
    print("全部通过：marlin 布局的多种实现对同一份随机数据产出逐字节相同的 B（4bit 三方、8bit 两份 kernel + 闭式）")
    return 0

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2 and sys.argv[1] == "--emit-unit":
        emit_unit(sys.argv[2])
        raise SystemExit(0)
    raise SystemExit(main())