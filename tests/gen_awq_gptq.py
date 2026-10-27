"""构造 AWQ / GPTQ 量化测试夹具（并生成参考反量化结果）。

为什么在本地造而不是下载：磁盘只剩几 GB，而验证要的是"落盘布局对不对"。
夹具的**打包**严格照官方落盘代码（下面标了出处），**反量化**也用官方语义
独立写一遍（脚本里的 dequant_*，与 C 的 src/qweight.c 零共享），最后把反量化
后的权重写成一份普通 HF safetensors，交给 tests/ref_llama.py（纯 numpy 前向）
算出参考 logits —— 所以端到端比对的金标准是 numpy/官方语义，不是 C 自己。

用法（在仓库根目录）：
    python sglang-c/tests/gen_awq_gptq.py \
        --src models/mistral-rand --out-root models

产出：
    models/mistral-rand-awq/          quant_method=awq,   group=8, zero_point
    models/mistral-rand-gptq/         quant_method=gptq,  group=8, sym, desc_act=False
    models/mistral-rand-gptq-act/     quant_method=gptq,  group=8, sym, desc_act=True
    sglang-c/tests/_ref_mistral_awq.json / _gptq.json / _gptq_act.json

打包/反量化的出处：
  AWQ  打包 awq/modules/linear/gemm.py::WQLinear_GEMM.from_linear
       （order_map=[0,2,4,6,1,3,5,7]；qweight[N/pack] 按 pack_num=8 装 4bit）
       反量化 awq/utils/packing_utils.py::unpack_awq + reverse_awq_order +
       dequantize_gemm（AWQ_REVERSE_ORDER=[0,4,1,5,2,6,3,7]）
  GPTQ 打包/解包 ModelCloud/GPTQModel nn_modules/qlinear/__init__.py
       （qweight 沿 K 每 pack 行一个 int32；qzeros 沿 N 每 pack 列一个 int32；
        dequant = scales[g_idx] * (q - zeros[g_idx])，不改行序）
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np

PACK = {4: 8, 8: 4}          # 每 int32 装的量化码个数
AWQ_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]     # 打包时槽位 i 放自然列 col*8+order[i]
AWQ_REVERSE = [0, 4, 1, 5, 2, 6, 3, 7]   # 解包时自然列 n 取槽位 reverse[n%8]


# ---------------------------------------------------------------- 量化


def awq_quantize(w, bits, group_size, zero_point=True):
    """AWQ 的 pseudo_quantize_tensor（asymmetric min/max，group 沿 K）。
    w: [N, K] float32 → (q [N,K] int, scales [K/G, N], zeros [K/G, N] 或 None)"""
    N, K = w.shape
    G = group_size if group_size > 0 else K
    if K % G:
        raise SystemExit(f"K={K} 不是 group_size={G} 的整数倍")
    g = w.reshape(N, K // G, G)
    max_int = 2 ** bits - 1
    if zero_point:
        maxv = g.max(axis=2, keepdims=True)
        minv = g.min(axis=2, keepdims=True)
        scales = np.clip(maxv - minv, 1e-5, None) / max_int
        zeros = np.clip(-np.round(minv / scales), 0, max_int)
        q = np.clip(np.round(g / scales) + zeros, 0, max_int)
    else:
        maxv = np.clip(np.abs(g).max(axis=2, keepdims=True), 1e-5, None)
        max_int = 2 ** (bits - 1) - 1
        scales = maxv / max_int
        zeros = None
        q = np.clip(np.round(g / scales), -(2 ** (bits - 1)), max_int) + 2 ** (bits - 1)
    # [N, K/G, G] → scales/zeros 需转成 [K/G, N]（与 gemm 落盘一致）
    sc = np.ascontiguousarray(scales[:, :, 0].T, dtype=np.float16)
    zz = np.ascontiguousarray(zeros[:, :, 0].T, dtype=np.int32) if zeros is not None else None
    return q.reshape(N, K).astype(np.int32), sc, zz


def awq_pack(q, z, bits):
    """qweight [K, N/pack] / qzeros [K/G, N/pack]（gemm 布局）
    q 是 [N, K]（HF 权重布局），而打包沿输出维 N → 先转置成 [K, N]。"""
    pack = PACK[bits]
    qt = q.T                       # [K, N]
    K = qt.shape[0]
    N = qt.shape[1]
    qw = np.zeros((K, N // pack), dtype=np.int32)
    for col in range(N // pack):
        for slot in range(pack):
            qw[:, col] |= qt[:, col * pack + AWQ_ORDER[slot]] << (bits * slot)
    qz = None
    if z is not None:
        qz = np.zeros((z.shape[0], N // pack), dtype=np.int32)
        for col in range(N // pack):
            for slot in range(pack):
                qz[:, col] |= z[:, col * pack + AWQ_ORDER[slot]] << (bits * slot)
    return qw, qz


def gptq_quantize(w, bits, group_size, groups):
    """对称 GPTQ：每组 scale=|max|/(2^(bits-1)-1)，zero=2^(bits-1)。
    groups[k] 给出自然列 k 所属的 group（desc_act 时可以不连续）。
    Scales/zeros 按 group 顺序排列。"""
    N, K = w.shape
    max_int = 2 ** (bits - 1) - 1
    ng = int(groups.max()) + 1
    scales = np.zeros((ng, N), dtype=np.float32)
    q = np.zeros((N, K), dtype=np.int32)
    for gi in range(ng):
        cols = np.where(groups == gi)[0]
        block = w[:, cols]
        sc = np.clip(np.abs(block).max(axis=1), 1e-5, None) / max_int
        scales[gi] = sc
        q[:, cols] = np.clip(np.round(block / sc[:, None]), -max_int - 1, max_int) + 2 ** (bits - 1)
    zeros = np.full((ng, N), 2 ** (bits - 1), dtype=np.int32)
    return q, scales.astype(np.float16), zeros


def gptq_pack(q, z, bits):
    """qweight [K/pack, N] / qzeros [K/G, N/pack]"""
    pack = PACK[bits]
    N, K = q.shape
    qw = np.zeros((K // pack, N), dtype=np.int32)
    for slot in range(pack):
        qw |= (q[:, slot::pack].T << (bits * slot))
    qz = np.zeros((z.shape[0], N // pack), dtype=np.int32)
    for slot in range(pack):
        qz |= (z[:, slot::pack] << (bits * slot))
    return qw, qz


# ---------------------------------------------------------------- 反量化（官方语义）


def awq_dequant(qw, qz, scales, bits, group_size):
    """packing_utils.dequantize_gemm 的逐位等价实现（含 8 元重排）"""
    pack = PACK[bits]
    K = qw.shape[0]
    N = scales.shape[1]
    G = group_size if group_size > 0 else K
    ng = qz.shape[0] if qz is not None else 0

    def unpack(t, rows, shift_of):
        out = np.zeros((rows, N), dtype=np.int32)
        for r in range(rows):
            for n in range(N):
                out[r, n] = (int(t[r, n // pack]) >> (bits * shift_of(n))) & (2 ** bits - 1)
        return out

    codes = unpack(qw, K, lambda n: AWQ_REVERSE[n % pack])
    zcodes = unpack(qz, ng, lambda n: AWQ_REVERSE[n % pack]) if qz is not None else None
    sc32 = scales.astype(np.float32)
    # w[n, k] = (q[k,n] - z[k//G, n]) * scale[k//G, n]
    w = np.zeros((N, K), dtype=np.float32)
    for k in range(K):
        g = k // G
        v = codes[k]
        if zcodes is not None:
            v = v - zcodes[g]
        w[:, k] = v * sc32[g, :]
    return w


def gptq_dequant(qw, qz, scales, g_idx, bits, group_size):
    """GPTQModel._dequantize_from_codes：scales[g_idx]*(q-zeros[g_idx])，不改行序"""
    pack = PACK[bits]
    K = qw.shape[0] * pack
    N = qw.shape[1]
    G = group_size if group_size > 0 else K
    sc32 = scales.astype(np.float32)
    # qzeros 沿 N 打包
    zc = np.zeros((qz.shape[0], N), dtype=np.int32)
    for g in range(qz.shape[0]):
        for n in range(N):
            zc[g, n] = (int(qz[g, n // pack]) >> (bits * (n % pack))) & (2 ** bits - 1)
    w = np.zeros((N, K), dtype=np.float32)
    for k in range(K):
        g = int(g_idx[k]) if g_idx is not None else k // G
        q = (qw[k // pack, :] >> (bits * (k % pack))) & (2 ** bits - 1)
        w[:, k] = (q - zc[g, :]) * sc32[g, :]
    return w


# ---------------------------------------------------------------- 写文件


def save_safetensors(path, tensors):
    """最小 safetensors 写出（F32 / F16 / I32）。

    必须逐张量 np.ascontiguousarray：astype() 默认 order='K'，对转置视图会返回
    F-contiguous 的数组，而 safetensors 按原始字节落盘（不看 strides），
    于是 scales 会被写成转置顺序——这正是最初 AWQ 夹具 logits 对不上的原因。
    """
    from safetensors.numpy import save_file
    save_file({k: np.ascontiguousarray(v) for k, v in tensors.items()}, path)


def build_variant(src, out, kind, bits, group_size):
    import struct

    cfg = json.load(open(os.path.join(src, "config.json"), encoding="utf-8"))
    E = cfg["hidden_size"]
    os.makedirs(out, exist_ok=True)

    # 读原始权重（用 safetensors 官方 numpy 接口，避免手写解析）
    from safetensors.numpy import load_file
    W = load_file(os.path.join(src, "model.safetensors"))

    tensors = {}
    n_q = 0
    rel_err = []
    for name, arr in sorted(W.items()):
        if not name.endswith(".weight") or arr.ndim != 2 or "embed_tokens" in name or \
                name == "lm_head.weight":
            # 非量化张量（norm / embed / lm_head）：存 f16 省盘
            tensors[name] = arr.astype(np.float16)
            continue
        w = arr.astype(np.float32)
        N, K = w.shape
        base = name[: -len(".weight")]
        if K % group_size:
            raise SystemExit(f"{name}: K={K} 不是 group_size={group_size} 的整数倍")
        if kind == "awq":
            q, sc, z = awq_quantize(w, bits, group_size, zero_point=True)
            qw, qz = awq_pack(q, z, bits)
            ref = awq_dequant(qw, qz, sc, bits, group_size)
            tensors[base + ".qweight"] = qw
            tensors[base + ".qzeros"] = qz
            tensors[base + ".scales"] = sc
        else:
            desc = kind == "gptq_act"
            if desc:
                # act-order：按"激活量级"（这里用 |W| 的列均值代理）排序后分组，
                # 布局保持自然列序，只有 group 归属变成不连续（正是 g_idx 的语义）
                act = np.abs(w).mean(axis=0)
                order = np.argsort(-act)
                groups = np.empty(K, dtype=np.int32)
                groups[order] = np.arange(K, dtype=np.int32) // group_size
            else:
                groups = (np.arange(K, dtype=np.int32) // group_size)
            q, sc, z = gptq_quantize(w, bits, group_size, groups)
            qw, qz = gptq_pack(q, z, bits)
            g_idx = groups.astype(np.int32)
            ref = gptq_dequant(qw, qz, sc, g_idx, bits, group_size)
            tensors[base + ".qweight"] = qw
            tensors[base + ".qzeros"] = qz
            tensors[base + ".scales"] = sc
            tensors[base + ".g_idx"] = g_idx
        n_q += 1
        den = float((w ** 2).sum())
        rel_err.append(float(np.sqrt(((ref - w) ** 2).sum() / max(den, 1e-12))))

    save_safetensors(os.path.join(out, "model.safetensors"), tensors)

    # config.json：加 quantization_config
    cfg_out = dict(cfg)
    if kind == "awq":
        cfg_out["quantization_config"] = {
            "bits": bits, "group_size": group_size, "modules_to_not_convert": None,
            "quant_method": "awq", "version": "gemm", "zero_point": True,
        }
    else:
        cfg_out["quantization_config"] = {
            "bits": bits, "group_size": group_size, "desc_act": kind == "gptq_act",
            "sym": True, "quant_method": "gptq", "format": "gptq",
        }
    json.dump(cfg_out, open(os.path.join(out, "config.json"), "w", encoding="utf-8"), indent=1)

    for f in ("tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json"):
        p = os.path.join(src, f)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(out, f))

    print("%-40s %2d 个 2D 权重量化，反量化 vs 原始 rel_rms 最大 %.4f 平均 %.4f"
          % (os.path.basename(out), n_q, max(rel_err), sum(rel_err) / len(rel_err)))
    return cfg, W, tensors


def write_ref_dir(out, cfg, W, tensors, kind, bits, group_size):
    """把官方语义反量化后的权重写成一份普通 HF 目录，给 ref_llama.py 算参考 logits"""
    refdir = out + "-ref"
    os.makedirs(refdir, exist_ok=True)
    ref_t = {}
    for name, arr in W.items():
        base = name[: -len(".weight")] if name.endswith(".weight") else None
        if base and (base + ".qweight") in tensors and arr.ndim == 2 and \
                "embed_tokens" not in name and name != "lm_head.weight":
            qw = tensors[base + ".qweight"]
            qz = tensors.get(base + ".qzeros")
            sc = tensors[base + ".scales"]
            if kind == "awq":
                ref_t[name] = awq_dequant(qw, qz, sc, bits, group_size)
            else:
                ref_t[name] = gptq_dequant(qw, qz, sc, tensors[base + ".g_idx"], bits, group_size)
        else:
            ref_t[name] = arr.astype(np.float16)
    save_safetensors(os.path.join(refdir, "model.safetensors"), ref_t)
    c2 = dict(cfg)
    c2.pop("quantization_config", None)
    json.dump(c2, open(os.path.join(refdir, "config.json"), "w", encoding="utf-8"), indent=1)
    for f in ("tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json"):
        p = os.path.join(out, f)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(refdir, f))
    return refdir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="models/mistral-rand")
    ap.add_argument("--out-root", default="models")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--ref-script", default="sglang-c/tests/ref_llama.py")
    ap.add_argument("--ref-out", default="sglang-c/tests")
    ap.add_argument("--variants", default="awq,gptq,gptq_act")
    args = ap.parse_args()

    made = []
    for kind in args.variants.split(","):
        kind = kind.strip()
        name = {"awq": "awq", "gptq": "gptq", "gptq_act": "gptq-act"}[kind]
        out = os.path.join(args.out_root, "mistral-rand-" + name)
        cfg, W, tensors = build_variant(args.src, out, kind, args.bits, args.group_size)
        refdir = write_ref_dir(out, cfg, W, tensors, kind, args.bits, args.group_size)
        refjson = os.path.join(args.ref_out, "_ref_mistral_%s.json" % name.replace("-", "_"))
        subprocess.check_call([sys.executable, args.ref_script, refdir, refjson])
        print("  ref json ->", refjson)
        shutil.rmtree(refdir)   # 只需要参考 json，ref 目录用完即删（省盘）
        made.append(out)
    print("done:", made)


if __name__ == "__main__":
    main()