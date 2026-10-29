"""从 models/mistral-rand-ct4（4bit）或直接从 fp 权重（8bit）生成 **marlin 布局**夹具。

先说明为什么这个东西存在：上游生态里**没有**把 AWQ/GPTQ 权重以 marlin 布局落盘的格式 ——
vLLM 的 `awq_marlin` / `gptq_marlin` 是在 `process_weights_after_loading` 里把标准落盘
权重重排成 marlin（kernel 直接吃的那份内存形态），TensorRT-LLM / modelopt 的
`W4A16_AWQ` 也有自己的 interleave，跟 marlin 不是一回事。所以这份夹具的作用是：
把「marlin ABI 张量」真实造出来（打包用 Marlin 官方 pack，已与 vLLM repack kernel 逐字节
对拍过，见 tests/marlin_layout.py），让 C 的解包路径能被端到端验证。

码直接取自 ct4 夹具（同一批 4bit 码），所以量化误差与 ct4 完全一致；
反量化后的权重由本脚本另存一份 ref 目录交给 ref_llama.py 算参考 logits。

marlin 的 tile 是 16(k) x 64(n)：K%16 且 N%64 的权重才重排，
其余（本夹具里的 k_proj/v_proj，N=32）保留 fp16 —— 与 vLLM 需要 pad N 的情况一致。

用法：
  python tests/make_marlin_fixture.py                      # 4bit（默认，从 ct4 的码 + Marlin 官方 pack）
  python tests/make_marlin_fixture.py --bits 8 --out ../models/mistral-rand-marlin8
      --ref-out tests/_ref_mistral_marlin8.json
  # 8bit：Marlin 官方 Python 只做 4bit，所以打包用 vLLM 的 GPTQ repack kernel
  # （与 C 的闭式索引是两份独立实现，已在 tests/marlin_layout.py 逐字节对账）；
  # 同时带一份对称零点（= 2^(bits-1) = 128），与真实 GPTQ 8bit 检查点一致。
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from gen_awq_gptq import save_safetensors  # noqa: E402
from make_ct_fixture import quantize_group_sym, unpack_from_int32_ct  # noqa: E402
from marlin_layout import (gptq_pack_natural, marlin_permute_scales,  # noqa: E402
                          marlin_ref_pack, marlin_zero_points,
                          vllm_gptq_marlin_repack)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="../models/mistral-rand")
    ap.add_argument("--ct", default="../models/mistral-rand-ct4")
    ap.add_argument("--out", default="../models/mistral-rand-marlin")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--ref-script", default="tests/ref_llama.py")
    ap.add_argument("--ref-out", default="tests/_ref_mistral_marlin.json")
    args = ap.parse_args()

    from safetensors.numpy import load_file

    bits, G = args.bits, args.group_size
    cfg = json.load(open(os.path.join(args.src, "config.json"), encoding="utf-8"))
    W = load_file(os.path.join(args.src, "model.safetensors"))
    C = load_file(os.path.join(args.ct, "model.safetensors"))
    os.makedirs(args.out, exist_ok=True)

    tensors, ref_t = {}, {}
    n_q = n_skip = 0
    for name, arr in sorted(W.items()):
        base = name[: -len(".weight")] if name.endswith(".weight") else name
        quant_ok = (name.endswith(".weight") and arr.ndim == 2 and "embed_tokens" not in name
                    and name != "lm_head.weight")
        if not quant_ok:
            tensors[name] = arr.astype(np.float16)
            ref_t[name] = arr.astype(np.float16)
            continue
        wp = C.get(base + ".weight_packed")
        if bits == 4 and wp is not None:
            sc = C[base + ".weight_scale"].astype(np.float32)     # [N, K/G]
            shape = C[base + ".weight_shape"]
            N, K = int(shape[0]), int(shape[1])
            src_codes = unpack_from_int32_ct(wp, bits, K)          # [N, K] 有符号码
        else:
            # 8bit（或没有 ct 夹具时）：直接从 fp 权重量化
            qc, sc = quantize_group_sym(arr.astype(np.float32), bits, G)   # [N,K], [N,K/G]
            N, K = arr.shape
            src_codes = qc.astype(np.int32) - (1 << (bits - 1))
        if K % 16 or N % 64:
            # marlin 的 tile 装不下（N=32 这类），保留 fp16：与 vLLM 需要 pad N 的情形对应
            print("  %-42s N=%d 不是 64 的倍数 → 保留 fp16" % (name, N))
            tensors[name] = arr.astype(np.float16)
            ref_t[name] = arr.astype(np.float16)
            n_skip += 1
            continue
        qcode = (src_codes + (1 << (bits - 1))).astype(np.uint8)   # marlin 用无符号码
        if bits == 4:
            qw = marlin_ref_pack(np.ascontiguousarray(qcode.T))    # [K/16, N*16/pack]
        else:
            # marlin 官方 Python 只做 4bit；8bit 用 vLLM 的 GPTQ repack kernel（与 C 的闭式
            # 索引是两份独立实现，已逐字节对账）
            pf = 32 // bits
            qw = vllm_gptq_marlin_repack(gptq_pack_natural(np.ascontiguousarray(qcode.T), bits),
                                         K, N, bits).reshape(K // 16, N * 16 // pf)
        sc_nat = np.ascontiguousarray(sc.T)                   # [K/G, N]
        sc_m = marlin_permute_scales(sc_nat, K, N, G)
        tensors[base + ".qweight"] = qw.view(np.int32)
        tensors[base + ".scales"] = sc_m.astype(np.float16)
        if bits != 4:
            # 8bit 走 GPTQ：带一份对称零点（= 2^(bits-1) = 128），跟真实 GPTQ 8bit
            # 检查点一样，顺便让端到端路径覆盖 8bit 的零点解包（4 元 interleave）
            zp_nat = np.full((K // G, N), 1 << (bits - 1), dtype=np.int32)
            zp_m = marlin_zero_points(zp_nat, K // G, N, bits)
            tensors[base + ".qzeros"] = zp_m.view(np.int32)
        ref_t[name] = (src_codes.astype(np.float32) * sc.repeat(G, axis=1)).astype(np.float16)
        n_q += 1

    save_safetensors(os.path.join(args.out, "model.safetensors"), tensors)

    c2 = dict(cfg)
    # 4bit 用 AWQ（AWQ 本身就只存 4bit）；8bit 用 GPTQ（gptq 的 bits=8 是真实存在的配置）
    c2["quantization_config"] = {
        "quant_method": "awq" if bits == 4 else "gptq",
        "bits": bits,
        "group_size": G,
        "zero_point": False,
        ("version" if bits == 4 else "format"): "marlin",  # 本仓读它 → 先按 marlin 布局读
    }
    if bits != 4:
        c2["quantization_config"]["sym"] = True
        c2["quantization_config"]["zero_point"] = True
        c2["quantization_config"]["desc_act"] = False
    json.dump(c2, open(os.path.join(args.out, "config.json"), "w", encoding="utf-8"), indent=1)
    for f in ("tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json"):
        p = os.path.join(args.src, f)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(args.out, f))

    # 反量化后的参考目录 → ref_llama.py → ref json
    refdir = args.out + "-ref"
    os.makedirs(refdir, exist_ok=True)
    save_safetensors(os.path.join(refdir, "model.safetensors"), ref_t)
    c3 = dict(cfg)
    c3.pop("quantization_config", None)
    json.dump(c3, open(os.path.join(refdir, "config.json"), "w", encoding="utf-8"), indent=1)
    for f in ("tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json"):
        p = os.path.join(args.src, f)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(refdir, f))
    subprocess.check_call([sys.executable, args.ref_script, refdir, args.ref_out])
    shutil.rmtree(refdir)

    print("%-28s %d 个权重重排成 marlin（另 %d 个 N%%64 != 0 保留 fp16）"
          % (os.path.basename(args.out), n_q, n_skip))
    print("ref json ->", args.ref_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())