"""从 models/mistral-rand 生成 gpt-oss 风格的 mxfp4 夹具。

落盘语义按 transformers 的官方实现：
  src/transformers/integrations/mxfp4.py
    _convert_moe_packed_tensors(blocks, scales)
       blocks: uint8 [..., K/32, 16]   （每 32 个元素压成 16 字节）
       scales: uint8 [..., K/32]       （e8m0：值 = 2^(x-127)）
       sub[:, 0::2] = lut[blk & 0x0F]；sub[:, 1::2] = lut[blk >> 4]
       torch.ldexp(sub, exp)  → 乘 2^(x-127)
       FP4_VALUES = [0,.5,1,1.5,2,3,4,6, -0,-.5,-1,-1.5,-2,-3,-4,-6]
  即：低半字节在前、每 32 元素一个 e8m0 scale、码书是 OCP MXFP4 的 e2m1。

本脚本把 2D 线性权重按同样语义量化 + 打包，并生成一份反量化参考给 ref_llama.py。

用法：python tests/make_mxfp4_fixture.py
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

FP4 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def nearest_fp4(x):
    """幅值 → e2m1 的 3 bit 码（0..7）"""
    idx = np.argmin(np.abs(np.abs(x)[..., None] - FP4[None, :]), axis=-1)
    return idx.astype(np.uint8)


def quantize_mxfp4(w):
    """w [N,K] f32 → blocks uint8 [N, K/2]（等价于 [N, K/32, 16]）、scales uint8 [N, K/32]"""
    N, K = w.shape
    assert K % 32 == 0, K
    wb = w.reshape(N, K // 32, 32)
    amax = np.abs(wb).max(axis=2, keepdims=True)          # [N, nb, 1]
    # scale = 2^e，使 amax/scale 落在 [4,8) 区间（e2m1 最大 6，留一档余量）
    e = np.floor(np.log2(np.maximum(amax, 1e-30) / 6.0)).astype(np.int32)
    sc = np.exp2(e.astype(np.float32))
    mag = nearest_fp4(wb / sc)
    sign = (wb < 0).astype(np.uint8)
    code = (mag | (sign << 3)).astype(np.uint8)            # [N, nb, 32]
    # 打包：每字节低半字节 = 偶下标
    lo = code[..., 0::2]
    hi = code[..., 1::2]
    blocks = ((hi << 4) | lo).astype(np.uint8).reshape(N, K // 2)
    scales = (e + 127).astype(np.uint8).reshape(N, K // 32)
    return blocks, scales


def dequant_mxfp4(blocks, scales, K):
    """按官方语义反量化（独立实现，用于生成参考）"""
    N = blocks.shape[0]
    nb = K // 32
    b = blocks.reshape(N, nb, 16)
    lo = b & 0x0F
    hi = b >> 4
    code = np.empty((N, nb, 32), dtype=np.uint8)
    code[..., 0::2] = lo
    code[..., 1::2] = hi
    exp = scales.astype(np.int32).reshape(N, nb, 1) - 127
    mag = FP4[code & 7]
    sign = np.where((code & 8) != 0, -1.0, 1.0).astype(np.float32)
    return np.ldexp(mag * sign, exp).reshape(N, K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="../models/mistral-rand")
    ap.add_argument("--out", default="../models/mistral-rand-mxfp4")
    ap.add_argument("--ref-script", default="tests/ref_llama.py")
    ap.add_argument("--ref-out", default="tests/_ref_mistral_mxfp4.json")
    args = ap.parse_args()

    from safetensors.numpy import load_file

    cfg = json.load(open(os.path.join(args.src, "config.json"), encoding="utf-8"))
    W = load_file(os.path.join(args.src, "model.safetensors"))
    os.makedirs(args.out, exist_ok=True)

    tensors, ref_t = {}, {}
    n_q = 0
    rel = []
    for name, arr in sorted(W.items()):
        if not name.endswith(".weight") or arr.ndim != 2 or "embed_tokens" in name or \
                name == "lm_head.weight":
            tensors[name] = arr.astype(np.float16)
            ref_t[name] = arr.astype(np.float16)
            continue
        w = arr.astype(np.float32)
        N, K = w.shape
        if K % 32:
            raise SystemExit("%s: K=%d 不是 32 的整数倍" % (name, K))
        base = name[: -len(".weight")]
        blocks, scales = quantize_mxfp4(w)
        tensors[base + ".weight"] = blocks
        tensors[base + ".scales"] = scales
        dq = dequant_mxfp4(blocks, scales, K).astype(np.float16)
        ref_t[name] = dq
        den = float((w ** 2).sum())
        rel.append(float(np.sqrt(((dq.astype(np.float32) - w) ** 2).sum() / max(den, 1e-12))))
        n_q += 1

    save_safetensors(os.path.join(args.out, "model.safetensors"), tensors)
    c2 = dict(cfg)
    c2["quantization_config"] = {"quant_method": "mxfp4"}
    json.dump(c2, open(os.path.join(args.out, "config.json"), "w", encoding="utf-8"), indent=1)
    for f in ("tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json"):
        p = os.path.join(args.src, f)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(args.out, f))

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
    print("%-28s %d 个权重 mxfp4，反量化 vs 原始 rel_rms 最大 %.4f"
          % (os.path.basename(args.out), n_q, max(rel)))
    print("ref json ->", args.ref_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())