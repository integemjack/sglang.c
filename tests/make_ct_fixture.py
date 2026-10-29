"""从 models/mistral-rand 生成 compressed-tensors W4A16（pack-quantized）夹具。

落盘格式按 vllm-project/compressed-tensors 的官方实现：
  src/compressed_tensors/compressors/pack_quantized/helpers.py
    pack_to_int32 / unpack_from_int32

  - 4bit：把 K 个元素按顺序塞进 int32 的位流（每 int32 = 8 个元素）；
    展开时把 int32 看成 4 个字节，**每字节低半字节是偶数下标、高半字节是奇数下标**，
    拼成一维取前 K 个，再整体减 1<<(bits-1)=8 得到有符号值。
  - 张量名：<base>.weight_packed (I32 [N, K/8])、<base>.weight_scale (F16 [N, K/G])、
    <base>.weight_shape (I64 [2] = [N, K])
  - config.quantization_config = {quant_method: compressed-tensors, format: pack-quantized,
    config_groups: {group_0: {weights: {num_bits: 4, type: int, strategy: group,
    group_size: G, symmetric: true}}}}

同时生成一份反量化后的参考目录（ref），用 ref_llama.py 算出参考 logits。

用法：python tests/make_ct_fixture.py [--group-size 8]
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


def quantize_group_sym(w, bits, group_size):
    """对称 per-group 量化 → 无符号码（带 1<<(bits-1) 偏移）+ scale [N, K/G]"""
    N, K = w.shape
    G = group_size
    assert K % G == 0, (K, G)
    wg = w.reshape(N, K // G, G)
    amax = np.abs(wg).max(axis=2, keepdims=True)
    qmax = (1 << (bits - 1)) - 1
    sc = np.where(amax > 0, amax / qmax, 1.0).astype(np.float32)
    off = 1 << (bits - 1)
    code = np.clip(np.round(wg / sc).astype(np.int32) + off, 0, (1 << bits) - 1)
    return code.reshape(N, K).astype(np.uint8), sc.reshape(N, K // G)


def pack_to_int32(code_u8, bits):
    """按官方 pack_to_int32 的位流（packed_dim=1）打包成 int32 [N, ceil(K*bits/32)]"""
    N, K = code_u8.shape
    nb = K * bits
    nwords = (nb + 31) // 32
    out = np.zeros((N, nwords), dtype=np.uint32)
    for k in range(K):
        bit = k * bits
        wi = bit // 32
        bo = bit % 32
        v = int(code_u8[0, k]) if False else None  # 占位，下面按行向量化
        break
    # 逐行向量化：先把每行的位流展开到 bit 数组
    kk = np.arange(K)
    bit = kk * bits
    wi = bit // 32
    bo = bit % 32
    for n in range(N):
        row = np.zeros(nwords, dtype=np.uint32)
        vals = code_u8[n].astype(np.uint32)
        np.add.at(row, wi, (vals << bo))
        out[n] = row
    return out.astype(np.int32)


def unpack_from_int32_ct(wp, bits, K):
    """按官方语义解包（用于生成参考）：字节小端，低半字节 = 偶下标"""
    N, _ = wp.shape
    out = np.zeros((N, K), dtype=np.int32)
    if bits == 4:
        bytes_ = wp.astype(np.uint32).view(np.uint8).reshape(N, -1)
        for k in range(K):
            b = k // 2
            out[:, k] = (bytes_[:, b] >> 4) if (k & 1) else (bytes_[:, b] & 0xF)
    else:
        for k in range(K):
            out[:, k] = (wp[:, k // 4] >> (8 * (k % 4))) & 0xFF
    return out - (1 << (bits - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="../models/mistral-rand")
    ap.add_argument("--out", default="../models/mistral-rand-ct4")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--ref-script", default="tests/ref_llama.py")
    ap.add_argument("--ref-out", default="tests/_ref_mistral_ct4.json")
    args = ap.parse_args()

    from safetensors.numpy import load_file

    G, bits = args.group_size, args.bits
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
        base = name[: -len(".weight")]
        if K % G:
            raise SystemExit("%s: K=%d 不是 group=%d 的整数倍" % (name, K, G))
        code, sc = quantize_group_sym(w, bits, G)
        wp = pack_to_int32(code, bits)
        tensors[base + ".weight_packed"] = wp
        tensors[base + ".weight_scale"] = sc.astype(np.float16)
        tensors[base + ".weight_shape"] = np.array([N, K], dtype=np.int64)
        # 参考：按官方解包语义反量化
        dq = (unpack_from_int32_ct(wp, bits, K) * sc.repeat(G, axis=1)).astype(np.float16)
        ref_t[name] = dq
        den = float((w ** 2).sum())
        rel.append(float(np.sqrt(((dq.astype(np.float32) - w) ** 2).sum() / max(den, 1e-12))))
        n_q += 1
        wf = w
        wf = None

    save_safetensors(os.path.join(args.out, "model.safetensors"), tensors)

    c2 = dict(cfg)
    c2["quantization_config"] = {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "quantization_status": "compressed",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {"num_bits": bits, "type": "int", "strategy": "group",
                            "group_size": G, "symmetric": True, "dynamic": False},
                "input_activations": None,
                "output_activations": None,
            }
        },
        "ignore": [],
    }
    json.dump(c2, open(os.path.join(args.out, "config.json"), "w", encoding="utf-8"), indent=1)
    for f in ("tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json"):
        p = os.path.join(args.src, f)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(args.out, f))

    # 参考目录 → ref_llama.py → ref json
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
    print("%-28s %d 个权重 W%dA16(group=%d)，反量化 vs 原始 rel_rms 最大 %.4f"
          % (os.path.basename(args.out), n_q, bits, G, max(rel)))
    print("ref json ->", args.ref_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())