"""从已有的 GPTQ(v1) 夹具派生一个 gptq_v2 夹具，并复用 v1 的 logits 参考。

GPTQ v2 的全部差别：落盘的 qzeros 存的是「v1 的零点 - 1」，解量化时要 +1
（vLLM srt/layers/quantization/gptq.py 的 v2→v1 转换就是 zeros + 1）。
所以：把每个 qzeros 的 4bit 槽位整体减 1（按 4bit 回绕），配上 format=gptq_v2，
加载出来的权重必须和 v1 目录**完全一致**，参考 logits 直接沿用 v1 的 ref json。

用法：python tests/make_gptq_v2.py [v1 目录] [输出目录]
"""
import json
import os
import shutil
import struct
import sys

import numpy as np

DT = {"F32": np.float32, "F16": np.float16, "I32": np.int32, "I64": np.int64,
      "U8": np.uint8, "I8": np.int8, "BF16": np.uint16}


def load_safetensors(path):
    with open(path, "rb") as f:
        raw = f.read()
    n = struct.unpack("<Q", raw[:8])[0]
    meta = json.loads(raw[8:8 + n].decode("utf-8"))
    base = 8 + n
    out = {}
    for name, info in meta.items():
        if name == "__metadata__":
            continue
        b, e = info["data_offsets"]
        dt = DT[info["dtype"]]
        out[name] = np.frombuffer(raw, dtype=dt, count=(e - b) // np.dtype(dt).itemsize,
                                  offset=base + b).reshape(info["shape"]).copy()
    return out, meta.get("__metadata__")


def save_safetensors(path, tensors, metadata=None):
    header = {}
    blobs = []
    off = 0
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr)
        data = arr.tobytes()
        header[name] = {"dtype": {np.dtype(np.float32): "F32", np.dtype(np.float16): "F16",
                                  np.dtype(np.int32): "I32", np.dtype(np.int64): "I64",
                                  np.dtype(np.uint8): "U8", np.dtype(np.int8): "I8"}[arr.dtype],
                        "shape": list(arr.shape), "data_offsets": [off, off + len(data)]}
        off += len(data)
        blobs.append(data)
    if metadata:
        header["__metadata__"] = metadata
    hb = json.dumps(header, separators=(",", ":")).encode("utf-8")
    hb += b" " * ((8 - len(hb) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for d in blobs:
            f.write(d)


def sub_one_per_4bit(word):
    """int32 里 8 个 4bit 槽位各减 1（按 16 回绕）"""
    w = word.astype(np.int64) & 0xFFFFFFFF
    r = np.zeros_like(w)
    for s in range(8):
        v = (w >> (4 * s)) & 0xF
        v = (v - 1) & 0xF
        r |= v << (4 * s)
    return (r & 0xFFFFFFFF).astype(np.uint32).view(np.int32)


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "../models/mistral-rand-gptq"
    dst = sys.argv[2] if len(sys.argv) > 2 else "../models/mistral-rand-gptq-v2"
    if not os.path.isdir(dst):
        os.makedirs(dst)
    for f in ("tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json"):
        p = os.path.join(src, f)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(dst, f))

    cfg = json.load(open(os.path.join(src, "config.json"), encoding="utf-8"))
    qc = cfg.get("quantization_config", {})
    qc["format"] = "gptq_v2"
    cfg["quantization_config"] = qc
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w", encoding="utf-8"), indent=1)

    tensors, md = load_safetensors(os.path.join(src, "model.safetensors"))
    nq = 0
    for name in list(tensors.keys()):
        if not name.endswith(".qzeros"):
            continue
        tensors[name] = sub_one_per_4bit(tensors[name])
        nq += 1
    save_safetensors(os.path.join(dst, "model.safetensors"), tensors, md)
    print("wrote %s: %d 个 qzeros 整体 -1（4bit 回绕），format=gptq_v2" % (dst, nq))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())