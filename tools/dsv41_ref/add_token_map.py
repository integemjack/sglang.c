"""从 GGUF 分片 1 的 KV 抽 deepseek41.engram.token_map，补进 mlx-dense/model.safetensors。"""
import os
import struct
import sys

import mlx.core as mx
import numpy as np

GGUF_DIR = sys.argv[1] if len(sys.argv) > 1 else "./models/dsv41"
path = os.path.join(
    GGUF_DIR, "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00006.gguf"
)

VTS = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
fh = open(path, "rb")
assert fh.read(4) == b"GGUF"
struct.unpack("<I", fh.read(4))
n_ten, n_kv = struct.unpack("<QQ", fh.read(16))


def rstr():
    n = struct.unpack("<Q", fh.read(8))[0]
    return fh.read(n)


token_map = None
for _ in range(n_kv):
    key = rstr().decode()
    t = struct.unpack("<I", fh.read(4))[0]
    if t == 8:
        rstr()
        continue
    if t == 9:
        et = struct.unpack("<I", fh.read(4))[0]
        n = struct.unpack("<Q", fh.read(8))[0]
        if key == "deepseek41.engram.token_map":
            fmt = {2: "<u2", 3: "<i2", 4: "<u4", 5: "<i4", 6: "<f4", 10: "<u8", 11: "<i8"}[et]
            token_map = np.frombuffer(fh.read(n * VTS[et]), dtype=fmt).astype(np.uint32)
            print("token_map:", token_map.shape, token_map[:8].tolist())
            break
        if et == 8:
            for _ in range(n):
                rstr()
        else:
            fh.seek(n * VTS[et], 1)
        continue
    fh.seek(VTS[t], 1)

assert token_map is not None, "token_map kv not found"

p = os.path.join(GGUF_DIR, "mlx-dense", "model.safetensors")
W = dict(mx.load(p))
print("before:", len(W))
W["engram_token_map"] = mx.array(token_map)
mx.save_safetensors(p + ".tmp.safetensors", W)
os.replace(p + ".tmp.safetensors", p)
print("after:", len(mx.load(p, return_metadata=False).keys() if hasattr(mx.load(p), "keys") else 0))
W2 = mx.load(p)
print("after:", len(W2), "has token_map:", "engram_token_map" in W2)
