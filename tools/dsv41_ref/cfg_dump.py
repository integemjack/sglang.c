"""从 GGUF shard1 dump 全部元数据 -> config.json（引擎加载用）"""
import json, struct, sys, mmap

path = sys.argv[1] if len(sys.argv) > 1 else \
    "/Volumes/9100 PRO/models/dsv41/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00006.gguf"
out = sys.argv[2] if len(sys.argv) > 2 else "/Volumes/9100 PRO/models/dsv41/mlx-dense/config.json"

f = open(path, "rb")
mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
off = 0

def rd(fmt):
    global off
    n = struct.calcsize(fmt)
    v = struct.unpack_from("<" + fmt, mm, off)
    off += n
    return v[0] if len(v) == 1 else v

def rstr():
    global off
    n = rd("Q")
    s = mm[off:off + n].decode("utf-8", "replace")
    off += n
    return s

SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

def rval(t):
    global off
    if t == 8:
        return rstr()
    if t == 9:
        et = rd("I")
        n = rd("Q")
        if et == 8:
            return [rstr() for _ in range(n)]
        if et == 10:  # u64 数组单独处理（大整数）
            base = off
            off += n * 8
            return {"__type": "u64", "n": n, "off": base, "raw": list(struct.unpack_from(f"<{n}Q", mm, base))}
        fmt = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "B", 10: "Q", 11: "q", 12: "d"}[et]
        base = off
        off += n * SZ[et]
        return list(struct.unpack_from(f"<{n}{fmt}", mm, base))
    fmt = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "B", 10: "Q", 11: "q", 12: "d"}[t]
    v = rd(fmt)
    return bool(v) if t == 7 else v

assert mm[0:4] == b"GGUF"
off = 4
rd("I")
n_ten = rd("Q")
n_kv = rd("Q")
cfg = {}
for _ in range(n_kv):
    k = rstr()
    t = rd("I")
    v = rval(t)
    if isinstance(v, dict):  # u64 数组
        cfg[k] = v["raw"]
    else:
        cfg[k] = v

cfg["_files"] = {
    "dense_safetensors": "model.safetensors",
    "experts_gguf": ["DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-%05d-of-00006.gguf", 5],
    "engram_gguf": "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf",
}
with open(out, "w") as fo:
    json.dump(cfg, fo, ensure_ascii=False, indent=1)
print(f"config.json：{len(cfg)} 键")
for k in sorted(cfg):
    v = cfg[k]
    if isinstance(v, list) and len(v) > 20:
        print(f"  {k} = [len {len(v)}]")
    else:
        print(f"  {k} = {v}")
