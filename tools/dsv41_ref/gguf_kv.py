import struct, mmap, sys

path = "/Volumes/9100 PRO/models/dsv41/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00006.gguf"
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
    n = rd("Q")
    global off
    s = mm[off:off + n].decode("utf-8", "replace")
    off += n
    return s

VT = {0: "u8", 1: "i8", 2: "u16", 3: "i16", 4: "u32", 5: "i32", 6: "f32",
      7: "bool", 8: "str", 9: "arr", 10: "u64", 11: "i64", 12: "f64"}
SZ = {"u8": ("B", 1), "i8": ("b", 1), "u16": ("H", 2), "i16": ("h", 2), "u32": ("I", 4),
      "i32": ("i", 4), "f32": ("f", 4), "bool": ("B", 1), "u64": ("Q", 8), "i64": ("q", 8), "f64": ("d", 8)}

def rval(t):
    t = VT[t]
    if t == "str":
        return rstr()
    if t == "arr":
        et = rd("I")
        n = rd("Q")
        return [rval(et) for _ in range(n)]
    fmt, _ = SZ[t]
    v = rd(fmt)
    return bool(v) if t == "bool" else v

magic = mm[0:4]
off = 4
ver = rd("I")
nten = rd("Q")
nkv = rd("Q")
print(f"magic={magic} ver={ver} tensors={nten} kv={nkv}")

want_prefix = ("deepseek41.attention.compress_ratios", "deepseek41.attention.kv_source_layers",
               "deepseek41.attention.indexer.source_layers", "deepseek41.attention.indexer.candidate_source_layer",
               "deepseek41.ple.", "deepseek41.engram.", "deepseek41.hash_layer_count",
               "deepseek41.expert_gating_func", "general.file_type",
               "deepseek41.attention.indexer.candidate_top_k_blocks")
seen = {}
for _ in range(nkv):
    k = rstr()
    t = rd("I")
    v = rval(t)
    if k.startswith(want_prefix):
        seen[k] = v

for k in sorted(seen):
    v = seen[k]
    if isinstance(v, list) and len(v) > 60:
        print(f"{k} = [len={len(v)}] {v}")
    else:
        print(f"{k} = {v}")
