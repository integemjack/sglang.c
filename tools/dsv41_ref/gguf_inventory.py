import re, struct
from gguf import GGUFReader

base = "/Volumes/9100 PRO/models/dsv41/"

def field_val(f):
    if f.types[0] == 8:
        return bytes(f.parts[-1]).decode("utf-8", "replace")
    if len(f.parts) >= 3:
        return f.parts[-1].tolist()
    return f.parts[-1].tolist()

r = GGUFReader(base + "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00006.gguf")
print("== 超参 ==")
for k in sorted(r.fields):
    if k.startswith("deepseek41.") or k in ("split.tensors.count", "general.file_type"):
        print(f"  {k} = {field_val(r.fields[k])}")

lines = []
tot = {}
for i in (2, 3, 4, 5):
    p = base + f"DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-0000{i}-of-00006.gguf"
    rr = GGUFReader(p)
    for t in rr.tensors:
        ty = str(t.tensor_type).split(".")[-1]
        shape = [int(x) for x in t.shape]
        lines.append(f"{t.name}\t{ty}\t{shape}\t{t.n_bytes}")
        tot[ty] = tot.get(ty, 0) + t.n_bytes

print(f"\n== 张量总数 {len(lines)}（含 Engram 文件外）==")
for k, v in sorted(tot.items(), key=lambda x: -x[1]):
    print(f"  {k:12s} {v/1e9:8.2f} GB")

with open(base + "tensors_iq2.txt", "w") as fo:
    fo.write("\n".join(lines))

pats = {}
for ln in lines:
    p = re.sub(r"blocks\.\d+", "blocks.N", ln.split("\t")[0])
    pats.setdefault(p, 0)
    pats[p] += 1
print("\n== 名称模式（去层号后）==")
for p, n in sorted(pats.items()):
    print(f"  {n:4d}x {p}")

# per-layer 每类张量的形状样例
seen = set()
print("\n== 形状样例（每模式首个）==")
for ln in lines:
    p = re.sub(r"blocks\.\d+", "blocks.N", ln.split("\t")[0])
    if p not in seen:
        seen.add(p)
        print(" ", ln)
