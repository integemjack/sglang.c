import json, glob

cands = sorted(glob.glob("/Users/wangchao/Library/Logs/DiagnosticReports/mlx_ds41_model*"), reverse=True)
f = cands[0]
print("file:", f.split("/")[-1])
txt = open(f).read()
p = txt.index("{", txt.index("}") + 1)
j = json.loads(txt[p:])
t = j["threads"][j.get("faultingThread", 0)]
imgs = j["usedImages"]
for fr in t["frames"][:14]:
    print(imgs[fr["imageIndex"]].get("name", "?"), str(fr.get("symbol", "?"))[:90])
