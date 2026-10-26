"""G12: 把 image.raw（CHW float）写成 PNG，并按「llava 风格」生成带 <image> added token
的 tokenizer.json（复制自 tokenizer 源目录，再追加一个 id=vocab_size 的特殊 token）。

用法: python make_g12_tokenizer.py <tokenizer_src_dir> <out_dir> [--image-raw <image.raw>]

产物（写进 out_dir）：
- tokenizer.json / tokenizer_config.json / vocab.json / merges.txt
- test.png / test.jpg / image.raw（可选）
"""
import json, os, shutil, sys
import numpy as np

src = sys.argv[1]
out = sys.argv[2]
argv = sys.argv[3:]
img_raw = None
if "--image-raw" in argv and argv.index("--image-raw") + 1 < len(argv):
    img_raw = argv[argv.index("--image-raw") + 1]
img_id = -1
if "--image-id" in argv and argv.index("--image-id") + 1 < len(argv):
    img_id = int(argv[argv.index("--image-id") + 1])

os.makedirs(out, exist_ok=True)
for f in ("vocab.json", "merges.txt", "tokenizer_config.json"):
    p = os.path.join(src, f)
    if os.path.exists(p):
        shutil.copy(p, os.path.join(out, f))

tj = json.load(open(os.path.join(src, "tokenizer.json"), encoding="utf-8"))
# 词表大小 = model.vocab 里最大 id + 1（added_tokens 另算）
mx = 0
for toks in tj["model"]["vocab"].values():
    pass
vocab = tj["model"]["vocab"]
mx = max(vocab.values()) if vocab else 0
added = tj.setdefault("added_tokens", [])
have = {t["content"]: t["id"] for t in added}
if "<image>" in have:
    print(f"tokenizer: {out} image_id={have['<image>']} (already present)")
else:
    if img_id < 0:
        # 没有指定就排在 vocab 末尾之后（真实 llava 就是这种：32000 < vocab 32064）
        img_id = max(vocab.values()) + 1
    added.append({"id": img_id, "content": "<image>", "single_word": False, "lstrip": False,
                  "rstrip": False, "normalized": False, "special": True})
    json.dump(tj, open(os.path.join(out, "tokenizer.json"), "w", encoding="utf-8"))
    print(f"tokenizer: {out} image_id={img_id}")

if img_raw:
    from PIL import Image
    img = np.fromfile(img_raw, dtype=np.float32).reshape(3, 32, 32).transpose(1, 2, 0)
    x = np.clip((img * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
    Image.fromarray(x).save(os.path.join(out, "test.png"))
    Image.fromarray(x).save(os.path.join(out, "test.jpg"), quality=92)
    shutil.copy(img_raw, os.path.join(out, "image.raw"))
    print(f"images: {out}/test.png test.jpg image.raw")