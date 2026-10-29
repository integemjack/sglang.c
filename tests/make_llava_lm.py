"""给 mistral-rand 的分词器加一个 <image> 特殊 token，得到多模态调度测试用的
语言模型目录 models/llava-lm（llava-rand 只有视觉塔，没有语言模型/分词器）。

用法：python tests/make_llava_lm.py
"""
import json
import os
import shutil

SRC = os.path.join(os.path.dirname(__file__), "..", "..", "models", "mistral-rand")
DST = os.path.join(os.path.dirname(__file__), "..", "..", "models", "llava-lm")


def main():
    if not os.path.isdir(SRC):
        print("skip: %s missing" % SRC)
        return 0
    if os.path.isdir(DST):
        shutil.rmtree(DST)
    os.makedirs(DST)
    for name in ("model.safetensors", "config.json", "tokenizer_config.json",
                 "merges.txt", "vocab.json"):
        p = os.path.join(SRC, name)
        if os.path.isfile(p):
            shutil.copy2(p, os.path.join(DST, name))

    tok = json.load(open(os.path.join(SRC, "tokenizer.json"), encoding="utf-8"))
    added = tok.setdefault("added_tokens", [])
    if not any(a.get("content") == "<image>" for a in added):
        used = {a.get("id") for a in added}
        nid = max(used | {0}) + 1
        added.append({"id": nid, "content": "<image>", "single_word": False,
                      "lstrip": False, "rstrip": False, "normalized": False,
                      "special": True})
    json.dump(tok, open(os.path.join(DST, "tokenizer.json"), "w", encoding="utf-8"))
    print("wrote %s（分词器加了 <image>，id=%d）" % (DST, added[-1]["id"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())