"""给 mistral-rand 的分词器加一个 <image> 特殊 token，得到多模态调度测试用的
语言模型目录 llava-lm（llava-rand 只有视觉塔，没有语言模型/分词器）。

用法：python tests/make_llava_lm.py [--src models/mistral-rand] [--out models/llava-lm]
默认路径是本地开发布局（repo 在 <root>/sglang-c、夹具在 <root>/models）；
CI 上模型目录在 checkout 内，用 --src/--out 显式指定（见 .github/workflows/build.yml）。
"""
import argparse
import json
import os
import shutil

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(ROOT, "models", "mistral-rand"))
    ap.add_argument("--out", default=os.path.join(ROOT, "models", "llava-lm"))
    a = ap.parse_args()
    SRC, DST = a.src, a.out
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