"""给随机 DeepSeek 测试模型补一个最小可用的 tokenizer.json（GPT-2 byte-level 格式）。

随机小模型的词表只有几十个 id，装不下完整的 GPT-2 词表；这里按 vocab 大小
挑前 N 个「可打印、常用」的字节 token 写进去，让 sglc 的 BPE 词表与模型 logits
维度对齐，从而能跑通端到端的 HTTP 演示。

用法：python tests/gen_ds_tokenizer.py --model ../models/deepseek-rand
"""
import argparse
import json
import os

# GPT-2 bytes_to_unicode 的可打印字符顺序（前若干项），够小模型用
BYTES = (
    list(range(ord("!"), ord("~") + 1))
    + list(range(ord("¡"), ord("¬") + 1))
    + list(range(ord("®"), ord("ÿ") + 1))
)


def bytes_to_unicode():
    cs = BYTES[:]
    n = 0
    for b in range(256):
        if b not in cs:
            cs.append(b)
            n += 1
            if n >= 68:
                break
    return {b: chr(c) for b, c in zip(range(256), cs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    # 优先装「能拼出 JSON」的字符（结构化输出演示用），不够再用字母补
    ap.add_argument("--chars", default=" {}\"[,]:0123456789abcdefgh")
    a = ap.parse_args()

    cfg_path = os.path.join(a.model, "config.json")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    V = int(cfg.get("vocab_size") or 24)

    b2u = bytes_to_unicode()
    chars = a.chars[:V]
    vocab = {}
    for i, ch in enumerate(chars):
        vocab[ch if ch != " " else b2u[0x20]] = i
    i = len(vocab)
    for b in range(256):  # 补齐到 V（多字节字符也能回退到单字节）
        if i >= V:
            break
        tok = b2u[b]
        if tok not in vocab:
            vocab[tok] = i
            i += 1
    chars = chars.ljust(V, "a")

    tok = {
        "version": "1.0",
        "model": {"type": "BPE", "vocab": vocab, "merges": []},
        "added_tokens": [],
        "pre_tokenizer": None,
        "decoder": None,
    }
    with open(os.path.join(a.model, "tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump(tok, f, ensure_ascii=False)
    with open(os.path.join(a.model, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump({"tokenizer_class": "GPT2Tokenizer"}, f)
    print("wrote tokenizer.json to %s (vocab=%d)" % (a.model, len(vocab)))

if __name__ == "__main__":
    main()