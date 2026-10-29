"""生成 Unigram tokenizer 的独立参考（tests/_ref_unigram.json）。

参考实现只共用「词表文件」这一份数据，切分算法在 Python 里独立写：
  - 归一化与 C 侧一致：段首补 "▁"，空格→"▁"（T5 的 Precompiled NFKC 不模拟，
    非 ASCII 归一化差异属于已知差异，见 docs/GAP_vs_upstream.md）
  - Viterbi 最大 logprob 切分，找不到 piece 的单字节回退 "<0xXX>"

用法：python tests/gen_unigram_ref.py [tokenizer_dir] [out_json]
"""
import json
import os
import sys

DEFAULT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models", "unigram-t5")

TEXTS = [
    "Hello world",
    "The capital of France is Paris.",
    "translate English to German: How are you?",
    "  leading and   multiple   spaces  ",
    "Mixed 中文 and English 混排",
    "emoji 😀👍 test",
    "def f(x):\n    return x + 1\n",
]


def build(tok_dir):
    path = os.path.join(tok_dir, "tokenizer.json")
    j = json.load(open(path, encoding="utf-8"))
    v = j["model"]["vocab"]
    piece_to_id = {}
    scores = {}
    for i, (p, s) in enumerate(v):
        piece_to_id[p] = i
        scores[p] = float(s)
    unk = piece_to_id.get("<unk>", 0)
    maxlen = max(len(p.encode("utf-8")) for p in piece_to_id)
    return piece_to_id, scores, unk, maxlen, v


def encode(text, piece_to_id, scores, unk, maxlen):
    buf = "\u2581" + text.replace(" ", "\u2581")
    b = buf.encode("utf-8")
    n = len(b)
    NEG = -1e30
    dp = [NEG] * (n + 1)
    prev = [-1] * (n + 1)
    pid = [0] * (n + 1)
    dp[0] = 0.0
    for i in range(n):
        if dp[i] == NEG:
            continue
        # i 必须落在字符边界：否则 b[i:j] 会把前面的残字节丢掉，
        # 被 decode(…, "ignore") 当成合法 piece（这是参考实现自己踩过的坑）
        if i > 0 and (b[i] & 0xC0) == 0x80:
            piece = "<0x%02X>" % b[i]
            iid = piece_to_id.get(piece, unk)
            sc = dp[i] - 20.0
            if sc > dp[i + 1]:
                dp[i + 1] = sc
                prev[i + 1] = i
                pid[i + 1] = iid
            continue
        any_match = False
        for l in range(1, maxlen + 1):
            jx = i + l
            if jx > n:
                break
            if jx < n and (b[jx] & 0xC0) == 0x80:
                continue
            piece = b[i:jx].decode("utf-8", "ignore")
            if piece not in piece_to_id:
                continue
            any_match = True
            sc = dp[i] + scores[piece]
            if sc > dp[jx]:
                dp[jx] = sc
                prev[jx] = i
                pid[jx] = piece_to_id[piece]
        if not any_match:
            piece = "<0x%02X>" % b[i]
            iid = piece_to_id.get(piece, unk)
            sc = dp[i] - 20.0
            if sc > dp[i + 1]:
                dp[i + 1] = sc
                prev[i + 1] = i
                pid[i + 1] = iid
    out = []
    p = n
    while p > 0:
        out.append(pid[p])
        p = prev[p]
    return list(reversed(out))


def decode(ids, vocab):
    s = ""
    for i in ids:
        p = vocab[i][0]
        if len(p) == 6 and p.startswith("<0x") and p.endswith(">"):
            s += chr(int(p[3:5], 16))
        else:
            s += p
    return s.replace("\u2581", " ")


def main():
    tok_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(__file__), "_ref_unigram.json")
    piece_to_id, scores, unk, maxlen, vocab = build(tok_dir)
    cases = []
    for t in TEXTS:
        ids = encode(t, piece_to_id, scores, unk, maxlen)
        cases.append({"text": t, "ids": ids, "decoded": decode(ids, vocab)})
    json.dump({"vocab_size": len(vocab), "cases": cases}, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("wrote %s (%d cases, vocab %d)" % (out, len(cases), len(vocab)))


if __name__ == "__main__":
    main()