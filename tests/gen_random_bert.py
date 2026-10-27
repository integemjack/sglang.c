#!/usr/bin/env python3
"""tests/gen_random_bert.py — 生成「随机权重的小 BERT / RoBERTa + WordPiece」模型目录，
并用 numpy 独立实现 BERT 前向（与 C 端零共享代码）导出参考值。

用法:
    python tests/gen_random_bert.py [out_dir] [--seed N]

产物（out_dir = argv[1]，默认 $TEMP/sgl_bert_gen；roberta 变体在 out_dir+"_roberta"）:
    config.json  model.safetensors  tokenizer.json  tokenizer_config.json
    ref_bert_embed.json      # {"model":…, "texts":[{text,ids,embed_mean,embed_cls,l2_mean}],
                             #  "rerank":[{query,doc,ids,score}], "rank":[…]}

模型规模：hidden=32, layers=2, heads=4, intermediate=64, vocab=96, max_len=64。
tokenizer.json 手写（WordPiece + BertNormalizer + BertPreTokenizer + TemplateProcessing）：
若本机装了 HF `tokenizers`，参考 token id 取它的输出（权威实现）；同时用本文件里
自己写的 WordPiece 再算一遍，两者不一致会打印 WARN，做到「参考实现与 C 零共享」。
"""
import json
import math
import os
import struct
import sys
import tempfile

import numpy as np

# ---------------- 规模（与 tests/test_bert_embed.c 的断言一致） ----------------
E, L, NH, I, V, MAXLEN = 32, 2, 4, 64, 96, 64
D = E // NH
EPS = 1e-12


def out_dirs(argv):
    seed = 42
    pos = []
    i = 0
    while i < len(argv):
        if argv[i] == "--seed" and i + 1 < len(argv):
            seed = int(argv[i + 1]); i += 2; continue
        pos.append(argv[i]); i += 1
    if pos:
        base = pos[0]
    else:
        tmp = os.environ.get("TEMP") or os.environ.get("TMP") or "."
        base = os.path.join(tmp, "sgl_bert_gen")
    return base, base + "_roberta", seed


# ---------------- 词表（WordPiece） ----------------
def build_vocab():
    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "cat",
             "hello", "world", "good", "bad", "query", "document", "test",
             "embedding", "model", "run", "runs", "running", "search", "result",
             "score", "relevant", "pizza", "alpha", "beta", "gamma", "delta",
             "machine", "learning", "neural", "corpus"]
    letters = [chr(ord("a") + k) for k in range(26)]
    subs = ["##s", "##t", "##e", "##d", "##n", "##g", "##ing", "##ed", "##er",
            "##ion", "##ly", "##a", "##o", "##l", "##r", "##m", "##y", "##i",
            "##c", "##k", "##u", "##p", "##h", "##b", "##f", "##w", "##v",
            "##x", "##z", "##j", "##q"]
    vocab = specials + words + letters + subs
    assert len(vocab) == V, f"vocab {len(vocab)} != {V}"
    assert len(set(vocab)) == V, "duplicate token in vocab"
    return {t: i for i, t in enumerate(vocab)}


VOCAB = build_vocab()
ID2TOK = {i: t for t, i in VOCAB.items()}
PAD, UNK, CLS, SEP, MASK = 0, 1, 2, 3, 4
CONT = "##"
MAX_CHARS = 100


def write_tokenizer_json(path):
    """手写 HF tokenizer.json（bert-base-uncased 的 normalizer/pretokenizer 组合）。"""
    added = [{"id": VOCAB[t], "content": t, "single_word": False, "lstrip": False,
              "rstrip": False, "normalized": False, "special": True}
             for t in ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")]
    tok = {
        "version": "1.0",
        "truncation": {"direction": "Right", "max_length": MAXLEN,
                       "strategy": "LongestFirst", "stride": 0},
        "padding": None,
        "added_tokens": added,
        "normalizer": {"type": "BertNormalizer", "clean_text": True,
                       "handle_chinese_chars": True, "strip_accents": None,
                       "lowercase": True},
        "pre_tokenizer": {"type": "BertPreTokenizer"},
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [{"SpecialToken": {"id": "[CLS]", "type_id": 0}},
                       {"Sequence": {"id": "A", "type_id": 0}},
                       {"SpecialToken": {"id": "[SEP]", "type_id": 0}}],
            "pair": [{"SpecialToken": {"id": "[CLS]", "type_id": 0}},
                     {"Sequence": {"id": "A", "type_id": 0}},
                     {"SpecialToken": {"id": "[SEP]", "type_id": 0}},
                     {"Sequence": {"id": "B", "type_id": 0}},
                     {"SpecialToken": {"id": "[SEP]", "type_id": 0}}],
            "special_tokens": {
                "[CLS]": {"id": "[CLS]", "ids": [CLS], "tokens": ["[CLS]"]},
                "[SEP]": {"id": "[SEP]", "ids": [SEP], "tokens": ["[SEP]"]},
            },
        },
        "decoder": None,
        "model": {"type": "WordPiece", "unk_token": "[UNK]",
                  "continuing_subword_prefix": CONT,
                  "max_input_chars_per_word": MAX_CHARS,
                  "vocab": dict(sorted(VOCAB.items(), key=lambda kv: kv[1]))},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tok, f, ensure_ascii=False, indent=1)
    cfg = {"do_lower_case": True, "tokenizer_class": "BertTokenizer",
           "unk_token": "[UNK]", "sep_token": "[SEP]", "pad_token": "[PAD]",
           "cls_token": "[CLS]", "mask_token": "[MASK]", "model_max_length": MAXLEN}
    with open(os.path.join(os.path.dirname(path), "tokenizer_config.json"), "w",
              encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=1)


# ---------------- 参考 WordPiece（本文件自己的实现，与 C 零共享） ----------------
def _is_ws(cp):
    return cp in (0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20, 0x85, 0xA0, 0x1680,
                  0x2028, 0x2029, 0x202F, 0x205F, 0x3000) or (0x2000 <= cp <= 0x200A)


def _is_control(cp):
    if cp in (0x09, 0x0A, 0x0D):
        return False
    return (0 <= cp <= 0x1F) or (0x7F <= cp <= 0x9F)


def _is_cjk(cp):
    return ((0x4E00 <= cp <= 0x9FFF) or (0x3400 <= cp <= 0x4DBF) or
            (0x20000 <= cp <= 0x2A6DF) or (0x2A700 <= cp <= 0x2B73F) or
            (0x2B740 <= cp <= 0x2B81F) or (0x2B820 <= cp <= 0x2CEAF) or
            (0xF900 <= cp <= 0xFAFF) or (0x2F800 <= cp <= 0x2FA1F) or
            (0x3000 <= cp <= 0x303F) or (0x3040 <= cp <= 0x30FF) or
            (0x31F0 <= cp <= 0x31FF) or (0xFF00 <= cp <= 0xFFEF) or
            (0xAC00 <= cp <= 0xD7AF))


def _is_punct(cp):
    if (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126):
        return True
    if cp < 0x80:
        return False
    ranges = [(0x2000, 0x206F), (0x2E00, 0x2E7F), (0x3001, 0x303F),
              (0xFE10, 0xFE19), (0xFE30, 0xFE6F), (0xFF01, 0xFF0F),
              (0xFF1A, 0xFF20), (0xFF3B, 0xFF40), (0xFF5B, 0xFF65)]
    return any(a <= cp <= b for a, b in ranges)


def py_words(text):
    """BertNormalizer(clean/lowercase) + BasicTokenizer 的空白/标点/CJK 切分。"""
    words, cur = [], []
    for ch in text:
        cp = ord(ch)
        if cp == 0 or cp == 0xFFFD or _is_control(cp):
            continue
        if _is_ws(cp):
            if cur:
                words.append("".join(cur)); cur = []
            continue
        ch = ch.lower()
        if _is_cjk(cp) or _is_punct(cp):
            if cur:
                words.append("".join(cur)); cur = []
            words.append(ch)
            continue
        cur.append(ch)
    if cur:
        words.append("".join(cur))
    return words


def py_wordpiece_ids(text):
    ids = [CLS]
    for w in py_words(text):
        if len(w) > MAX_CHARS:
            ids.append(UNK); continue
        pieces, start, ok = [], 0, True
        while start < len(w):
            end, found = len(w), None
            while start < end:
                sub = w[start:end]
                cand = sub if start == 0 else CONT + sub
                if cand in VOCAB:
                    found = cand; break
                end -= 1
            if found is None:
                ok = False; break
            pieces.append(found); start = end
        ids.extend(VOCAB[p] for p in pieces) if ok else ids.append(UNK)
        if len(ids) >= MAXLEN - 1:
            break
    ids = ids[:MAXLEN - 1]
    ids.append(SEP)
    return ids


# ---------------- numpy 参考前向 ----------------
def ln(x, w, b):
    m = x.mean(axis=-1, keepdims=True)
    v = x.var(axis=-1, keepdims=True)
    return (x - m) / np.sqrt(v + EPS) * w + b


def gelu(x):
    return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def forward(P, ids, pos_offset):
    n = len(ids)
    pos = np.arange(n) + pos_offset
    tt = np.zeros(n, dtype=np.int64)
    x = P["wte"][ids] + P["wpe"][pos] + P["wtt"][tt]
    x = ln(x, P["emb.ln.w"], P["emb.ln.b"])
    for l in range(L):
        q = x @ P[f"{l}.q.w"].T + P[f"{l}.q.b"]
        k = x @ P[f"{l}.k.w"].T + P[f"{l}.k.b"]
        v = x @ P[f"{l}.v.w"].T + P[f"{l}.v.b"]
        qh = q.reshape(n, NH, D).transpose(1, 0, 2)
        kh = k.reshape(n, NH, D).transpose(1, 0, 2)
        vh = v.reshape(n, NH, D).transpose(1, 0, 2)
        a = softmax(qh @ kh.transpose(0, 2, 1) / math.sqrt(D))
        c = (a @ vh).transpose(1, 0, 2).reshape(n, E)
        x = ln(x + (c @ P[f"{l}.o.w"].T + P[f"{l}.o.b"]),
               P[f"{l}.ln2.w"], P[f"{l}.ln2.b"])
        h = gelu(x @ P[f"{l}.i.w"].T + P[f"{l}.i.b"])
        x = ln(x + (h @ P[f"{l}.o2.w"].T + P[f"{l}.o2.b"]),
               P[f"{l}.ln3.w"], P[f"{l}.ln3.b"])
    return x


def mean_pool_norm(h):
    m = h.mean(axis=0)
    return m / max(float(np.linalg.norm(m)), 1e-12)


def score(P, ids, pos_offset, n_labels=1):
    h = forward(P, ids, pos_offset)
    if n_labels == 1:
        return float(h[0] @ P["cls.w"][0] + P["cls.b"][0])
    logits = P["cls.w"] @ h[0] + P["cls.b"]
    return float(logits[1])


# ---------------- 权重 ----------------
def make_weights(rng, prefix, pos_rows):
    P = {}
    W = {}

    def add(hf_name, arr, ref_key):
        W[hf_name] = arr.astype(np.float32)
        P[ref_key] = arr.astype(np.float64)

    def nrm(name, ref, shape):
        add(f"{prefix}{name}.weight", 1.0 + 0.1 * rng.standard_normal(shape), ref + ".w")
        add(f"{prefix}{name}.bias", 0.1 * rng.standard_normal(shape), ref + ".b")

    add(f"{prefix}embeddings.word_embeddings.weight",
        rng.normal(0, 0.1, (V, E)), "wte")
    add(f"{prefix}embeddings.position_embeddings.weight",
        rng.normal(0, 0.1, (pos_rows, E)), "wpe")
    add(f"{prefix}embeddings.token_type_embeddings.weight",
        rng.normal(0, 0.1, (2, E)), "wtt")
    nrm("embeddings.LayerNorm", "emb.ln", (E,))
    for l in range(L):
        p = f"encoder.layer.{l}"
        for proj, key in (("query", "q"), ("key", "k"), ("value", "v")):
            add(f"{prefix}{p}.attention.self.{proj}.weight",
                rng.normal(0, 0.1, (E, E)), f"{l}.{key}.w")
            add(f"{prefix}{p}.attention.self.{proj}.bias",
                rng.normal(0, 0.1, (E,)), f"{l}.{key}.b")
        add(f"{prefix}{p}.attention.output.dense.weight",
            rng.normal(0, 0.1, (E, E)), f"{l}.o.w")
        add(f"{prefix}{p}.attention.output.dense.bias",
            rng.normal(0, 0.1, (E,)), f"{l}.o.b")
        nrm(f"{p}.attention.output.LayerNorm", f"{l}.ln2", (E,))
        add(f"{prefix}{p}.intermediate.dense.weight",
            rng.normal(0, 0.1, (I, E)), f"{l}.i.w")
        add(f"{prefix}{p}.intermediate.dense.bias",
            rng.normal(0, 0.1, (I,)), f"{l}.i.b")
        add(f"{prefix}{p}.output.dense.weight",
            rng.normal(0, 0.1, (E, I)), f"{l}.o2.w")
        add(f"{prefix}{p}.output.dense.bias",
            rng.normal(0, 0.1, (E,)), f"{l}.o2.b")
        nrm(f"{p}.output.LayerNorm", f"{l}.ln3", (E,))
    add(f"{prefix}pooler.dense.weight", rng.normal(0, 0.1, (E, E)), "pool.w")
    add(f"{prefix}pooler.dense.bias", rng.normal(0, 0.1, (E,)), "pool.b")
    add("classifier.weight", rng.normal(0, 0.1, (1, E)), "cls.w")
    add("classifier.bias", rng.normal(0, 0.1, (1,)), "cls.b")
    return P, W


def save_safetensors(path, tensors):
    hdr, off = {}, 0
    for k, v in tensors.items():
        nbytes = v.nbytes
        hdr[k] = {"dtype": "F32", "shape": list(v.shape),
                  "data_offsets": [off, off + nbytes]}
        off += nbytes
    hb = json.dumps(hdr).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for v in tensors.values():
            f.write(v.astype(np.float32).tobytes())


TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "hello world",
    "machine learning model, embedding test!",
    "a query about corpus scoring",
    "the quick brown fox jumps over the lazy dog " * 8,   # 触发 max_len=64 截断
]
RERANK = [
    ("query about the quick fox", "the quick brown fox jumps over the lazy dog"),
    ("query about the quick fox", "hello world"),
    ("machine learning embedding", "machine learning model embedding test"),
    ("pizza", "the cat runs over the lazy dog"),
]


def hf_ids(text, hf_tok):
    if hf_tok is None:
        return None
    try:
        return list(hf_tok.encode(text).ids)
    except Exception as e:  # pragma: no cover
        print(f"WARN: HF tokenizers encode failed: {e}")
        return None


def gen(out_dir, seed, model_type):
    rng = np.random.default_rng(seed)
    roberta = model_type == "roberta"
    os.makedirs(out_dir, exist_ok=True)
    pos_offset = 2 if roberta else 0
    pad_id = 1 if roberta else 0
    cfg = {
        "architectures": ["BertModel" if not roberta else "RobertaModel"],
        "model_type": model_type,
        "hidden_size": E, "intermediate_size": I, "num_hidden_layers": L,
        "num_attention_heads": NH, "hidden_act": "gelu", "vocab_size": V,
        "max_position_embeddings": MAXLEN + pos_offset, "type_vocab_size": 2,
        "layer_norm_eps": EPS, "pad_token_id": pad_id,
        "position_embedding_type": "absolute", "initializer_range": 0.1,
    }
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1)

    P, W = make_weights(rng, "roberta." if roberta else "bert.", MAXLEN + pos_offset)
    save_safetensors(os.path.join(out_dir, "model.safetensors"), W)
    tok_path = os.path.join(out_dir, "tokenizer.json")
    write_tokenizer_json(tok_path)

    hf_tok = None
    try:
        from tokenizers import Tokenizer
        hf_tok = Tokenizer.from_file(tok_path)
    except Exception as e:
        print(f"WARN: HF tokenizers 不可用（{e}），参考 ids 用本文件自实现 WordPiece")

    ref = {"model": {"hidden": E, "layers": L, "heads": NH, "vocab": V,
                     "max_len": MAXLEN, "model_type": model_type,
                     "pos_offset": pos_offset},
           "texts": [], "rerank": []}
    mism = 0
    for text in TEXTS:
        ids = hf_ids(text, hf_tok)
        mine = py_wordpiece_ids(text)
        if ids is None:
            ids = mine
        elif ids != mine:
            mism += 1
            print(f"WARN: 参考 ids 不一致 HF={ids[:12]}… mine={mine[:12]}…  text={text[:40]!r}")
        h = forward(P, np.array(ids), pos_offset)
        emb = mean_pool_norm(h)
        clash = [int(i) for i in ids if i >= V]
        assert not clash, f"ids out of vocab: {clash}"
        ref["texts"].append({
            "text": text, "ids": ids,
            "embed_mean": [float(v) for v in emb],
            "embed_cls": [float(v) for v in (h[0] / max(float(np.linalg.norm(h[0])), 1e-12))],
            "l2_mean": float(np.linalg.norm(emb)),
        })
    scores = []
    for q, d in RERANK:
        qids = (hf_ids(q, hf_tok) or py_wordpiece_ids(q))[1:-1]
        dids = (hf_ids(d, hf_tok) or py_wordpiece_ids(d))[1:-1]
        # 与 src/bert.c 的 sgl_bert_pair_ids 完全一致：
        #   avail = max_len - 3 - (roberta 多一个 SEP)；query 占一半、doc 拿剩下
        extra_sep = 1 if roberta else 0
        avail = MAXLEN - 3 - extra_sep
        qn = min(len(qids), avail // 2 + avail % 2)
        dn = min(len(dids), avail - qn)
        ids = [CLS] + qids[:qn] + [SEP] + ([SEP] if extra_sep else []) + dids[:dn] + [SEP]
        sc = score(P, np.array(ids), pos_offset)
        scores.append(sc)
        ref["rerank"].append({"query": q, "doc": d, "ids": ids, "score": sc})
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    ref["rank"] = order
    with open(os.path.join(out_dir, "ref_bert_embed.json"), "w", encoding="utf-8") as f:
        json.dump(ref, f)
    print(f"saved random {model_type} -> {out_dir} "
          f"(E={E} L={L} NH={NH} I={I} V={V} max_len={MAXLEN} pos_offset={pos_offset}) "
          f"ids_source={'hf-tokenizers' if hf_tok else 'py-wordpiece'} "
          f"py_hf_mismatch={mism} scores={[round(s, 6) for s in scores]}")


def main():
    base, base_rb, seed = out_dirs(sys.argv[1:])
    gen(base, seed, "bert")
    gen(base_rb, seed + 1, "roberta")


if __name__ == "__main__":
    main()