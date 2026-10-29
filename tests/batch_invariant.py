"""batch-invariant 检查：同一请求在不同 batch 组成下必须逐字节一致。

上游 sglang 有 `batch_invariant_ops`（同请求不同 batch 组成下 bit 级可复现），
本仓没有显式实现，但需要证明「结果本来就不依赖 batch 组成」这件事成立。
做法（不依赖内部实现，端到端）：

  1) 单独发 prompt P → 文本 A1
  2) 同时发 P + 另外 3 个不同 prompt（并发，强制进同一 batch）
     → 拿到 P 的文本 A2
  3) 再串行发一遍 P → A3
  4) A1 == A2 == A3（逐字节）才算通过

再加两组：不同 max_tokens（1 / 8 / 32）和不同 temperature=0，
确认长度、批组成都不影响同一个 token 的内容。

用法：python tests/batch_invariant.py [port]
"""
import json
import sys
import threading
import urllib.request

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 30121
BASE = "http://127.0.0.1:%d" % PORT
fails = 0


def gen(prompt, max_tokens=16, extra=None, timeout=300):
    body = {"model": "x", "prompt": prompt, "max_tokens": max_tokens,
            "temperature": 0, "top_p": 1.0}
    if extra:
        body.update(extra)
    req = urllib.request.Request(BASE + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        j = json.loads(r.read().decode("utf-8", "replace"))
    return j["choices"][0]["text"]


def check(name, ok, detail=""):
    global fails
    if not ok:
        fails += 1
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else "  <- " + str(detail)[:300]))


def main():
    P = "The quick brown fox jumps over the lazy dog"
    others = ["Hello world, this is a test", "def f(x):\n    return x + 1\n",
              "Once upon a time there was a small cat"]

    a1 = gen(P)
    # 并发：P 与 3 个不同 prompt 同时提交，强制同批
    results = {}
    lock = threading.Lock()

    def worker(tag, text, max_tokens=16):
        v = gen(text, max_tokens=max_tokens, extra={"n": 1})
        with lock:
            results[tag] = v

    ths = [threading.Thread(target=worker, args=("P", P))]
    for i, o in enumerate(others):
        ths.append(threading.Thread(target=worker, args=("o%d" % i, o)))
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    a2 = results.get("P")
    a3 = gen(P)

    check("单独 vs 并发 同一 prompt 逐字节一致", a1 == a2, "a1=%r a2=%r" % (a1[:60], (a2 or "")[:60]))
    check("单独 vs 重发 同一 prompt 逐字节一致", a1 == a3, "a1=%r a3=%r" % (a1[:60], a3[:60]))

    # 长 prompt（触发分块 prefill）与短 prompt 混批，前缀必须一致
    longp = P + " " + " ".join(["word"] * 200)
    b1 = gen(longp, max_tokens=8)
    b2 = None
    ths = [threading.Thread(target=worker, args=("L", longp, 8)),
           threading.Thread(target=worker, args=("s", "hi", 8))]
    results.clear()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    b2 = results.get("L")
    check("分块 prefill（长 prompt）与其他请求混批一致", b1 == b2,
          "b1=%r b2=%r" % (b1[:60], (b2 or "")[:60]))

    # max_tokens 截断：前 N 个 token 必须与更长请求的前缀相同
    c4 = gen(P, max_tokens=4)
    c16 = gen(P, max_tokens=16)
    check("max_tokens=4 是 max_tokens=16 的前缀", c16.startswith(c4), "c4=%r c16=%r" % (c4, c16[:20]))

    print(("\n全部通过（batch 组成不影响结果）" if not fails else "\n%d 项不一致" % fails))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())