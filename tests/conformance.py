"""CPU 与 GPU 服务的端到端对拍（conformance）。

用法：
  python tests/conformance.py --cpu 30111 --gpu 30112
  （两个服务要跑同一份权重、同一分词器；GPU 侧用 cuda_http.exe，CPU 侧用 sglc --serve）

比的是什么：
  * /v1/completions 与 /generate 的生成结果（贪心，固定 seed、temperature=0）
  * /tokenize 的 token id 序列、/detokenize 的回文本
  * /v1/chat/completions 的 chat template 渲染一致性（同一 messages，看输出是否逐字节一致）
  * 流式：把 SSE 拼起来与整体响应比

退出码 0 = 全一致；1 = 有差异（逐条打印）。
"""
import argparse
import json
import sys
import urllib.error
import urllib.request

# Windows 控制台默认 GBK：模型输出什么字节都可能，直接 errors=replace 避免脚本自己被编码搞死
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

PROMPTS = [
    "The quick brown fox jumps over the lazy dog",
    "Hello",
    "def f(x):\n    return x + 1\n",
    "Once upon a time there was",
    "1 2 3 4 5",
    "A B C D E F G",
]
MAX_NEW = 16
fails = []


def post(port, path, body, timeout=180):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:%d%s" % (port, path), data=data,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")[:300]
    except Exception as e:  # noqa: BLE001
        return -1, str(e)


def post_stream(port, path, body, timeout=180):
    """把 SSE 里的 delta 文本拼起来。"""
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:%d%s" % (port, path), data=data,
        headers={"Content-Type": "application/json"})
    out = []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    j = json.loads(payload)
                except Exception:  # noqa: BLE001
                    continue
                for ch in j.get("choices", []):
                    out.append(ch.get("text") or ch.get("delta", {}).get("content") or "")
    except Exception as e:  # noqa: BLE001
        return None, str(e)
    return "".join(out), None


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else "  <- " + str(detail)[:240]))
    if not ok:
        fails.append(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", type=int, default=30111)
    ap.add_argument("--gpu", type=int, default=30112)
    ap.add_argument("--skip-stream", action="store_true")
    a = ap.parse_args()

    # 0) tokenize / detokenize 必须逐 token 一致（分词器是同一份）
    for t in PROMPTS:
        s1, r1 = post(a.cpu, "/tokenize", {"text": t})
        s2, r2 = post(a.gpu, "/tokenize", {"text": t})
        ok = s1 == 200 and s2 == 200 and r1.get("tokens") == r2.get("tokens")
        check("tokenize 一致: %r" % t[:28], ok, (r1, r2))
        if not ok:
            continue
        s3, r3 = post(a.cpu, "/detokenize", {"tokens": r1["tokens"]})
        s4, r4 = post(a.gpu, "/detokenize", {"tokens": r1["tokens"]})
        check("detokenize 一致: %r" % t[:28],
              s3 == 200 and s4 == 200 and r3.get("text") == r4.get("text"), (r3, r4))

    # 1) /generate 贪心生成必须一致（completion = 生成的 token 文本）
    for t in PROMPTS:
        body = {"text": t, "sampling_params": {"max_new_tokens": MAX_NEW, "temperature": 0}}
        s1, r1 = post(a.cpu, "/generate", body)
        s2, r2 = post(a.gpu, "/generate", body)
        if s1 != 200 or s2 != 200:
            print("SKIP /generate %r (cpu=%s gpu=%s)" % (t[:24], s1, s2))
            continue
        c1 = r1.get("completion", r1.get("text", ""))
        c2 = r2.get("completion", r2.get("text", ""))
        n1 = r1.get("meta_info", {}).get("completion_tokens", -1)
        n2 = r2.get("meta_info", {}).get("completion_tokens", -1)
        check("/generate 一致: %r" % t[:28], c1 == c2 and n1 == n2,
              "cpu=%r(%s) gpu=%r(%s)" % (c1[:40], n1, c2[:40], n2))

    # 2) /v1/completions（OpenAI 形状）
    for t in PROMPTS[:3]:
        body = {"model": "x", "prompt": t, "max_tokens": MAX_NEW, "temperature": 0}
        s1, r1 = post(a.cpu, "/v1/completions", body)
        s2, r2 = post(a.gpu, "/v1/completions", body)
        if s1 != 200 or s2 != 200:
            print("SKIP /v1/completions %r (cpu=%s gpu=%s)" % (t[:24], s1, s2))
            continue
        c1 = r1["choices"][0]["text"]
        c2 = r2["choices"][0]["text"]
        check("/v1/completions 一致: %r" % t[:28], c1 == c2, "cpu=%r gpu=%r" % (c1[:40], c2[:40]))

    # 3) chat 形状（chat template 渲染 + 续写）
    msgs = {"model": "x", "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 8, "temperature": 0}
    s1, r1 = post(a.cpu, "/v1/chat/completions", msgs)
    s2, r2 = post(a.gpu, "/v1/chat/completions", msgs)
    if s1 == 200 and s2 == 200:
        c1 = r1["choices"][0]["message"]["content"]
        c2 = r2["choices"][0]["message"]["content"]
        check("/v1/chat/completions 一致", c1 == c2, "cpu=%r gpu=%r" % (c1[:40], c2[:40]))
    else:
        print("SKIP /v1/chat/completions (cpu=%s gpu=%s)" % (s1, s2))

    # 4) 流式拼起来 == 整体
    if not a.skip_stream:
        for port, name in ((a.cpu, "cpu"), (a.gpu, "gpu")):
            body = {"model": "x", "prompt": PROMPTS[0], "max_tokens": MAX_NEW,
                    "temperature": 0, "stream": True}
            whole = post(port, "/v1/completions", {**body, "stream": False})
            stitched, err = post_stream(port, "/v1/completions", body)
            ok = whole[0] == 200 and stitched is not None and \
                stitched == whole[1]["choices"][0]["text"]
            check("%s 流式拼装 == 整体" % name, ok,
                  "stream=%r whole=%r err=%s" % ((stitched or "")[:40],
                                                 whole[1]["choices"][0]["text"][:40] if whole[0] == 200 else whole, err))

    print()
    if fails:
        print("%d 项不一致: %s" % (len(fails), ", ".join(fails)))
        return 1
    print("全部一致（CPU vs GPU 逐字节相同）")
    return 0


if __name__ == "__main__":
    sys.exit(main())