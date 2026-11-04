"""与上游 sglang 的语义对拍（conformance vs upstream）。

与 tests/conformance.py（CPU vs GPU 内部对拍）互补：这里对照物是
  * 上游 sglang 源码里写明的服务语义（stop 字符串截断、finish_reason、
    SSE 帧格式、stream_options.include_usage 的 usage 块）
  * HF transformers（本机 pip 装的官方实现）当真值：分词器 encode/decode、
    chat template 渲染

用法（脚本自己起服务、跑完自己收；--engine cuda 用 cuda_http.exe 跑 GPU 栈）：
  python tests/conformance_upstream.py
  python tests/conformance_upstream.py --engine cuda
  python tests/conformance_upstream.py --model <dir> --port 30121 --keep

退出码 0 = 全过；1 = 有 FAIL（逐条打印）。
"""
import argparse
import atexit
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DEFAULT = os.path.join(os.path.dirname(ROOT), "models", "qwen3-06")
SGLC = os.path.join(ROOT, "build", "bin", "sglc.exe")
# CUDA 栈在 build-cuda（SGL_CUDA=ON，MSVC 多配置 → bin/Release）
CUDA_HTTP = next((p for p in (
    os.path.join(ROOT, "build-cuda", "bin", "Release", "cuda_http.exe"),
    os.path.join(ROOT, "build", "bin", "cuda_http.exe")) if os.path.exists(p)),
    os.path.join(ROOT, "build-cuda", "bin", "Release", "cuda_http.exe"))

PROMPTS = [
    "The quick brown fox jumps over the lazy dog",
    "def f(x):\n    return x + 1\n",
    "Once upon a time there was",
    "你好，世界",
]
# 毒样本：专打预分词的边界（多空格/数字串/缩写/混排/悬挂空白）
FUZZ = [
    "a  b", "x   y    z", "  leading", "trailing   ", "123 4567", "3.14159",
    "don't CAN'T", "（中文）and English", "tab\there", "nl\n\nnl2",
    " \n \n ", "e=mc^2", "a\tb\tc", "İstanbul", "naïve café", "x=1;y=2;",
]
fails = []
server = None


def post(port, path, body, timeout=300):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:%d%s" % (port, path), data=data,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode("utf-8", "replace"))
        except Exception:  # noqa: BLE001
            return e.code, {}
    except Exception as e:  # noqa: BLE001
        return -1, {"error": str(e)}


def sse(port, path, body, timeout=300):
    """收一遍 SSE，返回 (data 帧列表, 拼接文本, 错误)。data 帧是解析后的对象。"""
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:%d%s" % (port, path), data=data,
        headers={"Content-Type": "application/json"})
    frames, text, err = [], [], None
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
                frames.append(j)
                for ch in j.get("choices", []):
                    text.append(ch.get("text") or
                                (ch.get("delta") or {}).get("content") or "")
    except Exception as e:  # noqa: BLE001
        err = str(e)
    return frames, "".join(text), err


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else "  <- " + str(detail)[:300]))
    if not ok:
        fails.append(name)


def wait_health(port, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(
                    "http://127.0.0.1:%d/health" % port, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:  # noqa: BLE001
            time.sleep(1.0)
    return False


def main():
    global server
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--port", type=int, default=30121)
    ap.add_argument("--engine", default="cpu", choices=["cpu", "cuda"],
                    help="cuda = 用 build/bin/cuda_http.exe 起 GPU 服务")
    ap.add_argument("--keep", action="store_true", help="跑完不关服务")
    a = ap.parse_args()

    # ---- 起服务（串行唯一实例；cuda_http.exe <model> <port>）----
    exe = CUDA_HTTP if a.engine == "cuda" else SGLC
    cmd = ([exe, a.model, str(a.port)] if a.engine == "cuda" else
           [exe, "--model", a.model, "--serve", "--port", str(a.port), "--device", "cpu"])
    server = subprocess.Popen(
        cmd,
        env=dict(os.environ, **({"SGL_CUDA_CTX": "8192"} if a.engine == "cuda" else {})),
        # 「press Enter to stop」读 stdin：管道不写就永不 EOF，否则 CPU 服务自退
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if not a.keep:
        atexit.register(lambda: (server.terminate(), server.wait(timeout=10)))
    if not wait_health(a.port):
        print("FAIL 服务没起来 (%s port=%d)" %
              ("cuda_http" if a.engine == "cuda" else "sglc --serve", a.port))
        return 1

    # HF 真值
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)

    # ---- 1) tokenize == HF encode（含中文/代码 + 毒样本）----
    for t in PROMPTS + FUZZ:
        ref = tok.encode(t)
        s, r = post(a.port, "/tokenize", {"text": t})
        check("tokenize==HF: %r" % t[:24],
              s == 200 and r.get("tokens") == ref,
              (s, str(r)[:120], str(ref[:8])))

    # ---- 2) detokenize == HF decode ----
    for t in PROMPTS[:2]:
        ids = tok.encode(t)
        s, r = post(a.port, "/detokenize", {"tokens": ids})
        check("detokenize==HF: %r" % t[:24],
              s == 200 and r.get("text") == tok.decode(ids), (s, str(r)[:120]))

    # ---- 3) chat template：内部渲染 == HF apply_chat_template（行为等价法）----
    msgs = [{"role": "user", "content": "Give me three numbers."}]
    rendered = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=False)
    s1, r1 = post(a.port, "/v1/completions",
                  {"model": "x", "prompt": rendered, "max_tokens": 24, "temperature": 0})
    s2, r2 = post(a.port, "/v1/chat/completions",
                  {"model": "x", "messages": msgs, "max_tokens": 24, "temperature": 0})
    if s1 == 200 and s2 == 200:
        c1 = r1["choices"][0]["text"]
        c2 = r2["choices"][0]["message"]["content"]
        check("chat 模板行为==HF 渲染+续写", c1 == c2,
              "rendered续写=%r chat=%r" % (c1[:60], c2[:60]))
        check("chat finish_reason 存在", r2["choices"][0].get("finish_reason") in ("stop", "length"),
              r2["choices"][0])
    else:
        check("chat 模板行为==HF 渲染+续写", False, (s1, s2, str(r1)[:120], str(r2)[:120]))

    # ---- 4) usage 数字对得上 HF 的 token 数 ----
    p = PROMPTS[0]
    s, r = post(a.port, "/v1/completions",
                {"model": "x", "prompt": p, "max_tokens": 8, "temperature": 0})
    if s == 200:
        u = r.get("usage", {})
        check("usage.prompt_tokens==len(HF encode)",
              u.get("prompt_tokens") == len(tok.encode(p)), u)
        check("usage.completion_tokens<=max_tokens 且>0",
              0 < u.get("completion_tokens", -1) <= 8, u)
        check("finish_reason==length（被 max_tokens 截断）",
              r["choices"][0].get("finish_reason") == "length", r["choices"][0])
    else:
        check("usage 数字", False, (s, str(r)[:150]))

    # ---- 5) stop 字符串（上游：输出截断在停止串之前、不含停止串、finish=stop）----
    # 先无限制贪心跑一段，从产物里挑一个词当 stop
    s, base = post(a.port, "/v1/completions",
                   {"model": "x", "prompt": PROMPTS[2], "max_tokens": 48, "temperature": 0})
    if s != 200:
        check("stop 字符串", False, (s, str(base)[:150]))
    else:
        full = base["choices"][0]["text"]
        words = [w for w in full.replace("\n", " ").split(" ") if len(w) >= 3]
        stop_w = words[len(words) // 2] if words else None
        if not stop_w:
            print("SKIP stop 字符串（48 token 产物里挑不出词）")
        else:
            s2, r2 = post(a.port, "/v1/completions",
                          {"model": "x", "prompt": PROMPTS[2], "max_tokens": 48,
                           "temperature": 0, "stop": [stop_w]})
            if s2 != 200:
                check("stop 字符串", False, (s2, str(r2)[:150]))
            else:
                cut = r2["choices"][0]["text"]
                hit = stop_w in full
                ok = cut in full and full.find(cut) >= 0
                if hit:
                    # 截断点必须恰在停止串首现处之前（不含停止串）
                    at = full.find(stop_w)
                    ok = ok and cut == full[:at] and \
                        r2["choices"][0]["finish_reason"] == "stop"
                else:
                    ok = ok and cut == full and \
                        r2["choices"][0]["finish_reason"] in ("stop", "length")
                check("stop=%r 截断语义（上游：不含停止串/finish=stop）" % stop_w[:12],
                      ok, "cut=%r full=%r fin=%r" %
                      (cut[:50], full[:50], r2["choices"][0].get("finish_reason")))
                # 5b) stop 也接受裸字符串（OpenAI 允许 str 或 list）
                s3, r3 = post(a.port, "/v1/completions",
                              {"model": "x", "prompt": PROMPTS[2], "max_tokens": 48,
                               "temperature": 0, "stop": stop_w})
                check("stop 裸字符串形式等价", s3 == 200 and
                      (s3 != 200 or r3["choices"][0]["text"] == cut),
                      (s3, str(r3)[:120]))
                # 5c) 流式截断后拼装 == 非流式
                frames, stitched, err = sse(a.port, "/v1/completions",
                                            {"model": "x", "prompt": PROMPTS[2],
                                             "max_tokens": 48, "temperature": 0,
                                             "stream": True, "stop": [stop_w]})
                check("stop 流式拼装==非流式", stitched == cut and err is None,
                      "stitch=%r cut=%r err=%s" % (stitched[:50], cut[:50], err))
                # 5d) 跨 token 边界的长词 stop（悬挂回扣：流式不能先吐出停止串前缀）
                longs = [w for w in words if len(w) >= 5]
                stop_l = max(longs, key=len) if longs else None
                if stop_l and stop_l in full:
                    s4, r4 = post(a.port, "/v1/completions",
                                  {"model": "x", "prompt": PROMPTS[2], "max_tokens": 48,
                                   "temperature": 0, "stop": [stop_l]})
                    fr2, st2, err2 = sse(a.port, "/v1/completions",
                                         {"model": "x", "prompt": PROMPTS[2],
                                          "max_tokens": 48, "temperature": 0,
                                          "stream": True, "stop": [stop_l]})
                    at2 = full.find(stop_l)
                    ok2 = (s4 == 200 and r4["choices"][0]["text"] == full[:at2] and
                           r4["choices"][0]["finish_reason"] == "stop" and
                           st2 == full[:at2] and err2 is None and
                           not st2.endswith(stop_l[:len(stop_l) - 1] or "\x00"))
                    check("长词 stop=%r 截断+流式（不吐前缀）" % stop_l[:10], ok2,
                          "cut=%r stitch=%r at=%d" %
                          (r4["choices"][0]["text"][:40], st2[:40], at2))

    # ---- 6) SSE 帧格式（上游 openai 协议）----
    frames, stitched, err = sse(a.port, "/v1/completions",
                                {"model": "x", "prompt": PROMPTS[0], "max_tokens": 12,
                                 "temperature": 0, "stream": True})
    s_whole, r_whole = post(a.port, "/v1/completions",
                            {"model": "x", "prompt": PROMPTS[0], "max_tokens": 12,
                             "temperature": 0})
    ok = err is None and frames and s_whole == 200
    if ok:
        fin = [f for f in frames if f.get("choices") and
               f["choices"][0].get("finish_reason")]
        ok = len(fin) == 1 and stitched == r_whole["choices"][0]["text"]
    check("SSE：恰好一个 finish 帧、拼装==整体", ok,
          "frames=%d fin=%d stitch=%r whole=%r err=%s" %
          (len(frames), len([f for f in frames if f.get("choices") and
                             f["choices"][0].get("finish_reason")]),
           stitched[:40], r_whole["choices"][0]["text"][:40] if s_whole == 200 else "?", err))

    # ---- 7) stream_options.include_usage（上游：[DONE] 前多一个 usage 帧）----
    frames, _, err = sse(a.port, "/v1/completions",
                         {"model": "x", "prompt": PROMPTS[0], "max_tokens": 8,
                          "temperature": 0, "stream": True,
                          "stream_options": {"include_usage": True}})
    usage_frames = [f for f in frames if f.get("usage")]
    ok = (err is None and usage_frames and
          usage_frames[-1].get("choices") in (None, []) and
          usage_frames[-1]["usage"].get("completion_tokens", 0) > 0)
    check("stream_options.include_usage -> usage 帧", ok,
          "usage帧=%d err=%s" % (len(usage_frames), err))

    # ---- 8) /generate 原生面（上游 native 协议的最小语义）----
    s, r = post(a.port, "/generate",
                {"text": PROMPTS[0],
                 "sampling_params": {"max_new_tokens": 8, "temperature": 0}})
    mi = r.get("meta_info", {}) if isinstance(r, dict) else {}
    check("/generate: completion + meta_info.finish_reason/completion_tokens",
          s == 200 and isinstance(r.get("completion") or r.get("text"), str) and
          mi.get("finish_reason") in ("stop", "length") and
          isinstance(mi.get("completion_tokens"), int),
          (s, str(r)[:150]))

    print()
    if fails:
        print("%d 项与上游不一致: %s" % (len(fails), ", ".join(fails)))
        return 1
    print("全部与上游语义一致（HF 真值 %s）" % a.model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
