"""G22b 混跑投机（MTP）验收：投机/不投机、单发/并发，输出必须逐字一致。

用随机 MTP 头（tests/gen_mtp_head.py 生成，接受率≈0）：拒绝采样必须把
烂草稿全拒掉，所以「投机开/关」「单发/4 并发混跑」四种组合的 greedy 输出
都得一样——混跑路径（hlast 按行快照/轮转恢复、投机行剔除出 decode 批）若
有错，输出必然分叉。

用法：python tools/spec_serve_check.py [--port 30141]
"""
import argparse
import atexit
import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(os.path.dirname(ROOT), "models", "qwen3next-rand")
CUDA_HTTP = os.path.join(ROOT, "build-cuda", "bin", "Release", "cuda_http.exe")

PROMPTS = ["The meaning of life is", "Once upon a time there was", "def fibonacci(n):"]
PROMPTS_CC = ["In a galaxy far away", "Q: 2+2? A:"]

server = None


def wait_health(port, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as h:
                if h.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def start(port, spec):
    global server
    env = dict(os.environ,
               SGL_CUDA_CTX="4096", SGL_CUDA_W8="1", SGL_MTP="1",
               SGL_LIN_SLOTS="8")
    if spec:
        env["SGL_SPEC_W"] = "3"
    else:
        env.pop("SGL_SPEC_W", None)
    server = subprocess.Popen(
        [CUDA_HTTP, MODEL, str(port)], env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if not wait_health(port):
        print(f"FAIL 服务没起来 (spec={spec})")
        sys.exit(1)


def stop():
    global server
    if server:
        server.terminate()
        try:
            server.wait(timeout=10)
        except Exception:
            server.kill()
        server = None
        time.sleep(1)


def gen(port, text, max_new=40):
    body = json.dumps({"text": text, "sampling_params": {
        "max_new_tokens": max_new, "temperature": 0.0}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode())


def main():
    global server
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=30141)
    a = ap.parse_args()
    atexit.register(stop)
    fails = []

    def check(name, ok, detail=""):
        print(("PASS " if ok else "FAIL ") + name + (f"  <- {detail}" if not ok and detail else ""))
        if not ok:
            fails.append(name)

    # ---- 1) 基线：投机关（同 W8 环境，消除量化差异）----
    start(a.port, spec=False)
    base = [gen(a.port, p) for p in PROMPTS]
    stop()

    # ---- 2) 投机开：单发逐条 == 基线 ----
    start(a.port, spec=True)
    for i, p in enumerate(PROMPTS):
        r = gen(a.port, p)
        check(f"spec 单发 #{i} == 基线", r["text"] == base[i]["text"],
              f"{r['text'][:40]!r} vs {base[i]['text'][:40]!r}")

    # ---- 3) 混跑：3 个贪心 + 1 个采样（非资格）并发 == 基线 ----
    import threading
    out = {}

    def one(i, p):
        out[i] = gen(a.port, p)

    def sampled():
        out["s"] = gen(a.port, PROMPTS_CC[0])

    ths = [threading.Thread(target=one, args=(i, p)) for i, p in enumerate(PROMPTS)]
    ths.append(threading.Thread(target=sampled))
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    for i in range(len(PROMPTS)):
        check(f"spec 混跑 #{i} == 基线", out[i]["text"] == base[i]["text"],
              f"{out[i]['text'][:40]!r} vs {base[i]['text'][:40]!r}")
    check("混跑含采样请求不崩", isinstance(out.get("s", {}).get("text"), str))
    stop()

    if fails:
        print(f"\n{len(fails)} 项不一致: " + ", ".join(fails))
        return 1
    print("\n混跑投机验收全过（随机 MTP 头：投机/混跑输出 == 不投机基线）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
