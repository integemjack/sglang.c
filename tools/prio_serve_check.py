"""W2 GPU 等待队列优先级验证（GAP B3 复核）。

SGL_HTTP_CONC=1 制造单槽：A（prio 0）先占槽长跑，B（prio 5）与 C（prio 0）
随后排队。断言：B 比 C 先完成（等待队列按 priority 插队，同优先级 FIFO）。

用法：python tools/prio_serve_check.py [--port 30161]
"""
import argparse
import atexit
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(os.path.dirname(ROOT), "models", "qwen3next-rand")
CUDA_HTTP = os.path.join(ROOT, "build-cuda", "bin", "Release", "cuda_http.exe")

server = None


def stop():
    global server
    if server:
        server.terminate()
        try:
            server.wait(timeout=10)
        except Exception:
            server.kill()
        server = None


def gen(port, text, prio, max_new):
    body = json.dumps({"text": text, "priority": prio, "sampling_params": {
        "max_new_tokens": max_new, "temperature": 0.0, "ignore_eos": True}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        json.loads(r.read().decode())
    return time.time() - t0


def main():
    global server
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=30161)
    a = ap.parse_args()
    atexit.register(stop)
    env = dict(os.environ, SGL_HTTP_CONC="1", SGL_LIN_SLOTS="4",
               SGL_CUDA_W8="1")  # qwen3next 的 linear-attn 层必须 W8
    server = subprocess.Popen([CUDA_HTTP, MODEL, str(a.port)], env=env,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(120):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=2) as h:
                if h.status == 200:
                    break
        except Exception:
            time.sleep(0.5)
    else:
        print("FAIL 服务没起来")
        return 1

    done = {}

    def run(tag, prio, max_new, delay):
        time.sleep(delay)
        done[tag] = gen(a.port, f"request {tag} says", prio, max_new)

    ths = [threading.Thread(target=run, args=("A", 0, 80, 0.0)),
           threading.Thread(target=run, args=("B", 5, 30, 0.8)),
           threading.Thread(target=run, args=("C", 0, 30, 0.9))]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    # A 先占槽；B(prio 5) 与 C(prio 0) 排队 → B 必须先完成
    b_fin = 0.8 + done["B"]
    c_fin = 0.9 + done["C"]
    ok = b_fin < c_fin
    print(f"A 总时长 {done['A']:.1f}s | B(prio5) 完成@{b_fin:.1f}s  C(prio0) 完成@{c_fin:.1f}s")
    print(("PASS " if ok else "FAIL ") + "等待队列优先级生效（B 先于 C）")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
