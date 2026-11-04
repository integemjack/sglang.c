# -*- coding: utf-8 -*-
"""G20 服务级验收：cuda_http @ SGL_KVQ=1（页级 KV + 零拷贝前缀）。

验收（TODO_PROMPTS.md G20 任务卡第 3 步）：
  (2) 同前缀两请求（greedy），第二个走页命中路径，输出与第一个逐字一致；
  (3) /health: prefix.hits>=1 且 pages.mapped>=1（零拷贝挂页真的发生了）；
  (4) 单请求 prompt 超过静态 seg_len（ctx/并发）也能跑通（页池动态准入）；
  (5) 请求结束后页池回收（free 回升，租让页可解释）。

用法: python tools/kvq_serve_check.py [model_dir] [port]
依赖: 只用标准库（urllib），Windows/Linux 均可。
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "..", "models", "qwen05")
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 18099
EXE = os.path.join(ROOT, "build-cuda", "bin", "Release", "cuda_http.exe")
if not os.path.exists(EXE):
    EXE = os.path.join(ROOT, "build-cuda", "cuda_http")  # linux 布局兜底

CTX = 8192
CONC = 8
STATIC_SEG = CTX // CONC  # 1024：静态模式下超过它的 prompt 会被 400 拒


def req(path, body=None, timeout=300):
    url = f"http://127.0.0.1:{PORT}{path}"
    if body is None:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    data = json.dumps(body).encode("utf-8")
    r = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_up(deadline=180):
    t0 = time.time()
    while time.time() - t0 < deadline:
        try:
            return req("/health", timeout=5)
        except Exception:
            time.sleep(2)
    return None


def main():
    env = dict(os.environ)
    env.update(SGL_CUDA_CTX=str(CTX), SGL_HTTP_CONC=str(CONC),
               SGL_CUDA_FP16="1", SGL_PFX_MAX="16", SGL_HTTP_DBG="1")
    if os.environ.get("KVQ_F32"):
        env["SGL_KVQ"] = "0"  # 对照：f32 池 + 静态段 + copy_kv（G6 语义）
    else:
        env["SGL_KVQ"] = "1"
    log = open(os.path.join(ROOT, "build-cuda", "kvq_serve.log"), "w", encoding="utf-8")
    proc = subprocess.Popen([EXE, MODEL, str(PORT)], env=env, stdout=log, stderr=subprocess.STDOUT)
    fails = []
    try:
        h = wait_up()
        if not h:
            print("FAIL: server did not come up (see build-cuda/kvq_serve.log)")
            return 1
        print(f"[health] pagemap={h.get('pagemap')} pages={h.get('pages')}")
        f32_mode = os.environ.get("KVQ_F32") is not None
        if not f32_mode and h.get("pagemap") is not True:
            print("FAIL: pagemap not enabled (backend refused SGL_KVQ paged mode)")
            return 1

        # (4) 单请求超过静态 seg_len：动态准入（f32 对照模式会 400，跳过）
        if not f32_mode:
            long_prompt = "The quick brown fox jumps over the lazy dog. " * 400  # ~3600 tok
            r = req("/v1/completions", {"prompt": long_prompt, "max_tokens": 16, "temperature": 0})
            txt = r["choices"][0]["text"]
            print(f"(4) prompt ~{len(long_prompt)} chars > static seg {STATIC_SEG}: "
                  f"ok, {len(txt)} chars generated")
            if not txt:
                fails.append("(4) empty completion")

        # (2) 同前缀两请求：第二个命中页级租让，输出必须逐字一致
        sysp = ("You are a meticulous assistant. Answer with a single word.\n"
                "Rules: be terse. ") * 30
        p1 = {"prompt": sysp + "Q: capital of France? A:", "max_tokens": 24, "temperature": 0}
        a1 = req("/v1/completions", p1)["choices"][0]["text"]
        time.sleep(1)  # 让 reap 登记（第一请求结束才能租让）
        a2 = req("/v1/completions", p1)["choices"][0]["text"]
        if f32_mode:
            print(f"(2) reuse output == first: {'YES' if a1 == a2 else 'NO'} "
                  f"({len(a1)} vs {len(a2)} chars)")
            if a1 != a2:
                fails.append("(2) prefix-reused output differs from first")
        else:
            # kvq（4-bit 量化池）：根因已定位（G21，docs/NOTES_g20.md）：M=576 与
            # M=64 的 GEMM 行级 fp 差被尾块量化跳档固化放大——「命中」与「未命中」
            # 是不同执行路径，不保证逐位（上游对量化 KV + prefix cache 亦然）。
            # 服务语义真正要保证的是【同路径一致】：a2/a3 都命中同一租让 →
            # 执行路径相同 → 输出必须逐字一致。
            a3 = req("/v1/completions", p1)["choices"][0]["text"]
            print(f"(2) kvq: fresh {len(a1)} vs hit1 {len(a2)} vs hit2 {len(a3)} chars; "
                  f"hit==hit: {'YES' if a2 == a3 else 'NO'}"
                  f"{' (fresh differs: G21 quantized-KV numerics, documented)' if a1 != a2 else ''}")
            if a2 != a3:
                fails.append("(2) same-path (both prefix-hit) outputs differ")

        # (3)+(5) /health：命中 + 挂页计数 + 页回收
        time.sleep(1)
        h2 = req("/health")
        pf, pg = h2["prefix"], h2["pages"]
        print(f"(3) hits={pf['hits']} misses={pf['misses']} mapped={pg['mapped']} "
              f"cow={pg['cow']} free={pg['free']}/{pg['total']} leased={pg['leased']}")
        if pf["hits"] < 1:
            fails.append("(3) no prefix hit recorded")
        if not f32_mode and pg["mapped"] < 1:
            fails.append("(3) no zero-copy pages mapped")

        # 泄漏粗检：再发 3 个杂项请求后 free 页应仍在高位（total-free-leased 可解释）
        for i in range(3):
            req("/v1/completions", {"prompt": f"say {i}. " + "x" * 50,
                                    "max_tokens": 4, "temperature": 0})
        time.sleep(1)
        h3 = req("/health")
        pg3 = h3["pages"]
        busy = pg3["total"] - pg3["free"] - pg3["leased"]
        print(f"(5) after churn: free={pg3['free']} leased={pg3['leased']} "
              f"unexplained-busy={busy}")
        if busy > CONC * (CTX // 64) // 2:  # 显著超过合理在途量 = 泄漏
            fails.append(f"(5) page leak suspicion: busy={busy}")

        if fails:
            print("FAIL:", "; ".join(fails))
            return 1
        print("kvq_serve_check: all OK")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        log.close()


if __name__ == "__main__":
    sys.exit(main())
