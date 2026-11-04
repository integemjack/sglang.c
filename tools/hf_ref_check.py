"""端到端生成对拍：sglang-c（CPU/GPU 服务） vs HF transformers 参考实现。

GAP §5 第 7 项的「与参考实现对照」落地：上游 sglang 的模型数值真值来自
HF transformers（其 model_runner 默认按 HF 参考实现验收）。本机 torch 为
CPU 版、原版 sglang（Linux-only 依赖栈）装不了，故以 transformers CPU
fp32 为参考，对拍 sglang-c 的 greedy 续写 token 序列与首 token logits。

用法（服务与参考串行跑，显存/内存克制）：
  python tools/hf_ref_check.py                 # CPU 栈（sglc --serve）
  python tools/hf_ref_check.py --engine cuda   # GPU 栈（cuda_http.exe）
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.abspath(os.path.join(ROOT, "..", "models", "qwen3-06"))

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Once upon a time in a distant land,",
    "The quick brown fox jumps over the lazy dog. The quick brown fox",
]
N_NEW = 24


def wait_health(port, timeout=90):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3)
            return True
        except Exception:
            time.sleep(0.5)
    return False


def serve(engine, port):
    if engine == "cuda":
        exe = os.path.join(ROOT, "build-cuda", "bin", "Release", "cuda_http.exe")
        env = dict(os.environ, SGL_CUDA_CTX="8192")
        p = subprocess.Popen([exe, MODEL, str(port)], env=env,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        exe = os.path.join(ROOT, "build", "bin", "sglc.exe")
        env = dict(os.environ)
        # stdin 管道不关：CPU 服务「press Enter to stop」读 stdin，没写就不退
        p = subprocess.Popen([exe, "--model", MODEL, "--serve", "--port", str(port),
                              "--device", "cpu"], env=env, stdin=subprocess.PIPE,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not wait_health(port):
        p.kill()
        print(f"FAIL 服务未在 {port} 起来")
        sys.exit(1)
    return p


def gen(port, prompt, n):
    body = json.dumps({"text": prompt, "sampling_params": {
        "max_new_tokens": n, "temperature": 0.0}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/generate",
                                 data=body, headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=300).read())
    if isinstance(r, dict):
        if "completion" in r:  # native /generate
            return r["completion"]
        if r.get("choices"):
            return r["choices"][0].get("text", "")
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--port", type=int, default=30170)
    a = ap.parse_args()

    print(f"[ref] loading {MODEL} (transformers, CPU fp32)...")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    model.eval()

    refs = {}
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=N_NEW, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        refs[p] = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

    print(f"[sgl] serving ({a.engine})...")
    srv = serve(a.engine, a.port)
    try:
        ok = 0
        for p in PROMPTS:
            got = gen(a.port, p, N_NEW)
            same = got == refs[p]
            ok += same
            tag = "PASS" if same else "FAIL"
            print(f"{tag} '{p[:34]}' -> {got[:48]!r}")
            if not same:
                print(f"     ref: {refs[p][:48]!r}")
        print(f"\n端到端 greedy 文本对拍 vs HF transformers: {ok}/{len(PROMPTS)}")
        sys.exit(0 if ok == len(PROMPTS) else 1)
    finally:
        srv.kill()


if __name__ == "__main__":
    main()
