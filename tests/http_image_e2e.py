"""G8/G11 e2e: 真实 JPEG/PNG 字节 → HTTP /v1/chat/completions（OpenAI vision 风格）→ 输出随图变。

用法（二选一）：
  CPU 版（脚本自己起 sglc）：
    python http_image_e2e.py <sglc_exe> <model_dir> <vision_dir> <clip_dir> [port]
  GPU 版（先自己起 cuda_http，脚本只打端口；G11 新增）：
    python http_image_e2e.py --url http://127.0.0.1:30024 --images <含 test.png/test.jpg 的目录>
  --images 里放测试图（默认从 clip_dir 取）。

G11 参考值（mistral-rand + llava-rand 的 mm-test 组合）：
    jpeg/png/text 三种都 200，且同 seed 输出互不相同；坏 base64 → 400。
"""
import base64, json, os, subprocess, sys, time

import requests

# 输出里可能混入任何字节（随机小模型的 token 不一定是合法 UTF-8/GBK），
# Windows 控制台默认 GBK 会 UnicodeEncodeError —— 统一按 utf-8 + replace 输出。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

URL = None
IMG_DIR = None
proc = None

argv = sys.argv[1:]
if "--url" in argv:
    URL = argv[argv.index("--url") + 1]
    IMG_DIR = argv[argv.index("--images") + 1] if "--images" in argv else "."
    # 不打服务：只打已有的端口（cuda_http 那条路）。端口从 URL 里读，用于健康检查。
    PORT = URL.rsplit(":", 1)[-1].strip("/")
else:
    SGLC, MODEL, VISION, CLIPDIR = argv[:4]
    PORT = argv[4] if len(argv) > 4 else "30077"
    URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
    IMG_DIR = CLIPDIR

URL = URL.rstrip("/")
if not URL.endswith("/v1/chat/completions"):
    URL = URL + "/v1/chat/completions"


def post(parts, mx=8, seed=7):
    body = {"messages": [{"role": "user", "content": parts}],
            "max_tokens": mx, "temperature": 0.0, "seed": seed}
    r = requests.post(URL, json=body, timeout=120)
    return r.status_code, r.text


def img_part(path, mime):
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


if SGLC_MODE := ("--url" not in sys.argv[1:]):
    proc = subprocess.Popen([SGLC, "--model", MODEL, "--serve", "--host", "127.0.0.1",
                             "--port", PORT, "--vision", VISION],
                            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
try:
    base = URL.replace("/v1/chat/completions", "")
    ok = False
    for _ in range(90):
        try:
            if requests.get(f"{base}/health", timeout=2).status_code == 200:
                ok = True
                break
        except Exception:
            time.sleep(1)
    assert ok, "server did not come up (health check failed)"

    txt = [{"type": "text", "text": "Q: what is in the picture? A:"}]
    jpg = os.path.join(IMG_DIR, "test.jpg")
    png = os.path.join(IMG_DIR, "test.png")

    code, a = post(txt + [img_part(jpg, "image/jpeg")])
    code2, b = post(txt + [img_part(png, "image/png")])
    code3, t = post(txt)  # 无图
    print("jpeg http", code, "| png http", code2, "| text http", code3)

    def content(s):
        try:
            return json.loads(s)["choices"][0]["message"]["content"]
        except Exception:
            return s[:200]

    ca, cb, ct = content(a), content(b), content(t)
    print("with jpeg :", repr(ca))
    print("with png  :", repr(cb))
    print("no image  :", repr(ct))
    assert code == 200 and code2 == 200 and code3 == 200, "非 200"
    print("image-vs-image differs:", ca != cb)
    print("image-vs-text differs :", ca != ct)

    # 坏 base64 → 400
    bad = {"messages": [{"role": "user", "content":
            [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,!!!!"}}]}],
           "max_tokens": 4}
    r = requests.post(URL, json=bad, timeout=30)
    print("bad base64 ->", r.status_code, r.text[:120])
    assert r.status_code == 400
    print("E2E OK")
finally:
    if proc is not None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()