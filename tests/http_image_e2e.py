"""G8 e2e: 真实 JPEG/PNG 字节 → HTTP /v1/chat/completions（OpenAI vision 风格）→ 输出随图变。
用法: python http_image_e2e.py <sglc_exe> <model_dir> <vision_dir> <clip_dir> [port]"""
import base64, json, subprocess, sys, time, os
import requests

SGLC, MODEL, VISION, CLIPDIR = sys.argv[1:5]
PORT = sys.argv[5] if len(sys.argv) > 5 else "30077"
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"

def post(parts, mx=8, seed=7):
    body = {"messages": [{"role": "user", "content": parts}],
            "max_tokens": mx, "temperature": 0.0, "seed": seed}
    r = requests.post(URL, json=body, timeout=120)
    return r.status_code, r.text

def img_part(path, mime):
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}

proc = subprocess.Popen([SGLC, "--model", MODEL, "--serve", "--host", "127.0.0.1",
                         "--port", PORT, "--vision", VISION],
                        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL)
try:
    ok = False
    for _ in range(60):
        try:
            if requests.get(f"http://127.0.0.1:{PORT}/health", timeout=2).status_code == 200:
                ok = True
                break
        except Exception:
            time.sleep(1)
    assert ok, "server did not come up"

    txt = [{"type": "text", "text": "Q: what is in the picture? A:"}]
    jpg = os.path.join(CLIPDIR, "test.jpg")
    png = os.path.join(CLIPDIR, "test.png")

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
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except Exception:
        proc.kill()
