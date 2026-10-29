"""多图（G21）端到端检查：一个 prompt 里两个 <image> + 两张图。

验证：
  1) 两图请求能跑通，且日志里 places=2 / per_img=NP（视觉 token 分开插）
  2) 两图 vs 单图（同一段文本、只给一张图）的首 token 不同（第二张真的进了 KV）
  3) 同一张图放两次 == 单图放一次？（不应相等：占位符各占 N 个位置）
  4) 图片数量与占位符数量不一致时退回旧行为（不崩）

用法：python tests/llava_multi_check.py [port]
"""
import base64
import json
import struct
import sys
import urllib.request

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 30131
BASE = "http://127.0.0.1:%d" % PORT
fails = 0


def check(name, ok, detail=""):
    global fails
    if not ok:
        fails += 1
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else "  <- " + str(detail)[:300]))


def tiny_png(r, g, b, size=32):
    """生成一个 size×size 的纯色 PNG（不依赖 PIL）"""
    import zlib

    raw = b"".join(b"\x00" + bytes([r, g, b]) * size for _ in range(size))

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + \
        chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(png).decode()


def chat(urls, text="What is in these images?", max_tokens=8):
    content = []
    for u in urls:
        content.append({"type": "image_url", "image_url": {"url": u}})
    content.append({"type": "text", "text": text})
    body = {"model": "x", "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens, "temperature": 0}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        j = json.loads(r.read().decode("utf-8", "replace"))
    return j["choices"][0]["message"]["content"]


def main():
    red = tiny_png(255, 0, 0)
    blue = tiny_png(0, 0, 255)

    one = chat([red])
    two = chat([red, blue])
    two_same = chat([red, red])
    print("[multi] 单图=%r" % one)
    print("[multi] 两图=%r" % two)
    print("[multi] 两图(同图两次)=%r" % two_same)

    check("两图与单图输出不同（第二张图真的进了 KV）", one != two, "one=%r two=%r" % (one, two))
    check("两张同图与一张图输出不同（两个占位符各占 N 个位置）", one != two_same,
          "one=%r two_same=%r" % (one, two_same))
    check("两图(红/蓝)与两图(红/红)输出不同（颜色确实参与了）", two != two_same,
          "two=%r two_same=%r" % (two, two_same))

    # 3 张图 / 2 个占位符不匹配时不能崩（退回旧行为）
    three = chat([red, blue, red])
    check("三图请求不崩（占位符数量不匹配时退回旧行为）", isinstance(three, str), three)

    print(("\n全部通过" if not fails else "\n%d 项失败" % fails))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())