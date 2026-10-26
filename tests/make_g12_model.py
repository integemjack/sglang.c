"""G12 测试模型：小 LM（mistral-rand）+ 2 层 MLP projector 的 llava 随机模型 +
带 <image> added token 的 tokenizer。

用法: python make_g12_model.py [--out C:/.../models/g12-mlp]
      [--lm <dir>] [--llava <dir>] [--tok <dir>]
默认：lm=../models/mistral-rand, llava=../models/llava-mlp（不存在则先生成）,
tok=../models/qwen3-06

为什么 <image> 要映射到已有 id：LM 的 vocab_size 是固定的，随机模型没法加词表条目，
所以这里把 "<image>" 登记成一个「未被测试文本用到的」现成 id（默认 vocab-1），
只用于验证「占位符位置插入」这条逻辑；真实 llava tokenizer 里它是词表内的 32000。
"""
import json, os, shutil, subprocess, sys

argv = sys.argv[1:]
def opt(name, default):
    return argv[argv.index(name) + 1] if name in argv and argv.index(name) + 1 < len(argv) else default

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                      # .../sglang-c
MODELS = os.path.join(os.path.dirname(ROOT), "models")
out = opt("--out", os.path.join(MODELS, "g12-mlp"))
lm = opt("--lm", os.path.join(MODELS, "mistral-rand"))
llava = opt("--llava", os.path.join(MODELS, "llava-mlp"))
tok = opt("--tok", os.path.join(MODELS, "qwen3-06"))

if not os.path.exists(os.path.join(llava, "projector.safetensors")):
    subprocess.check_call([sys.executable, os.path.join(HERE, "gen_random_llava.py"),
                           "--out", llava, "--mlp", "1", "--lm-e", "64"])

os.makedirs(out, exist_ok=True)
# LM
for f in ("config.json", "model.safetensors"):
    shutil.copy(os.path.join(lm, f), os.path.join(out, f))
# CLIP + projector
for f in ("clip_config.json", "clip_model.safetensors", "projector.safetensors"):
    shutil.copy(os.path.join(llava, f), os.path.join(out, f))
# tokenizer + 图片（make_g12_tokenizer 会顺带把 image.raw 转 png/jpg）
cfg = json.load(open(os.path.join(out, "config.json")))
V = cfg["vocab_size"]
img_id = V - 1
subprocess.check_call([sys.executable, os.path.join(HERE, "make_g12_tokenizer.py"), tok, out,
                       "--image-raw", os.path.join(llava, "image.raw"),
                       "--image-id", str(img_id)])
print(f"G12 test model -> {out} (LM E={cfg['hidden_size']} V={V}, <image> id={img_id})")