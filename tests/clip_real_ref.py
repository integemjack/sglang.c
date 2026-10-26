"""G8: 真实 CLIP 权重 —— 转换 + torch 分层参考导出。

用法: python clip_real_ref.py <clip_dir>
  <clip_dir> 需含 pytorch_model.bin（openai/clip-vit-base-patch32）
产物（写进 <clip_dir>）:
  model.safetensors      视觉塔权重（F32，保留 HF 原张量名）
  ref/pv.npy             HF CLIPImageProcessor 处理 test.png 后的 pixel_values [1,3,224,224]
  ref/emb.npy            embeddings（class+patch+pos，pre_ln 之前）[1,50,768]
  ref/preln.npy          pre_layrnorm 之后（= encoder 输入）[1,50,768]
  ref/layer00..11.npy    每个 encoder layer 输出 [1,50,768]
  ref/last.npy           last_hidden_state [1,50,768]
  ref/pooled.npy         post_layernorm(pooled cls) [1,768]
  ref/image_embeds.npy   visual_projection(pooled) [1,512]
  ref/meta.json          各阶段含义 + 结构判定（pre_ln 是否 affine、post_ln 位置）
另外生成 test.png / test.jpg（固定种子随机图）与 ref/expect_norm.npy
  （PIL 解码→resize(BICUBIC)→center_crop→/255→normalize 的 CHW fp32 [3,224,224]）
"""
import json
import os
import struct
import sys

import numpy as np
import torch

D = sys.argv[1] if len(sys.argv) > 1 else "."
REF = os.path.join(D, "ref")
os.makedirs(REF, exist_ok=True)

# ---------- 1) 只取视觉塔，转 F32 safetensors（已存在则跳过）----------
STF = os.path.join(D, "model.safetensors")
if os.path.exists(STF) and "--force" not in sys.argv:
    print("model.safetensors 已存在，跳过转换")
else:
    sd = torch.load(os.path.join(D, "pytorch_model.bin"), map_location="cpu")
    vis = {k: v.to(torch.float32).contiguous().numpy()
           for k, v in sd.items() if k.startswith("vision_model.") or k == "visual_projection.weight"}
    del sd
    hdr, off = {}, 0
    for k, a in vis.items():
        b = a.tobytes()
        hdr[k] = {"dtype": "F32", "shape": list(a.shape), "data_offsets": [off, off + len(b)]}
        off += len(b)
    hb = json.dumps(hdr).encode()
    with open(STF, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for a in vis.values():
            f.write(a.tobytes())
    del vis
    print(f"model.safetensors: {len(hdr)} vision tensors, {off/1e6:.1f} MB data")

def st_load(name):
    """从刚写的 model.safetensors 读一个张量（不依赖 safetensors 包）"""
    with open(STF, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
        m = h[name]
        a, b = m["data_offsets"]
        f.seek(8 + n + a)
        return np.frombuffer(f.read(b - a), dtype="<f4").reshape(m["shape"]).copy()

# ---------- 2) 固定测试图 + HF 预处理 ----------
from PIL import Image

rng = np.random.default_rng(20261026)
# 方形测试图：CLIP 4.x 语义（直接缩到 img×img）与 5.x 语义（短边 224 + center_crop）
# 在方形输入上完全一致，C 侧只需实现一种。
arr = rng.integers(0, 256, (320, 320, 3), dtype=np.uint8)
arr[:80, :80, 0] = 255
Image.fromarray(arr, "RGB").save(os.path.join(D, "test.png"))
# JPEG 用平滑图（白噪声经 4:2:0 + DCT 后内容本身就变了，无法比解码器）：
# 渐变 + 几个圆块，quality=95、subsampling=0（4:4:4），两个解码器只差重建细节。
yy, xx = np.mgrid[0:320, 0:320]
sm = np.zeros((320, 320, 3), dtype=np.uint8)
sm[..., 0] = (xx * 255 // 319).astype(np.uint8)
sm[..., 1] = (yy * 255 // 319).astype(np.uint8)
sm[..., 2] = ((xx + yy) * 255 // 638).astype(np.uint8)
sm[60:140, 100:200] = (200, 60, 120)
cv = (yy - 240) ** 2 + (xx - 250) ** 2
sm[cv < 30 * 30] = (20, 220, 40)
Image.fromarray(sm, "RGB").save(os.path.join(D, "test.jpg"), quality=95, subsampling=0)

from transformers import CLIPImageProcessor, CLIPVisionModel

proc = CLIPImageProcessor.from_pretrained(D)
pv = proc(images=[Image.open(os.path.join(D, "test.png")).convert("RGB")],
          return_tensors="pt")["pixel_values"].numpy().astype(np.float32)
np.save(os.path.join(REF, "pv.npy"), pv)
print("pixel_values:", pv.shape, "range", pv.min(), pv.max())

# ---------- 3) torch 真实模型逐层导出 ----------
model = CLIPVisionModel.from_pretrained(D)
model.eval()
E = model.config.hidden_size
with torch.no_grad():
    full = model(pixel_values=torch.from_numpy(pv), output_hidden_states=True)
    hs = full.hidden_states
    print("hidden_states:", None if hs is None else (len(hs), tuple(hs[0].shape)))
    emb_raw = model.embeddings(torch.from_numpy(pv))          # [1,50,E]
    x = model.pre_layrnorm(emb_raw)
    np.save(os.path.join(REF, "emb.npy"), emb_raw.numpy().astype(np.float32))
    np.save(os.path.join(REF, "preln.npy"), x.numpy().astype(np.float32))
    if hs is not None:
        assert torch.allclose(hs[0], x, atol=1e-6), "hs[0] != pre_ln output"
        for i, h in enumerate(hs[1:]):
            np.save(os.path.join(REF, f"layer{i:02d}.npy"), h.numpy().astype(np.float32))
        last = hs[-1]
    else:
        for i, layer in enumerate(model.encoder.layers):
            x = layer(x)[0]
            np.save(os.path.join(REF, f"layer{i:02d}.npy"), x.numpy().astype(np.float32))
        last = x
    pooled = model.post_layernorm(last[:, 0, :])
    vproj = torch.from_numpy(st_load("visual_projection.weight"))
    img_emb = pooled @ vproj.t()
np.save(os.path.join(REF, "last.npy"), last.numpy().astype(np.float32))
np.save(os.path.join(REF, "pooled.npy"), pooled.numpy().astype(np.float32))
np.save(os.path.join(REF, "image_embeds.npy"), img_emb.numpy().astype(np.float32))

# 与整模型 forward 对账（确认手工逐层 == 模型自身输出）
with torch.no_grad():
    out = model(pixel_values=torch.from_numpy(pv))
d_last = float((out.last_hidden_state - last).abs().max())
d_pool = float((out.pooler_output - pooled).abs().max())

cfgv = model.config
meta = {
    "hidden_size": cfgv.hidden_size, "num_heads": cfgv.num_attention_heads,
    "num_layers": cfgv.num_hidden_layers, "patch_size": cfgv.patch_size,
    "image_size": cfgv.image_size, "intermediate_size": cfgv.intermediate_size,
    "projection_dim": cfgv.projection_dim, "layer_norm_eps": cfgv.layer_norm_eps,
    "hidden_act": cfgv.hidden_act,
    "pre_layrnorm_affine": bool(model.pre_layrnorm.elementwise_affine),
    "post_layernorm_affine": bool(model.post_layernorm.elementwise_affine),
    "manual_vs_forward_last_maxdiff": d_last,
    "manual_vs_forward_pooled_maxdiff": d_pool,
    "last_eq_encoder_out_only": True,
}
json.dump(meta, open(os.path.join(REF, "meta.json"), "w"), indent=1)
print(json.dumps(meta, indent=1))

# ---------- 4) 图片解码期望值（PIL 语义：resize BICUBIC → center_crop → /255 → normalize）----------
def pil_norm(path):
    im = Image.open(path).convert("RGB")
    r = im.resize((224, 224), Image.BICUBIC)         # CLIP: 直接缩到 224×224
    left = (r.width - 224) // 2
    top = (r.height - 224) // 2
    c = r.crop((left, top, left + 224, top + 224))   # center_crop（此处恒等）
    a = np.asarray(c).astype(np.float32) / 255.0
    return (a - np.array(proc.image_mean, dtype=np.float32)) / np.array(proc.image_std, dtype=np.float32)

a = pil_norm(os.path.join(D, "test.png"))
np.save(os.path.join(REF, "expect_norm.npy"), a.transpose(2, 0, 1).astype(np.float32))
np.save(os.path.join(REF, "expect_norm_png.npy"), a.transpose(2, 0, 1).astype(np.float32))
print("expect_norm vs pv maxdiff:",
      float(np.abs(a - pv[0].transpose(1, 2, 0)).max()))
print("expect_norm:", a.shape, "max", float(np.abs(a).max()))
assert float(np.abs(a - pv[0].transpose(1, 2, 0)).max()) < 0.02, "PIL 路径与 HF processor 不一致"

aj = pil_norm(os.path.join(D, "test.jpg"))
np.save(os.path.join(REF, "expect_norm_jpg.npy"), aj.transpose(2, 0, 1).astype(np.float32))

# 解码后的原始像素（分开看「解码器差」与「resize 差」）
np.save(os.path.join(REF, "png_rgb.npy"),
        np.asarray(Image.open(os.path.join(D, "test.png")).convert("RGB")).astype(np.float32))
np.save(os.path.join(REF, "jpg_rgb.npy"),
        np.asarray(Image.open(os.path.join(D, "test.jpg")).convert("RGB")).astype(np.float32))
print("PIL 自身 jpg(平滑图) vs png(噪声图) 不可比，只报各自尺寸")
