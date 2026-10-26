"""G12: 真实 CLIP-L/14-336 + 真实 llava mm_projector 的 torch 参考。

用 HF transformers（CPU）跑一遍：
  PNG → CLIPImageProcessor（真实 preprocessor_config） → CLIPVisionModel
  → hidden_states[mm_vision_select_layer]（llava-v1.5 是 -2）→ 去掉 class 行
  → mm_projector（linear_1 → GELU → linear_2）
把三个中间结果存成 .npy 供 C 侧对照：
  _ref_pv.npy    归一化后的输入 CHW [3,336,336]
  _ref_vis.npy   视觉塔 patch 行 [576,1024]
  _ref_real_proj.npy  projector 输出 [576,4096]

用法: python llava_real_ref.py --clip <clip-l-336 dir> --proj <llava-real dir>
产物还包含 test336.png（C 侧从这张图自己解码+预处理）。
"""
import argparse, json, os
import numpy as np
import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModel

ap = argparse.ArgumentParser()
ap.add_argument("--clip", required=True)
ap.add_argument("--proj", required=True)
a = ap.parse_args()
torch.manual_seed(0)

# ---- 测试图：336x336 的确定性图案（渐变 + 色块 + 噪声），存 PNG ----
H = W = 336
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
img = np.zeros((H, W, 3), np.float32)
img[..., 0] = xx / W
img[..., 1] = yy / H
img[..., 2] = ((xx // 28 + yy // 28) % 2).astype(np.float32) * 0.7
rng = np.random.default_rng(5)
img += (rng.random((H, W, 3)).astype(np.float32) - 0.5) * 0.25
img = np.clip(img, 0, 1)
png = np.clip(img * 255, 0, 255).astype(np.uint8)
Image.fromarray(png).save(os.path.join(a.clip, "test336.png"))
print("image:", a.clip + "/test336.png")

# ---- 预处理（真实 CLIPImageProcessor）----
proc = CLIPImageProcessor.from_pretrained(a.clip)
pv = proc(images=Image.open(os.path.join(a.clip, "test336.png")).convert("RGB"),
          return_tensors="pt")["pixel_values"]
print("pixel_values:", tuple(pv.shape), float(pv.min()), float(pv.max()))
np.save(os.path.join(a.clip, "_ref_pv.npy"), pv[0].numpy().astype(np.float32))

# ---- 视觉塔 ----
model = CLIPVisionModel.from_pretrained(a.clip, torch_dtype=torch.float32).eval()
lcfg = json.load(open(os.path.join(a.proj, "config.json"))) if os.path.exists(
    os.path.join(a.proj, "config.json")) else {}
sel = lcfg.get("mm_vision_select_layer", -1)
with torch.no_grad():
    out = model(pixel_values=pv, output_hidden_states=True)
hs = out.hidden_states[sel if sel >= 0 else len(out.hidden_states) + sel]
# llava 的 CLIPVisionTower：vision_feats = hidden_states[layer][:, 1:, :]
vis = hs[:, 1:, :]
print("hidden_states:", len(out.hidden_states), "select_layer:", sel,
      "vis:", tuple(vis.shape))
np.save(os.path.join(a.clip, "_ref_vis.npy"), vis[0].numpy().astype(np.float32))

# ---- mm_projector（真实权重，mlp2x_gelu）----
pj = torch.load(os.path.join(a.proj, "mm_projector.bin"), map_location="cpu",
                weights_only=True)
w1, b1 = pj["model.mm_projector.0.weight"].float(), pj["model.mm_projector.0.bias"].float()
w2, b2 = pj["model.mm_projector.2.weight"].float(), pj["model.mm_projector.2.bias"].float()
mid = torch.nn.functional.gelu(vis[0] @ w1.T + b1)          # GELU 默认 = erf 版
proj = mid @ w2.T + b2
print("projector:", tuple(proj.shape), "checksum", float(proj.sum()))
np.save(os.path.join(a.proj, "_ref_real_proj.npy"), proj.numpy().astype(np.float32))
print("saved _ref_pv.npy / _ref_vis.npy / _ref_real_proj.npy")