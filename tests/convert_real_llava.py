"""G12: 把 HF 的 CLIP-L/14-336（pytorch_model.bin）转成本项目的
`<dir>/clip_model.safetensors`（fp32，只保留视觉塔部分），并把 llava 的
`mm_projector.bin` 转成 `projector.safetensors`（linear_1/linear_2，fp32）。

用法:
  python convert_real_llava.py --clip <clip_dir> --proj <llava_dir>

产物：
  <clip_dir>/clip_model.safetensors   （vision_model.* + visual_projection.*）
  <llava_dir>/projector.safetensors   （linear_1/linear_2）
另外把 llava config 里的 mm_vision_select_layer 写进 clip 目录的 clip_config.json
（本项目的 sgl_clip_load 会读它），方便直接用。
"""
import argparse, json, os, struct
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--clip", required=True)
ap.add_argument("--proj", required=True)
a = ap.parse_args()

def save_st(path, tensors):
    hdr, off = {}, 0
    for k, v in tensors.items():
        b = v.numpy().tobytes()
        hdr[k] = {"dtype": "F32", "shape": list(v.shape), "data_offsets": [off, off + len(b)]}
        off += len(b)
    hb = json.dumps(hdr).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb))); f.write(hb)
        for v in tensors.values():
            f.write(v.numpy().tobytes())
    print(f"wrote {path}: {len(tensors)} tensors, {off/1e6:.0f} MB")

# ---- CLIP vision tower ----
sd = torch.load(os.path.join(a.clip, "pytorch_model.bin"), map_location="cpu",
                weights_only=True)
vis = {}
for k, v in sd.items():
    if not k.startswith("vision_model."):
        continue
    vis[k] = v.float().contiguous()
print(f"CLIP vision tensors: {len(vis)}")
save_st(os.path.join(a.clip, "clip_model.safetensors"), vis)

# ---- llava mm_projector -> linear_1 / linear_2 ----
pj = torch.load(os.path.join(a.proj, "mm_projector.bin"), map_location="cpu",
                weights_only=True)
out = {}
for k, v in pj.items():
    # model.mm_projector.0.weight -> linear_1.weight（0=Linear, 1=GELU, 2=Linear）
    if k.endswith("0.weight"): out["linear_1.weight"] = v.float().contiguous()
    elif k.endswith("0.bias"): out["linear_1.bias"] = v.float().contiguous()
    elif k.endswith("2.weight"): out["linear_2.weight"] = v.float().contiguous()
    elif k.endswith("2.bias"): out["linear_2.bias"] = v.float().contiguous()
    else: print("skip", k)
save_st(os.path.join(a.proj, "projector.safetensors"), out)

# ---- 把 mm_vision_select_layer 抄进 clip 目录，sgl_clip_load 会读 ----
cfg = json.load(open(os.path.join(a.clip, "config.json")))
lcfg_path = os.path.join(a.proj, "config.json")
sel = -1
if os.path.exists(lcfg_path):
    sel = json.load(open(lcfg_path)).get("mm_vision_select_layer", -1)
vcfg = cfg.get("vision_config", {})
small = {
    "hidden_size": vcfg["hidden_size"], "num_heads": vcfg["num_attention_heads"],
    "num_hidden_layers": vcfg["num_hidden_layers"], "patch_size": vcfg["patch_size"],
    "image_size": vcfg["image_size"], "intermediate_size": vcfg["intermediate_size"],
    "projection_dim": vcfg["projection_dim"], "layer_norm_eps": vcfg["layer_norm_eps"],
    "hidden_act": vcfg["hidden_act"],
    "mm_vision_select_layer": sel,
}
json.dump(small, open(os.path.join(a.clip, "clip_config.json"), "w"), indent=1)
print("clip_config.json:", small)
print(f"projector: in={out['linear_1.weight'].shape[1]} mid={out['linear_1.weight'].shape[0]} "
      f"out={out['linear_2.weight'].shape[0]}")