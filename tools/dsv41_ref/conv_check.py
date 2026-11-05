import sys

import mlx.core as mx
import numpy as np

sys.path.insert(0, "/tmp")
GGUF_DIR = sys.argv[1]
sys.argv = [sys.argv[0]]
import conv2_dsv41 as cv  # noqa: E402  (复用它的 GGUF 读取与 Q5_K 反量化)

W = mx.load(GGUF_DIR + "/mlx-dense/model.safetensors")
sh = cv.Shards(cv.SHARDS)
for name in ["blk.1.ffn_gate_shexp.weight", "blk.1.ffn_up_shexp.weight", "blk.1.ffn_down_shexp.weight"]:
    orig = sh.read(name)  # GGUF Q5_K -> f32 [ne1, ne0]
    k = name
    ours = np.array(mx.dequantize(W[k], W[k + ".scales"], W[k + ".biases"], 64, 4).astype(mx.float32))
    print(f"{name}: gguf{orig.shape} ours{ours.shape}")
    if orig.shape != ours.shape:
        print("  SHAPE MISMATCH!")
        continue
    d = np.abs(orig - ours)
    rel = d / (np.abs(orig) + 1e-9)
    print(f"  max|diff|={d.max():.6g}  mean|diff|={d.mean():.6g}  rel_p99={np.percentile(rel, 99):.4g}  corr={np.corrcoef(orig.ravel()[:100000], ours.ravel()[:100000])[0,1]:.6f}")
    # 抽一行看首 8 个值
    r = 5
    print(f"  row{r} gguf[:8]={np.round(orig[r, :8], 4)}")
    print(f"  row{r} ours[:8]={np.round(ours[r, :8], 4)}")
