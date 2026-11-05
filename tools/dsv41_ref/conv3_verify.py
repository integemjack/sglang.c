import gc
import os

import mlx.core as mx

p = "./models/dsv41/mlx-dense"
os.replace(p + "/model.safetensors.new.safetensors", p + "/model.safetensors")
print("renamed, size:", os.path.getsize(p + "/model.safetensors"))
gc.collect()
W = mx.load(p + "/model.safetensors")
k = "blk.1.ffn_gate_shexp.weight"
dq = mx.dequantize(W[k], W[k + ".scales"], W[k + ".biases"], 64, 4).astype(mx.float32)
mean = float(mx.mean(dq))
rms = float(mx.sqrt(mx.mean(dq * dq)))
mxmax = float(mx.max(mx.abs(dq)))
print(f"VERIFY blk.1 shexp: mean={mean:+.6f} rms={rms:.6f} max={mxmax:.6f} (GGUF mean=+0.000001 rms=0.019431 max=1.25064)")
ok = abs(mean) < 5e-4 and abs(rms - 0.019431) < 3e-3
print("VERIFY_RESULT:", "PASS" if ok else "FAIL")
