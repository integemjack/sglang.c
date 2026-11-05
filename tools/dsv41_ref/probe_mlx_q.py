import mlx.core as mx
import numpy as np

w = mx.array(np.array([[0.13, -0.72, 0.55, 1.37, -2.11, 0.04, 0.99, -0.48,
                        0.21, -0.13, 0.7, -1.9, 0.33, 0.88, -0.27, 0.05,
                        1.1, -0.9, 0.4, 0.06, -0.66, 2.2, -1.3, 0.5,
                        0.09, 0.71, -0.34, 1.85, -0.58, 0.22, -1.1, 0.63]], dtype=np.float32))
qw, scales, biases = mx.quantize(w, group_size=32, bits=2)
print("q:", qw.tolist())
print("scales:", scales.tolist())
print("biases:", biases.tolist())
dw = mx.dequantize(qw, scales, biases, group_size=4, bits=2)
print("deq:", dw.tolist())
# 手工反推：q 是 u32 打包；试 bits=2: 每 u32 装 16 个值
q_np = np.array(qw.tolist(), dtype=np.uint32).reshape(-1)
vals = []
for u in q_np:
    for k in range(16):
        vals.append(int((u >> (2 * k)) & 3))
print("unpack:", vals)
s = np.array(scales.tolist(), dtype=np.float32).reshape(-1)
b = np.array(biases.tolist(), dtype=np.float32).reshape(-1)
for i, q in enumerate(vals):
    g = i // 32
    print(f"q={q} s={s[g]:.4f} b={b[g]:.4f} -> s*(q-b)={s[g]*(q-b[g]):+.4f} s*q+b={s[g]*q+b[g]:+.4f}")
print("orig:", w.tolist())
