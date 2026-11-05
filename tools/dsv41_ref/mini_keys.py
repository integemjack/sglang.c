import mlx.core as mx

W = mx.load("/tmp/mini_ref.safetensors")
print(f"键数 {len(W)}")
for k in sorted(W):
    a = W[k]
    print(f"{k:50s} {str(a.dtype):8s} {list(a.shape)}")
