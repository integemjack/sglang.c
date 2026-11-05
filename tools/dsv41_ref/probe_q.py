import mlx.core as mx

w = mx.random.normal((8, 128))
wq, s, b = mx.quantize(w, group_size=64, bits=4)
print("types:", wq.dtype, s.dtype, b.dtype, wq.shape, s.shape)
d = mx.dequantize(wq, s, b, 64, 4)
print("roundtrip maxerr:", mx.abs(d - w).max())