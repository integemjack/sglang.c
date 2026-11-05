import mlx.core as mx
import sys

d = sys.argv[1] if len(sys.argv) > 1 else "./models/dsv41/mlx-dense"
W = mx.load(d + "/model.safetensors")
k = "blk.1.ffn_gate_shexp.weight"
w, s, b = W[k], W[k + ".scales"], W[k + ".biases"]
print("w", w.shape, w.dtype, "s", s.shape, s.dtype, "b", b.shape, b.dtype)
for gs in (64,):
    dq = mx.dequantize(w, s, b, gs, 4)
    fin = mx.isfinite(dq)
    nbad = int(dq.size - mx.sum(fin))
    # 非有限元素的位置
    if nbad:
        idx = mx.argwhere(~fin)[:5]
        print(f"gs={gs}: nbad={nbad}/{dq.size} first_bad_idx={idx.tolist()}")
    print(
        f"gs={gs}: nbad={nbad} max={float(mx.max(dq)):.6g} "
        f"min={float(mx.min(dq)):.6g} mean={float(mx.mean(dq)):.6g}"
    )
    # 抽一个坏位置手工复算
    if nbad and gs == 64:
        import numpy as np

        r, c = idx[0].tolist()
        grp_c = (c // gs) * gs
        wp = w[r].tolist()
        sc = float(s[r, c // gs]) if s.ndim == 2 else float(s[c // gs])
        bi = float(b[r, c // gs]) if b.ndim == 2 else float(b[c // gs])
        # 第 c 个 4bit 值
        word = wp[c // 8]
        shift = (c % 8) * 4
        nib = (word >> shift) & 0xF
        print(f"  manual r={r} c={c}: nib={nib} scale={sc:.6g} bias={bi:.6g} -> {sc * nib + bi:.6g}")
        print(f"  raw word=0x{word:08x} words_around={[hex(x) for x in wp[c//8-1:c//8+2]]}")
        print(f"  scales around: {s[r, max(0,c//gs-2):c//gs+3].tolist() if s.ndim==2 else s[max(0,c//gs-2):c//gs+3].tolist()}")
        # 整行 word 是否异常（全 F？）
        rowwords = mx.array(wp)
        print(f"  row r={r}: frac_word_max={float(mx.sum(rowwords == 0xFFFFFFFF))/rowwords.size:.4g}")
# 对照一个健康张量
k2 = "blk.1.ffn_gate_inp.weight"
w2 = W[k2]
print("gate_inp", w2.shape, w2.dtype)
