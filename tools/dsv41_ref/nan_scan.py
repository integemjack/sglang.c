import struct, json

f = open('/Volumes/9100 PRO/models/dsv41/mlx-dense/model.safetensors', 'rb')
n = struct.unpack('<Q', f.read(8))[0]
h = json.loads(f.read(n))
info = h['blk.0.attn_kv.weight']
d0 = h['__metadata__'] if '__metadata__' in h else {}
print('dtype', info['dtype'], 'shape', info['shape'], 'offs', info['data_offsets'])
f.seek(info['data_offsets'][0])
nb = info['data_offsets'][1] - info['data_offsets'][0]
raw = f.read(nb)
vals = struct.unpack(f'<{nb // 2}e', raw)  # f16 le
nan_pos = [i for i, v in enumerate(vals) if v != v]
print('total', len(vals), 'nan_count', len(nan_pos), 'first_nans', nan_pos[:8])
inf_count = sum(1 for v in vals if v == float('inf') or v == float('-inf'))
print('inf_count', inf_count)
import math
mx = max(abs(v) for v in vals if not math.isnan(v))
print('max_abs_finite', mx)
if nan_pos:
    i = nan_pos[0]
    print('around first nan:', [float(v) for v in vals[max(0, i - 4):i + 4]])
    # 原始位模式
    bits = struct.unpack(f'<{nb // 2}H', raw)
    print('bits around:', [hex(b) for b in bits[max(0, i - 4):i + 4]])