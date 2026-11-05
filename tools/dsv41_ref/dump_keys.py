import json, struct

f = open('/Volumes/9100 PRO/models/dsv41/mlx-dense/model.safetensors', 'rb')
n = struct.unpack('<Q', f.read(8))[0]
h = json.loads(f.read(n))
for k in sorted(h):
    if k.startswith('blk.0.') and any(s in k for s in ('attn_kv', 'attn_q_a', 'attn_q_b', 'attn_output_b', 'hc_attn', 'attn_norm', 'ffn_gate_inp', 'attn_sinks')):
        print(k, h[k]['dtype'], h[k]['shape'])