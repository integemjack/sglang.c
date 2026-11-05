import json

c = json.load(open('/Volumes/9100 PRO/models/dsv41/mlx-dense/config.json'))
for k in sorted(c):
    v = c[k]
    if isinstance(v, list) and len(v) > 20:
        v = v[:10] + ['...'] + [v[-3]]
    print(k, '=', v)