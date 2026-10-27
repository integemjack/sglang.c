"""生成 YaRN 逆频率表与 attention_factor 的金标准（transformers 自己的
`modeling_rope_utils._compute_yarn_parameters`），供 test_yarn 逐值比对。

用法：python tests/gen_yarn_ref.py tests/ref_yarn.json
"""
import json
import math
import sys
from types import SimpleNamespace

from transformers.modeling_rope_utils import _compute_yarn_parameters


def compute(head_dim, base, factor, orig, beta_fast=32, beta_slow=1,
            mscale=None, mscale_all_dim=None, truncate=True):
    rp = {
        "rope_theta": base,
        "rope_type": "yarn",
        "factor": factor,
        "original_max_position_embeddings": orig,
        "beta_fast": beta_fast,
        "beta_slow": beta_slow,
        "truncate": truncate,
    }
    if mscale is not None:
        rp["mscale"] = mscale
    if mscale_all_dim is not None:
        rp["mscale_all_dim"] = mscale_all_dim
    cfg = SimpleNamespace(
        head_dim=head_dim,
        hidden_size=head_dim,
        num_attention_heads=1,
        max_position_embeddings=orig * factor,
        rope_parameters=rp,
        rope_scaling=None,
        standardize_rope_params=lambda: None,
        per_layer_config={},
    )
    inv, attn = _compute_yarn_parameters(cfg)
    return [float(x) for x in inv.tolist()], float(attn)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "tests/ref_yarn.json"
    cases = [
        # 名字, head_dim(=qk_rope), base, factor, orig, beta_fast, beta_slow, mscale, mscale_all_dim, truncate
        {"name": "ds-v3", "head_dim": 64, "base": 10000.0, "factor": 40.0, "orig": 4096,
         "mscale": 1.0, "mscale_all_dim": 1.0},
        {"name": "ds-v2", "head_dim": 64, "base": 10000.0, "factor": 40.0, "orig": 4096,
         "mscale": 0.707, "mscale_all_dim": 0.707},
        {"name": "plain-factor", "head_dim": 8, "base": 10000.0, "factor": 4.0, "orig": 16},
        {"name": "tiny", "head_dim": 4, "base": 10000.0, "factor": 4.0, "orig": 16,
         "mscale": 1.0, "mscale_all_dim": 1.0},
        {"name": "no-trunc", "head_dim": 16, "base": 500000.0, "factor": 8.0, "orig": 2048,
         "beta_fast": 8, "beta_slow": 2, "truncate": False},
        {"name": "llama3", "head_dim": 128, "base": 500000.0, "factor": 8.0, "orig": 8192,
         "beta_fast": 32, "beta_slow": 1},
    ]
    ref = []
    for c in cases:
        inv, attn = compute(
            c["head_dim"], c["base"], c["factor"], c["orig"],
            c.get("beta_fast", 32), c.get("beta_slow", 1),
            c.get("mscale"), c.get("mscale_all_dim"), c.get("truncate", True),
        )
        ref.append(dict(c, inv_freq=inv, attention_factor=attn))
        print("%-12s attn=%.9f inv[0..3]=%s" % (c["name"], attn, ["%.6g" % x for x in inv[:4]]))
    with open(out, "w", encoding="utf-8") as f:
        json.dump(ref, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()