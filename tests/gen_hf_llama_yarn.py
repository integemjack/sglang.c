"""用 transformers 的 LlamaForCausalLM 生成一个 rope_type=yarn 的小模型 + 参考 logits，
验证 sglang-c 的 llama 后端 YaRN 路径（llama 家族里 HF 的 yarn 是真的生效的，
不像 deepseek_v2 会静默忽略）。

用法：python tests/gen_hf_llama_yarn.py --out ../models/hf-llama-yarn
"""
import argparse
import json
import os

import torch
from transformers import LlamaConfig, LlamaForCausalLM

TOKENS = [3, 7, 11, 19, 23, 2, 5, 13, 17]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../models/hf-llama-yarn")
    ap.add_argument("--seed", type=int, default=99)
    # 权重整体缩放的幅度：太小（1）时注意力分数近于均匀，RoPE 频率就几乎不影响输出，
    # 测不出 yarn 的差异；2.0 让 yarn/plain 的 logits 差到 1e-2 量级。
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--mscale", type=float, default=1.0, help="mscale（attention_factor 分子）")
    ap.add_argument("--mscale-all-dim", type=float, default=1.0, help="mscale_all_dim（分母）")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    cfg = LlamaConfig(
        vocab_size=24,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        rope_scaling={
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 16,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": a.mscale,
            "mscale_all_dim": a.mscale_all_dim,
            # 明确给 attention_factor 也可以；这里不写，让 HF 自己按 mscale 推（测我们的推导）
        },
        tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(cfg).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(a.scale)
        logits = model(input_ids=torch.tensor([TOKENS]), use_cache=False).logits[0]

    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out, safe_serialization=True)
    attn = None
    inv = None
    try:
        from transformers.modeling_rope_utils import _compute_yarn_parameters
        _inv, attn = _compute_yarn_parameters(model.model.rotary_emb.config)
        inv = [float(x) for x in _inv.tolist()]
        attn = float(attn)
    except Exception:
        pass
    ref = {
        "tokens": TOKENS,
        "logits_last": [float(x) for x in logits[-1].tolist()],
        "attention_factor": attn,
        "inv_freq": inv,
        "note": "注意：transformers 5.20 的 llama 前向实测不受 rope yarn 影响"
                "（同权重 yarn/plain 输出差 7.5e-08），所以 logits 只能当「非 yarn 路径」"
                "的对照；yarn 本身靠 inv_freq/attention_factor 与 gen_yarn_ref 的表比对。",
    }
    with open(os.path.join(a.out, "ref_hf_llama.json"), "w", encoding="utf-8") as f:
        json.dump(ref, f)
    print("wrote %s (attention_factor=%s) files=%s" % (a.out, attn, sorted(os.listdir(a.out))))


if __name__ == "__main__":
    main()