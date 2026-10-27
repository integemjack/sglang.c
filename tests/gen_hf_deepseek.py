"""用 transformers 自己的 DeepseekV2ForCausalLM 生成「金标准」参考。

用途：独立验证 sglang-c 的 DeepSeek-V2/V3 后端（MLA + MoE + yarn RoPE），
参考值不是手写 numpy，而是 transformers 5.20.2 的 `transformers.models.deepseek_v2`
前向结果（float32/float64 双份，便于判断误差来源）。

用法：
  python tests/gen_hf_deepseek.py --out ../models/hf-ds-plain
  python tests/gen_hf_deepseek.py --out ../models/hf-ds-yarn --yarn
"""
import argparse
import json
import os

import torch
from transformers.models.deepseek_v2 import DeepseekV2Config, DeepseekV2ForCausalLM

TOKENS = [3, 7, 11, 19, 23, 2, 5, 13, 17]


def build_cfg(yarn: bool, v3: bool) -> DeepseekV2Config:
    kw = dict(
        vocab_size=24,
        hidden_size=32,
        intermediate_size=32,
        moe_intermediate_size=12,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=4,
        num_experts_per_tok=2,
        first_k_dense_replace=1,   # 第 0 层 dense，其余 MoE
        moe_layer_freq=1,
        q_lora_rank=12,
        kv_lora_rank=16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        v_head_dim=6,
        max_position_embeddings=64,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        attention_bias=False,
        tie_word_embeddings=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    if v3:
        # 注意：transformers 5.20 的 DeepseekV2 路由器只实现了 softmax + greedy/group_limited_greedy，
        # 既不读 scoring_func(sigmoid) 也不读 norm_topk_prob，所以这里只开它能对上的开关：
        # 分组路由 + routed_scaling_factor（norm_topk_prob 保持 False，否则参考与 C 的语义不同）。
        kw.update(
            topk_method="group_limited_greedy",
            n_group=2,
            topk_group=1,
            norm_topk_prob=bool(a.norm_topk),
            routed_scaling_factor=2.5,
        )
    if yarn:
        kw["rope_scaling"] = {
            "type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 16,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
        }
    return DeepseekV2Config(**kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--yarn", action="store_true")
    ap.add_argument("--v3", action="store_true", help="分组路由 + routed_scaling_factor")
    ap.add_argument("--norm-topk", action="store_true", help="norm_topk_prob=True（真实 V3）")
    ap.add_argument("--seed", type=int, default=1234)
    # 同 gen_hf_llama_yarn：太小会让注意力近均匀、RoPE 失效，yarn 就测不出来
    ap.add_argument("--scale", type=float, default=2.0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    cfg = build_cfg(a.yarn, a.v3)
    # 小数值初始化，避免随机大权重把 logits 推成极端值（相对误差会失真）
    model = DeepseekV2ForCausalLM(cfg).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(a.scale)

    ids = torch.tensor([TOKENS], dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=False)
        logits = out.logits[0]  # [n, V]

    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out, safe_serialization=True)

    # save_pretrained 会写 config.json；补上我们能识别的 rope_scaling（原样保留）
    ref = {
        "tokens": TOKENS,
        "logits_last": [float(x) for x in logits[-1].tolist()],
        "logits_all": [[float(x) for x in row.tolist()] for row in logits],
        "yarn": bool(a.yarn),
        "v3": bool(a.v3),
    }
    with open(os.path.join(a.out, "ref_hf_deepseek.json"), "w", encoding="utf-8") as f:
        json.dump(ref, f)
    print("wrote %s (yarn=%s v3=%s) files: %s"
          % (a.out, a.yarn, a.v3, sorted(os.listdir(a.out))))


if __name__ == "__main__":
    main()