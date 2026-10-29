"""派生一个「纯 MoE、无 hybrid」的小夹具，用来精确对拍 GPU MoE 前向。

qwen3next-rand 的 layer_types 是 [linear_attention, full_attention, ...]（hybrid），
GPU 的 lin 层必须走 SGL_CUDA_W8=1（int8），会带来 ~6e-3 的量化误差，掩盖 MoE 本身的误差。
这里把 layer_types 全改成 full_attention，得到同一套权重下的 dense-attention + MoE 模型：
GPU 就能走 f32 路径，和 CPU 的 moe_row() 逐张量对拍。

用法：python tests/make_moe_fixture.py
"""
import json
import os
import shutil

SRC = os.path.join(os.path.dirname(__file__), "..", "..", "models", "qwen3next-rand")
DST = os.path.join(os.path.dirname(__file__), "..", "..", "models", "qwen3moe-rand")


def main():
    if not os.path.isdir(SRC):
        print("skip: %s missing" % SRC)
        return 0
    if os.path.isdir(DST):
        shutil.rmtree(DST)
    os.makedirs(DST)
    for name in ("model.safetensors", "tokenizer.json"):
        p = os.path.join(SRC, name)
        if os.path.isfile(p):
            shutil.copy2(p, os.path.join(DST, name))
    cfg = json.load(open(os.path.join(SRC, "config.json"), encoding="utf-8"))
    L = cfg.get("num_hidden_layers", 4)
    cfg["layer_types"] = ["full_attention"] * L
    # 去掉 hybrid 专属字段：避免 llama.c 去加载 lin 权重
    for k in ("linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim",
              "linear_value_head_dim", "linear_conv_kernel_dim", "full_attention_interval",
              "linear_attn_config", "hybrid_override_pattern"):
        cfg.pop(k, None)
    json.dump(cfg, open(os.path.join(DST, "config.json"), "w", encoding="utf-8"), indent=2)
    print("wrote %s (layer_types=full x%d, experts=%s topk=%s)" %
          (DST, L, cfg.get("num_experts"), cfg.get("num_experts_per_tok")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())