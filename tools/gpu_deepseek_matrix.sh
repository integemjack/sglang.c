#!/bin/bash
# 在远端把 cuda_deepseek_test 跑遍所有 deepseek 小模型变体。
# 每个变体覆盖不同的路由/布局分支：
#   deepseek-rand      MLA + MoE（softmax、greedy、q_lora）
#   deepseek-rand-v3   sigmoid 打分 + group_limited + norm_topk + e_score_correction_bias
#   deepseek-rand-3d   3D 融合专家布局
#   deepseek-rand-ql0  无 q_lora（q_b_proj 直接吃 hidden）
#   deepseek-rand-il   interleaved RoPE
#   deepseek-rand-yarn YaRN（attention_factor 乘在 cos/sin 上）
set -u
cd /root/sglang-c/build || exit 1
fail=0
for m in deepseek-rand deepseek-rand-v3 deepseek-rand-3d deepseek-rand-ql0 deepseek-rand-il deepseek-rand-yarn; do
  if [ ! -f "../models/$m/model.safetensors" ]; then
    echo "=== $m === (缺模型，跳过)"
    continue
  fi
  echo "=== $m ==="
  out=$(./bin/cuda_deepseek_test "../models/$m" 2>&1)
  echo "$out" | grep -E '^\[(prefill|decode|chunk|vram)\]' || true
  if echo "$out" | grep -qE 'FAIL|SKIP'; then
    echo "$out" | grep -E 'FAIL|SKIP'
    fail=1
  fi
done
exit $fail