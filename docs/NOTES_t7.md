# NOTES_t7 — Speculative Decoding v2 收尾验收

日期：2026-10-27 ｜ models/tinyllama（target 1.1B）+ models/llama-160m（draft，同 Llama-2 词表）

## 验收结果（SGL_DRAFT_DIR=models/llama-160m test_spec）

- **接受率 13%**（drafted=60 accepted=8）——与 M18/M19 结论一致（未蒸馏同源小模型的上限）
- **数学保证保持**：spec greedy == target-only greedy 逐 token ✓（test_spec 断言）
- **端到端（CPU）**：spec 1.0 tok/s vs target-only 3.6 tok/s——**净减速**

## 结论

- v2 的 n-outer 批量 verify（sgl_llama_forward_n，权重驻留、K 候选一次过）已落地：
  verify 成本 ≈ 单步×1.2 而非 ×K（spec.c 调用，test_spec 通过）。
- CPU 的负收益瓶颈在 **draft 的 K 次串行前向**（160m 每步 ~0.1s），不在 verify。
  spec 的正收益场景是 **GPU**（draft 前向 <1ms 时接受率 13% 即可回本）——
  留作 CUDA spec 的后续工作（TODO_PROMPTS 任务 7 的 GPU 部分未在本轮展开）。
- 修复记录：v2 初版 kv_catch_up 把 1 行 logits 缓冲传给 forward_n 的 n>1 prefill
  导致堆溢出（已修：借 n 行临时缓冲末行拷回）；test_llama 增加
  decode_batch==forward 断言防回归。
