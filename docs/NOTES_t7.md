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

## G7 补记：GPU 侧实现（2026-10-28，src/cuda_spec.c + sgl_cuda_forward_seg_rows）

- 后端新增多行 logits 出口 `sgl_cuda_forward_seg_rows`（lm_head 一张 M=n GEMM）；
  ensure_lgb 增长路径补 invalidate_graphs（图里绑的是 h_lgb/d_lgb 裸指针，铁律 4）。
- 正确性（cuda_test [spec]，全绿）：
  - qwen05 draft==target（同实例双 KV 段 0/8192）：接受率 100% (20/20)，24 tok 与非 spec greedy 全同；
  - tinyllama(fp16)+llama-160m：接受率 13.3% (8/60)，24 tok 与非 spec greedy 全同（与 CPU 版 13% 一致）。
- 性能（RTX 2060，free VRAM 2881MB，prompt='The meaning of life is'，new=32，best-of-3，
  非 spec 基线 = prefill + decode_batch B=1 图步）：
  - plain 80.3-82.9 tok/s；spec K=1 35.1 (0.44x, acc 39.1%) / K=2 28.6 (0.35x, acc 28.6%) /
    K=4 22.0 (0.27x, acc 14.8%)——**仍净减速**。
  - 成本实测分解（K=4）：每轮 ≈ 135ms（修补齐前）：4 次 draft 图步 ~20ms + verify n=4
    plain 路径 ~30ms + reject/bonus 补齐各 1 次 plain forward_seg ~45ms。补齐改走
    decode_batch 图步后降到 ~97ms/轮（11.9→22.0 tok/s）。
  - 根因：① 接受率上限（未蒸馏同源小模型 13-15%，每轮只产 1.6 tok）；② verify/catchup
    的 plain 路径在 WDDM 桌面 GPU 下不能图化（n 可变），固定吃 2-3 倍图步成本。
    正收益条件：接受率≥ 50%（同源蒸馏 draft）或 verify 图化（固定 n）+ 批量 spec。
- temperature>0 的修正采样（speculative sampling）GPU/CPU 均未实现，只支持 greedy。
