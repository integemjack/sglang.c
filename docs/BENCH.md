# sglang-c 性能快照 (BENCH)

行 schema: 栈(cpu/cuda) | 精度(fp32/fp16/w8/int8) | 模型 | 场景(single/batch) | tok/s | 显存峰值 MB | 备注

由 `sglang-c/bench_all.ps1` 维护，每次运行整表重写（幂等）。固定 prompt 与 max-new，每项跑 2 次取较好值。
CUDA fp16 / int8 行：设环境变量 `SGL_CUDA_FP16=1` / `SGL_CUDA_W8=1` 后重跑本脚本即出数。显存峰值由 nvidia-smi 每 120ms 采样（仅 cuda 行）。

## 最新快照 (2026-10-26 15:13)

| 栈 | 精度 | 模型 | 场景 | tok/s | 显存峰值 MB | 备注 |
|----|------|------|------|-------|-------------|------|
| cpu | fp32 | tinyllama | single | 4.1 | - | 64 tok / 15.48s (best of 2) |
| cpu | w8 | tinyllama | single | 3.2 | - | W8A16 + int8 KV, 64 tok / 20.12s (best of 2) |
| cpu | w8 | qwen05 | single | 4.5 | - | W8A16 + int8 KV, 64 tok / 14.29s (best of 2) |
| cuda | fp32 | qwen05 | single | 32.1 | 3212 | prompt='The meaning of life is' greedy 40 tok / 1.25s (best of 2) |
| cuda | fp16 | qwen05 | single | 34.8 | 3212 | SGL_CUDA_FP16=1, 40 tok / 1.15s (best of 2) |
| cuda | fp16 | tinyllama | single | 42.3 | 3940 | 1.1B fp16 ~2.2GB fits 6GB card, 40 tok / 0.95s (best of 2) |
| cuda | int8 | qwen05 | single | TODO(待 t8) | - | set SGL_CUDA_W8=1 and rerun |
| cuda | fp32 | qwen05 | batch | 601.5 | - | cuda_test [batch] B=4 x 24 steps, 96 tok / 0.16s aggregate |

