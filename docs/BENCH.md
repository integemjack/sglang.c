# sglang-c 性能快照 (BENCH)

行 schema: 栈(cpu/cuda) | 精度(fp32/fp16/w8/int8) | 模型 | 场景(single/batch) | tok/s | 显存峰值 MB | 备注

由 `sglang-c/bench_all.ps1` 维护，每次运行整表重写（幂等）。固定 prompt 与 max-new，每项跑 2 次取较好值。
CUDA fp16 / int8 行：设环境变量 `SGL_CUDA_FP16=1` / `SGL_CUDA_W8=1` 后重跑本脚本即出数。显存峰值由 nvidia-smi 每 120ms 采样（仅 cuda 行）。

## 最新快照 (2026-10-26 13:15)

| 栈 | 精度 | 模型 | 场景 | tok/s | 显存峰值 MB | 备注 |
|----|------|------|------|-------|-------------|------|
| cpu | fp32 | tinyllama | single | 2.7 | - | 64 tok / 23.82s (best of 2) |
| cpu | w8 | tinyllama | single | 3.4 | - | W8A16 + int8 KV, 64 tok / 18.89s (best of 2) |
| cpu | w8 | qwen05 | single | 4.7 | - | W8A16 + int8 KV, 64 tok / 13.76s (best of 2) |
| cuda | fp32 | qwen05 | single | 28.2 | 3292 | prompt='The meaning of life is' greedy 40 tok / 1.42s (best of 2) |
| cuda | fp16 | qwen05 | single | TODO(待 t1) | - | set SGL_CUDA_FP16=1 and rerun |
| cuda | int8 | qwen05 | single | TODO(待 t8) | - | set SGL_CUDA_W8=1 and rerun |
| cuda | fp32 | qwen05 | batch | 335.6 | - | cuda_test [batch] B=4 x 24 steps, 96 tok / 0.29s aggregate |

