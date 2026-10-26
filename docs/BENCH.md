# sglang-c 性能快照 (BENCH)

行 schema: 栈(cpu/cuda) | 精度(fp32/fp16/w8/int8) | 模型 | 场景(single/batch) | tok/s | 显存峰值 MB | 备注

由 `sglang-c/bench_all.ps1` 维护，每次运行整表重写（幂等）。固定 prompt 与 max-new，每项跑 2 次取较好值。
CUDA fp16 / int8 行：设环境变量 `SGL_CUDA_FP16=1` / `SGL_CUDA_W8=1` 后重跑本脚本即出数。显存峰值由 nvidia-smi 每 120ms 采样（仅 cuda 行）。

## 最新快照 (2026-10-27 终版，三轮实测取优：fp32 基线 / SGL_CUDA_FP16=1 / SGL_CUDA_W8=1；同 prompt 同参数 greedy 40 tok best of 2，GPU 空载时测)

CUDA 三精度：qwen05 单流 fp32 33.5 / fp16 36.3 / int8(W8A8 dp4a) 33.8 tok/s（decode 带宽瓶颈下持平）；
显存（cuda_cli 上传后 free VRAM）：fp32 剩 216MB / fp16 剩 3611MB / **int8 剩 3947MB**（最省：投影 0.55GB + wte fp16）；
tinyllama-1.1B fp16 39.4 tok/s（~2.2GB 权重装入 6GB 卡；GPU 被后台服务占用时会降到 ~28）。
W8 数值验收：16 步 top1 与 fp32 全同、gap 偏差±0.1 内无翻转（NOTES_t8.md）。
int8 IMMA vs dp4a 对比见 NOTES_t8.md 补记（M=1 持平选 dp4a；M>1 IMMA 胜出待切）。

| 栈 | 精度 | 模型 | 场景 | tok/s | 显存峰值 MB | 备注 |
|----|------|------|------|-------|-------------|------|
| cpu | fp32 | tinyllama | single | 4.1 | - | 64 tok / 15.71s (best of 2) |
| cpu | w8 | tinyllama | single | 3.3 | - | W8A16 + int8 KV, 64 tok / 19.58s (best of 2) |
| cpu | w8 | qwen05 | single | 4.5 | - | W8A16 + int8 KV, 64 tok / 14.24s (best of 2) |
| cuda | fp32 | qwen05 | single | 33.5 | 4073 | prompt='The meaning of life is' greedy 40 tok / 1.2s (best of 2) |
| cuda | fp16 | qwen05 | single | 36.3 | 4073 | SGL_CUDA_FP16=1, 40 tok / 1.1s (best of 2) |
| cuda | fp16 | tinyllama | single | 39.4 | 5137 | 1.1B fp16 ~2.2GB fits 6GB card, 40 tok / 1.02s (best of 2) |
| cuda | int8 | qwen05 | single | 33.8 | 4073 | SGL_CUDA_W8=1 (W8A8 dp4a), 40 tok / 1.18s; free VRAM 3947MB（三档最优） |
| cuda | fp32 | qwen05 | batch | 540.8 | - | cuda_test [batch] B=4 x 24 steps, 96 tok / 0.18s aggregate（graph ON 时 ~330，见 t3） |
| cpu | fp32 | tinyllama | single | 4.1 | - | 64 tok / 15.71s (best of 2) |
| cpu | w8 | tinyllama | single | 3.3 | - | W8A16 + int8 KV, 64 tok / 19.58s (best of 2) |
| cpu | w8 | qwen05 | single | 4.5 | - | W8A16 + int8 KV, 64 tok / 14.24s (best of 2) |
| cuda | fp32 | qwen05 | single | 33.5 | 4073 | prompt='The meaning of life is' greedy 40 tok / 1.2s (best of 2) |
| cuda | fp16 | qwen05 | single | 36.3 | 4073 | SGL_CUDA_FP16=1, 40 tok / 1.1s (best of 2) |
| cuda | fp16 | tinyllama | single | 39.4 | 5137 | 1.1B fp16 ~2.2GB fits 6GB card, 40 tok / 1.02s (best of 2) |
| cuda | int8 | qwen05 | single | 33.8 | 4073 | SGL_CUDA_W8=1 (W8A8 dp4a), 40 tok / 1.18s; free VRAM 3947MB（三档最优） |
| cuda | fp32 | qwen05 | batch | 540.8 | - | cuda_test [batch] B=4 x 24 steps, 96 tok / 0.18s aggregate（graph ON 时 ~330，见 t3） |

