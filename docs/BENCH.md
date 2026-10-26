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

## 缺口关闭补充（G1–G6，2026-10-27，同进程 A/B）

本卡空闲显存常小于模型常驻量，WDDM 溢出会把绝对时间放大 3–20x，所以下列都是
**同一进程内切换代码路径**的 A/B（`cuda_perf --ab` / `cuda_test` 内建断言），不是跨进程绝对值。
详见 docs/NOTES_gaps.md。

| 项 | 场景 | before | after | 倍率 | 正确性 |
|----|------|--------|-------|------|--------|
| G1 flash prefill | qwen05 8192-tok prefill | 72.4s（two-pass） | 8.7s（flash） | **8.3x** | flash vs two-pass @4300 行 maxerr=0、greedy 全同 |
| G1 flash prefill | qwen05 2048-tok | 3980.8ms | 2345.6ms | 1.7x | 同上 |
| G2 W8 IMMA | [2048,2048] M=8 投影 | 0.925ms（dp4a） | 0.069ms（IMMA） | **13.4x** | IMMA vs dp4a prefill maxerr=0、10/10 greedy 同 fp32 |
| G3 decode graph | qwen05 fp16 B=1 decode | 23.0ms/step（plain） | 9.7ms/step（graph） | 2.4x | 输出与 plain 全等 |
| G3 decode graph | qwen05 fp16 B=4 decode | 36.6ms/step | 8.9ms/step | 4.1x | 同上 |
| G3 decode graph | qwen05 fp16 B=8 decode | 38.5ms/step | 11.3ms/step | 3.4x | 同上 |
| G6 前缀缓存 | 756-tok 共享 system prompt 二发 | 0.79s（冷） | 0.35s（热） | **2.2x** | 输出与无缓存逐字一致 |
| G7 投机解码 | tinyllama+llama-160m K=1 | 80.3 tok/s（plain） | 35.1 tok/s（spec） | 0.44x（净减速） | spec == 非 spec greedy 逐 token 全同 |

G7 净减速的根因：未蒸馏同源 draft 接受率上限 ~13–15%（每轮只产 ~1.6 tok），且 verify/catchup
的 plain 路径在 WDDM 桌面 GPU 下不能图化。正收益需接受率 ≥50%（同源蒸馏 draft）或 verify 固定 n 图化 + 批量 spec。详见 NOTES_t7.md G7 补记。

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
| cuda | fp16 | tinyllama+llama-160m | spec single | 35.1 | ~2.9GB | G7 spec K=1 acc 39.1% vs plain 80.3 tok/s（**0.44x 净减速**）；K=2 28.6(0.35x) K=4 22.0(0.27x, acc 14.8%)；prompt='The meaning of life is' new=32 best-of-3，free VRAM 2881MB；正确性/成本分解见 NOTES_t7.md G7 补记 |
