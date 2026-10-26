# NOTES_s0 — 一键脚本与基线数字

日期：2026-10-26 ｜ 机器：Windows，RTX 2060 6GB（nvidia-smi 实测 `memory.total = 6144 MiB`，CUDA 13.3，sm_75）

## 基线数字（实测，每项跑 2 次取较好值）

固定参数：CPU `--max-new 64`；CUDA 单流 prompt=`The meaning of life is`、40 tok、greedy(0)；批处理取 cuda_test `[batch]` 行。

| 栈 | 精度 | 模型 | 场景 | tok/s | 显存峰值 MB | 备注 |
|----|------|------|------|-------|-------------|------|
| cpu | fp32 | tinyllama | single | 2.7~2.8 | - | 64 tok / ~23.6s |
| cpu | w8 | tinyllama | single | 3.4 | - | W8A16 + int8 KV，64 tok / ~19.0s（vs fp32 ~1.25x） |
| cpu | w8 | qwen05 | single | 4.7 | - | W8A16 + int8 KV，64 tok / ~13.8s |
| cuda | fp32 | qwen05 | single | 28.2~28.7（采样）/ 48.7（无采样直跑） | 3292 | bench 脚本在运行期间每 120ms 调 nvidia-smi，对单流 decode 有可测干扰（48.7→28）；批处理场景不受此影响 |
| cuda | fp32 | qwen05 | batch | 333~338 | - | cuda_test [batch] B=4 × 24 steps aggregate |
| cuda | fp16 | qwen05 | single | TODO(待 t1) | - | 设 `SGL_CUDA_FP16=1` 重跑 bench_all.ps1 出数 |
| cuda | int8 | qwen05 | single | TODO(待 t8) | - | 设 `SGL_CUDA_W8=1` 重跑 bench_all.ps1 出数 |

VRAM 峰值 3292 MB 是 nvidia-smi `memory.used` 采样最大值（含桌面占用 ~5.1GB 中的未释放部分；上传后 free 2655 MB），6GB 卡跑 fp32 qwen05（权重 942 MB + KV 96 MB + 工作区）没有压力，但再大的模型（如 1.1B bf16 双份）就会接近上限——t1 做 fp16 时可顺带省一半权重显存。

## GPU 实况说明

- 显卡：NVIDIA GeForce RTX 2060，**6144 MiB（6GB）**，不是部分旧记录里写的 4GB（nvidia-smi 实测确认）。
- CUDA Toolkit 13.3，sm_75；`VRAM before upload ≈ 5113 MB`（桌面+系统占用），上传 qwen05 后 free 2655 MB。
- 规划后续步骤时按 6GB 预算：fp32 下 qwen05(0.5B) 可用，tinyllama(1.1B f32 约 2GB 权重+KV) 需 w8 或 fp16。

## 脚本用法

### build_all.bat（一键构建+测试，位于 sglang-c/ 根）

```bat
:: 在任意目录（sglang/ 根推荐）
sglang-c\build_all.bat
```

步骤：① MinGW 构建 build → ② MSVC 构建 build-cuda（cuda_test cuda_cli cuda_http）→ ③ ctest --test-dir sglang-c/build --output-on-failure → ④ cuda_test models/qwen05。任一步失败打印明确错误并非零退出；全部通过打印 `[build_all] ALL OK`，退出码 0。已验证：连续两遍全绿（ctest 100% 15/15，cuda_test all OK）。

### bench_all.ps1（性能快照，位于 sglang-c/ 根）

```powershell
powershell -ExecutionPolicy Bypass -File sglang-c\bench_all.ps1
# 出 cuda fp16 / int8 数字（t1 / t8 完成后）：
$env:SGL_CUDA_FP16 = "1"; powershell -ExecutionPolicy Bypass -File sglang-c\bench_all.ps1
$env:SGL_CUDA_W8   = "1"; powershell -ExecutionPolicy Bypass -File sglang-c\bench_all.ps1
```

- Markdown 表打印到 stdout，并整表重写 `sglang-c/docs/BENCH.md`（幂等，可重复执行）。
- 显存峰值：cuda 行运行期间 nvidia-smi 每 120ms 采样取最大；cpu 行不采样（`-`）。
- 批处理行引用 cuda_test 输出的 `[batch] ... tok/s aggregate`。
- `--soak` 模式本版未实现（t11 加）。
- 已知现象：nvidia-smi 采样会拖慢单流 decode（见上表备注）；若要对比论文级数字可临时注释采样循环。

## 铁律遵守情况

本步只新增 sglang-c/build_all.bat、sglang-c/bench_all.ps1、sglang-c/docs/BENCH.md、sglang-c/docs/NOTES_s0.md，未改任何 C 代码；两个脚本均可重复执行（已各验证 ≥2 次）。
