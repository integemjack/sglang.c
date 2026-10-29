# MLX 后端：进度与验收记录

> 对应 [PLAN_mlx_backend.md](PLAN_mlx_backend.md) 的分期。性能数据与定位过程见
> [NOTES_mlx_t2.md](NOTES_mlx_t2.md)（fp16）与 [NOTES_mlx_q4.md](NOTES_mlx_q4.md)（4-bit）。

## T1 —— llama 系 fp16 + 服务（✅ 2026-10-29 完成，2026-11-04 复核）

### 交付物

| 文件 | 作用 |
|---|---|
| `include/sgl/mlx.h` | 与 `cuda.h` 同形状的后端接口（prefill 分段 / 批量 decode / copy_kv / 内存估算） |
| `src/mlx_backend.cpp` | ~950 行 C++20，llama 系 dense 前向走 Metal |
| `apps/mlx_http.c` | 薄壳，把入口装进 `sgl_accel_api` 表 |
| `tests/mlx_test.c` | 对拍 CPU 金标准（13 项检查）+ 性能 |
| `CMakeLists.txt` 末尾 `SGL_MLX` 块 | 产出 `mlx_test` / `mlx_http`，吃 pip wheel 的 MLX |
| `tools/mlx_smoke/` | API 语义探针（smoke / probe2 / probe3 / probe4） |

### 覆盖范围

- llama / Qwen2 / Qwen2.5 / Mistral / Gemma / Gemma-2 系 dense 层
- GQA、qk-norm、q_gate、partial rotary、逐层不同 RoPE 表、sliding window、
  逐层窗口（gemma-2 的 local/global 交替）、gemma 的 softcap + sqrt(E)、`norm_p1`
- 分段 prefill（与 `accel_http` 的 seg 语义对接）、批量 decode（段隔离）、
  `copy_kv`（前缀缓存搬运）
- **明确拒绝**并说明原因：MoE 层、MLA（DeepSeek 系）、hybrid linear attention
  （Qwen3-Next）—— 这些在 T2/T3

### 验收（iMac M1 真机）

| 模型 | 结果 |
|---|---|
| models/tiny-llama-hf | 12/12 |
| models/gemma-rand | 13/13 |
| models/gemma2-rand | 13/13 |
| models/mistral-rand | 13/13 |
| models/mistral-win4（自造的 window=4） | 13/13 |
| **Qwen2.5-0.5B-Instruct fp16** | 12/12 |
| **Qwen2.5-0.5B-Instruct fp32** | 13/13 |

- 真模型 fp16：对 CPU 概率误差 1.16e-3，16 步 greedy top-1 全一致
- 真模型 fp32：跨路径概率误差 8.9e-7（逻辑逐位对）
- HTTP：`/health` `/v1/models` `/v1/chat/completions` 全部通过，出正确中文

> `mistral-win4` 是特意造的：原 fixture 的 `sliding_window=64`，8~16 token 的测试
> 根本触发不到窗口裁剪，改成 4 才真的走到那条分支。

## T2（性能部分）—— ✅ 2026-11-04 完成

| 优化 | decode B=1 | 累计提速 |
|---|---|---|
| T1 完成时 | 69.0 ms/步 · 14.5 tok/s | 1.00x |
| `mx::fast::rope` 快路径（可判定时） | 62.7 ms/步 · 15.9 tok/s | 1.10x |
| q/k/v 与 gate/up 权重合并成一次 matmul | 45.4 ms/步 · 22.0 tok/s | 1.52x |
| 去掉每层 `krows.eval()`（关键） | **23.2 ms/步 · 43.2 tok/s** | **2.98x** |
| 官方 mlx_lm 同机同模型（基准） | 23.89 ms/步 · 41.9 tok/s | — |

批量 decode B=4：33.5 ms/步 · 119.6 tok/s 合计。
HTTP 端到端 290 token：37.8 tok/s（decode 23.5 ms + 调度 gap 3.3 ms）。
并发 4 条各 64 token：0.86 s 完成 ≈ 297 tok/s 合计。

**结论：比官方 mlx_lm 略快。**

## T2 量化（4-bit + 8-bit）—— ✅ 2026-11-04 完成

细节见 [NOTES_mlx_q4.md](NOTES_mlx_q4.md)。

| 项 | 结果 |
|---|---|
| 检查点 | `mlx-community/Qwen2.5-0.5B-Instruct-4bit`（磁盘 276 MB，fp16 是 951 MB）及 **8-bit**（525 MB） |
| logits vs `mlx_lm` | 两种位宽都与官方 2D 路径**逐位一致**（4-bit top10 14.2578/11.2578/…；8-bit 含 lg[198]=11.46094） |
| 4-bit decode | 单条 **75.2 tok/s**（bench），官方 80.7，差 6%；并发 4 条 **111 tok/s** 合计 |
| 8-bit decode | **58.3 tok/s**，官方 65.3 |
| 加载 | 0.4 s（config-only + 惰性 safetensors） |
| fp16 回归 | 6 个模型 12~13 项全绿 |

8-bit 的四个坑（bits 推断漏除 group / `cat2` 丢量化参数 / `emb_lookup` biases
未按行 take / lm_head 检查漏 `lm_head_q`）全被对拍抓出，详见 NOTES §5。
另发现 **mlx_lm 自身 1D/2D 输入数值路径不同**（batch=1 语义上应一致），
对拍脚本统一 2D。

两个关键优化：
1. **lm_head 保留量化形式**（47.5 → 76.9 tok/s）：tie 时不把 wte 解量化成
   fp16 `[V,E]`（272 MB）再逐 token 扫，而是另存量化三件套（87 MB）。
2. **并发时紧凑化活跃 KV 行**（seg=4096 下 25.9 → 111 tok/s）：原本 `Smax`
   跟着段的**跨度**走，4 条序列分乘在 4 个 4096 段里就要扫 12416 行；
   改成 `take` gather 成 `[B, maxlen, KVH, D]`。

生成质量（4-bit）：
- 「什么是编译器」→「编译器是一种程序，用于将源代码转换为机器代码……」
- 「量子纠缠」→ 正确的中文解释
- 「What is the capital of France?」→「The capital of France is Paris.」

## T2 剩余（进行中）

- **多分片**：7B-4bit 下载验收中
- **MXFP4**：代码支持 + 单测（无现成小检查点）

## 还没做

### T2 剩余
- **现有 AWQ/GPTQ → MLX 量化**（现在只读 mlx-community 的格式）
- **前缀缓存**（`copy_kv` 已通，但 radix/前缀命中还没接进 MLX 路径）、
  多模态 emb（接口已有 `emb` 参数）、gpt2 后端（`from_gpt2`）
- **embed 的 `gather_qmm`**：现在 wte 存的是解量化的 fp16，7B 上会到 1 GB

### T3
- GDN（`mx::fast::metal_kernel` 自定义）、MLA（DeepSeek 系）、MTP 投机解码
- `mx::compile` 进一步调优（现在靠「一次 eval」拿到了大部分收益，
  再往下要重构 decode 成纯函数把 KV 池当输入输出传）

## 环境备忘（复现用）

```bash
# iMac 上
export PATH=/opt/homebrew/bin:$PATH            # cmake 在这里
MLXROOT=~/tools/venv/lib/python3.9/site-packages/mlx   # pip wheel 自带 C++ 开发包
cd ~/v2t/sglang-c && tar xzf ../sgl-src.tgz
cmake -S . -B build-mlx -G Ninja -DSGL_MLX=ON -DSGL_MLX_ROOT=$MLXROOT
cmake --build build-mlx --target mlx_test mlx_http -j 8

SGL_MLX_CTX=4096 ./build-mlx/bin/mlx_test "/Volumes/9100 PRO/models/qwen25-05b" 16
./build-mlx/bin/mlx_q4_test "/Volumes/9100 PRO/models/qwen25-05b-4bit" 9707   # logits 对拍
./build-mlx/bin/sglc --model "/Volumes/9100 PRO/models/qwen25-05b-4bit" --serve --device mlx --port 30001
```

诊断开关：`SGL_MLX_PROF=1`（每步 B/Smax/耗时）、`SGL_MLX_DBG=1`（权重加载）、
`SGL_MLX_COMPACT=0`（关紧凑化）、`SGL_MLX_SDPA=1`（改用 `fast::sdpa`，实测更慢）、
`SGL_MLX_BUCKET=N`（Smax 对齐档位，默认 32）。

本地（Windows）同步+构建：`pwsh -File build/imac_sync_build.ps1 mlx_test mlx_http`
（tar → scp → 远端 cmake --build）。iMac 内置盘只剩几 GB，源码/构建/模型都放
`/Volumes/9100 PRO`（经 `~/v2t` 符号链接访问，避开路径里的空格）。