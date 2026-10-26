# sglang-c：SGLang 的 C 语言重构

> 状态：M0–M19 + M9 + M21 + M26 已完成（2026-02）；**CPU 15/15 + CUDA 全绿**；
> M26 长上下文 + CUDA Graph：两段式 attention 解除 4096 钳制（SGL_CUDA_CTX 默认
> 16384，tied lm_head 省 545MB，workspace 按需），4300-token 批处理与单流全等、
> 两段式 vs 原 kernel maxerr 2.2e-5；decode CUDA Graph（capture/replay，输入 H2D 入图 +
> graph-safe KV 写）4 序列 177 → 335 tok/s（+89%）输出全等；修两段式 kernel block
> 归约缺读后 barrier 的 race（racecheck 定位，偶发 NaN）；
> M9 CUDA 后端（sm_75，Qwen2.5-0.5B 实测）：prefill maxerr 3e-5（vs CPU），greedy 序列
> GPU/CPU/numpy 三方一致，纯 decode 45.7 tok/s（vs CPU 4.2 → 10.9x）； kernels：
> gemm/rmsnorm/rope/attention(GQA+shared softmax)/silu；修 3 个真 bug：KV 写入漏层偏移、
> KV 分配漏乘层数、RoPE 行号误作位置（与 CPU 侧同型 bug）；工具链：MSVC 14.44 + nvcc 13.3，
> 头文件加 BOM + /utf-8 解决 GBK 误读，MSBuild CUDA 集成不稳改自定义命令直接驱动 nvcc；
> M19 真 draft spec（JackFram/llama-160m 同词表）：接受率 13%（vs 截层 0-2%），仍净减速；
> 定位两个结构性原因：逐步验证无批收益、CPU GEMV 无权重复用（需 n-outer 分块 GEMM +
> 接受率≥50% 的同分布 draft）；sglc 加 --w8/--kv8；README.md 完成；
> M18 speculative decoding：draft-verify 框架（K≤8，逐步验证用上一位置分布比对；
> KV 回退不释放页防 double-unref）；greedy 输出与 target-only 全等（数学保证验证通过）；
> 负结果：未训练的截层自草稿接受率 0-2%（LayerSkip 需训练），净减速——接入同词表
> 专门训练的小 draft 模型（如 TinyLlama-110M）即可获加速；
> M17 stop/n；M16 beam；M15 流式；M14 SIMD；M12/13 量化。

> 目标：把 [sgl-project/sglang](https://github.com/sgl-project/sglang)（核心 Python 约 138 万行 + C++/CUDA kernel 约 16 万行）
> 逐步重构为一个纯 C（C11）+ 可选 CUDA 的高性能 LLM 推理引擎。
> 原仓库浅克隆在本目录的父目录 `../`，仅作行为参照，不改动。

## 现实评估

sglang 的 Python 层只是调度层，其下依赖 PyTorch（张量算子/内存）、FlashInfer 与 vLLM 的
attention kernel、NCCL 分布式、CUDA Graph。完整重构 = 用 C 重写「PyTorch 子集 + kernel +
调度系统」，是团队年级工程。本项目按下面的里程碑**自底向上**推进：每个里程碑都交付可编译、
可测试、可运行的 C 代码，先跑通最小闭环，再逐模块对齐 sglang 的能力。

## 里程碑

| 阶段 | 内容 | 对应 sglang 模块 | 状态 |
|---|---|---|---|
| M0 | 项目骨架、CMake、基础库：日志 / 内存 / mmap / 张量 / JSON | (基础设施，torch 提供) | ✅ |
| M1 | CPU 张量算子：分块 matmul、layernorm、softmax、GELU、embedding（OpenMP 多线程） | `srt/layers/` + torch ops 子集 | ✅ |
| M2 | Tokenizer：字节级 BPE，加载 HF `tokenizer.json`，encode/decode | `srt/tokenizer/` | ✅ |
| M3 | 权重加载：safetensors 解析（mmap 零拷贝） | `srt/model_loader/` | ✅ |
| M4 | 模型实现：GPT-2 forward（prefill + KV cache 增量 decode） | `srt/models/gpt2.py` | ✅ |
| M5 | 采样与生成循环：greedy / temperature / top-k / top-p | `srt/sampling/` | ✅ |
| M6 | CLI `sglc`：加载 GPT-2 124M 真实权重跑通文本生成 | `srt/entrypoints/engine.py` | ✅ |
| M7 | RadixAttention：radix tree 前缀复用 KV cache | `srt/mem_cache/radix_cache.py` | 🔜 |
| M8 | Scheduler：请求队列、continuous batching、抢占 | `srt/managers/scheduler.py` | |
| M9 | HTTP 服务器：`/generate`、`/v1/chat/completions`（纯 C socket） | `srt/entrypoints/http_server.py` | |
| M10 | CUDA 后端：matmul/attention kernel 走 RTX 2060 | `csrc/` | |
| M11 | 更多模型：Llama / Qwen（RMSNorm、RoPE、GQA、SwiGLU） | `srt/models/llama.py` 等 | |
| M12 | 量化权重（GGUF/fp8）、beam search、约束解码 | `srt/constrained/` 等 | |

> M7 起为后续轮次的推进目标；每轮对话推进一个或多个里程碑。

## 目录结构

```
sglang-c/
├── ROADMAP.md            # 本文件
├── CMakeLists.txt
├── include/sgl/          # 公共头文件（每模块一个 .h）
├── src/                  # 实现
├── apps/sglc.c           # CLI 入口
└── tests/                # 单元测试（ctest 注册）
```

## 构建与测试

```powershell
$env:Path = "<WinLibs>\mingw64\bin;$env:Path"   # MinGW-w64 GCC
cmake -S sglang-c -B sglang-c/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build sglang-c/build
ctest --test-dir sglang-c/build --output-on-failure
```

## 构建约定

- 语言标准 C11，编译选项 `-O2 -march=native -fopenmp -Wall -Wextra`
- CUDA（M10 起）通过 `-DSGL_CUDA=ON` 开关；nvcc 需要 MSVC 宿主，MinGW 仅编译 CPU 路径
- 不依赖任何第三方库：JSON、BPE、safetensors、HTTP 全部自实现
