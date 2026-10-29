# MLX 后端：现状差距与实施计划

> 2026-10-29 盘点。结论：**MLX 目前只是测试用的数值参考，不是推理后端**；要「支持 MLX」
> 需要新增第三个加速后端。本文给出有实测依据的差距清单（W1–W8）、算子映射、分期范围（T1–T3）
> 和三个需要拍板的决策。所有「现状」都标了证据出处，所有「环境」都是当天在真机上探到的。

---

## 0. 一句话结论

代码上 MLX 后端 **0 行**（`--device mlx` 只是打一行日志然后回落 CPU）；但**可行性已经实测确认**：
iMac 上 pip 装的 MLX wheel 自带完整 C++ 开发包（`libmlx.dylib` + 头文件 + `MLXConfig.cmake`），
`mlx::core::fast` 直接给了 rms_norm / layer_norm / rope / scaled_dot_product_attention /
自定义 `metal_kernel`，所以**不需要从源码编 MLX，也不需要自己写 Metal 算子**（GDN 那种例外）。
主要工作量在「接缝」（把 CUDA 专用的 HTTP 引擎抽成后端无关）和「MLX 的惰性求值 + 不可变数组
语义」这两件事上，不在算子上。

---

## 1. 现状（证据）

| 项 | 现状 | 出处 |
|---|---|---|
| `--device mlx` | 接受这个值，但只打印「本仓库未提供 MLX 后端（只有 macOS 数值参考脚本），回落 cpu」 | `apps/sglc.c:340-341`；帮助文案 `apps/sglc.c:179-180` |
| 仓库里的 MLX 代码 | 只有 `tests/ref_llava_mlx.py`：用 `mlx.core`(Metal) 重算 LLaVA 的 CLIP+projector 参考，覆盖 `_ref_*.npy`，让 `test_llava` 变成「C vs MLX」 | `tests/ref_llava_mlx.py`、`docs/PROGRESS_g11_g19.md:221-237` |
| CI 的 `macos-arm64-mlx` | 只跑上面那个参考脚本 + CPU 版 ctest，**不编任何 MLX 后端**（`flavor: cpu-mlx`，没有 `-DSGL_MLX`） | `.github/workflows/build.yml:36,117-130,266-285` |
| CMake | 只有 `option(SGL_CUDA)`，没有 `SGL_MLX`；macOS 走的是纯 CPU 分支（AppleClang 无 `-fopenmp` → 单线程） | `CMakeLists.txt:197-198`、`README.md:120` |
| 引擎数量 | **两套**：CPU（`include/sgl/model.h` 抽象 + `scheduler.c` 811 行 + `http.c` 1400 行，分页 KV + radix + LoRA 池 + grammar + spec）；CUDA（`include/sgl/cuda.h` 29 个 `sgl_cuda_*` + `src/cuda_backend.cu` 6084 行 / 54 个 kernel + `apps/cuda_http.c` 2095 行**自带一套 HTTP**，段式 KV，47 处 `sgl_cuda_*` 调用） | `include/sgl/{model,cuda}.h`、`apps/cuda_http.c` |
| 文档口径 | 4 个文件明写「MLX 无后端」，做完必须一起改 | `README.md:245-247`、`README.zh-CN.md:98,136`、`CHANGELOG.md:189-191`、`docs/GAP_vs_upstream.md:22` |

**含义**：CUDA 后端不是「CPU 引擎换个设备层」，而是一条**平行的独立栈**（自己的 KV 语义、
自己的 HTTP server）。所以加 MLX 的第一个决策不是「怎么写 kernel」，而是「要不要先把这条
平行栈抽出接缝」——见 W1。

---

## 2. 目标机器实测环境（iMac，2026-10-29 探测，`ssh wangchao@192.168.20.14` 免密）

| 项 | 实测值 |
|---|---|
| 芯片 / 内存 | Apple M1，8 核 GPU，**16 GB 统一内存** |
| 系统 / Metal | macOS 27.0.1，Metal 4（`mx.metal.is_available()` = True，`Device(gpu, 0)`） |
| 编译器 | Apple clang 21.0.0（无 `-fopenmp`） |
| cmake / ninja | `~/tools/cmake/CMake.app/Contents/bin/cmake`、`~/tools/ninja`（1.13.1）；**无 Homebrew** |
| MLX | **0.29.3**，在 `~/tools/venv`（python 3.9）；`python -m mlx --cmake-dir` → `~/tools/venv/lib/python3.9/site-packages/mlx` |
| wheel 里的 C++ 开发包 | `lib/libmlx.dylib`（19 MB）、`lib/mlx.metallib`（105 MB，预编译 Metal kernel）、`include/mlx/*.h`（含 `fast.h`、`fast_primitives.h`）、`include/metal_cpp/`、`share/cmake/MLX/{MLXConfig,MLXConfigVersion,MLXTargets,MLXTargets-release}.cmake`，整个包 123 MB |
| dylib 依赖 | install name `@rpath/libmlx.dylib`；链 Metal / Foundation(weak) / QuartzCore / Accelerate / libc++ |
| 磁盘 | 内置盘 **只剩 3.9 GB**（99% 满）；挂载盘 **`/Volumes/9100 PRO`：1.8 TiB，967 GB 可用，wangchao 可写** → 源码/构建/模型一律放这里 |
| 已有模型 | 2T 盘上有 `Qwen3.8-27B-Uncensored-Q8_0.gguf`、`build-cache/models/gguf/Qwen3-0.6B-Q4_K_M.gguf`、`Qwen3VL-2B-Instruct-Q4_K_M.gguf`；`build-cache/models/mlx-community/` 里**全是 TTS 模型，没有 LLM** |
| 上一轮遗留 | `~/sglang-work` 是第 16 轮的源码快照（**不是 git 仓库**），需要重新同步 |

### 由环境推出的两条硬约束

1. **16 GB 统一内存**：27B（项目当前的目标模型）在这台机器上跑不了，Q8 GGUF 约 29 GB、
   Q4 约 15 GB 都超。MLX 后端的验收模型只能是 **0.5B–3B 级**（qwen05 / tinyllama-1.1B /
   Qwen3-0.6B），4-bit 下最多试 7B。这不是缺陷，是这台机器的物理上限；27B 的验收继续留在 5090。
2. **内置盘 3.9 GB**：任何构建目录、模型下载都必须落在 `/Volumes/9100 PRO`，否则一次
   `cmake --build` 就能把系统盘写满。

---

## 3. 关键可行性结论（**已在真机实测通过**，2026-10-29）

§8 的冒烟程序已在 iMac 上真编真跑，输出 `SMOKE_PASS`。以下每条都有实测数字：

1. **不需要从源码编 MLX**：pip wheel 自带 `MLXConfig.cmake`，`find_package(MLX CONFIG REQUIRED)`
   直接成功（实测：`Found MLX: .../site-packages/mlx/lib/libmlx.dylib`，
   `MLX_BUILD_METAL=ON`、`MLX_BUILD_ACCELERATE=ON`、`MLX_CXX_FLAGS=-DACCELERATE_NEW_LAPACK;-D_METAL_`，
   include 目录自动带上 `include/`、`include/metal_cpp/`、`include/mlx/backend/metal/kernels/metal_3_1`）。
   官方指南：<https://ml-explore.github.io/mlx/build/html/dev/mlx_in_cpp.html>
   **坑（已实测）**：`MLX_LIBRARIES` 只是目标名 `mlx`，不是路径 —— 拿 rpath 目录必须
   `get_target_property(... mlx IMPORTED_LOCATION)` 再取 dirname，直接对 `MLX_LIBRARIES` 做
   `get_filename_component` 会得到空串。
2. **算子基本不用自己写**：`mlx/fast.h`（0.29.3，真机 `cat` 确认）导出
   `rms_norm`、`layer_norm`、`rope`（**两个重载，其中 `const array& offset` 那个正好是
   batch decode 的逐行位置**）、`scaled_dot_product_attention(scale, mask_mode, mask_arrs, sinks)`、
   `metal_kernel`（**用源码字符串现场编 Metal kernel**）、`cuda_kernel`、`precompiled_cuda_kernel`。
   **实测精度（vs 主机参考实现）**：`fast::rms_norm` maxerr **1.19e-07**（[4,64] fp32）；
   `fast::scaled_dot_product_attention(causal)` maxerr **2.98e-08**（B=3 H=2 S=6 D=8）；
   `fast::rope` 的**数组 offset 重载与逐行标量 offset 完全相等（maxerr = 0）** ——
   这条最关键，说明 batch decode 的逐行位置可以直接用它，不必自己拼。
   fp16 路径（`astype(float16)` 后 rms_norm）跑通、无 NaN/Inf。
3. **KV cache 有官方推荐写法**：预分配 + `mx::slice_update`（或 `cache[:, off:off+1, :] = x`），
   步耗时几乎与上下文长度无关（M4 Max 实测 4096 上下文：concatenate 3.73 ms/步 vs
   预分配 0.22 ms/步）；**chunk 取 256 的倍数**
   （<https://ml-explore.github.io/mlx/build/html/usage/kv_cache.html>）。这与本仓库 CUDA 侧
   「连续 KV + 段基址」的布局天然对得上。
4. **自定义算子有退路**：GDN（qwen3-next / Qwen3.8 的线性注意力）和 MLA 在 MLX 里没有对应物，
   但 `mx::fast::metal_kernel` 允许直接从 C++ 传 Metal 源码字符串，不必新增 `.metal` 编译链路。
5. **CI 有现成落点**：`macos-arm64-mlx` job 已经在跑，且已经用 `grep -q 'MLX('` 卡住
   「runner 上 Metal 必须可用」——把它从「只跑参考脚本」升级成「`-DSGL_MLX=ON` 真编真测」
   是增量改动，不是新建 job。
6. **内存预算实测**：`mx::metal::device_info()` 在这台 M1 上给出
   `memory_size = 17179869184`（16 GB）、**`max_recommended_working_set_size = 12713115648`
   （11.84 GB）**、`max_buffer_length = 9534832640`（8.88 GB）、`architecture = applegpu_g13g`。
   → 「权重 + KV + 图缓存」的实际可用上限约 **11.8 GB**，这就是 W4/§2 里
   「能装下的最大模型」的硬判据（4-bit 下 7B 宽松、13B 边界、27B 不可能）。
   显存预检逻辑应基于这两个数，而不是 CUDA 那套 `cudaMemGetInfo`。

---

## 4. 差距清单（W1–W8）

每条按仓库惯例写：**现状证据 → 怎么做 → 验收**。

### W1 接缝：把「加速后端」从 CUDA 专用抽成后端无关（**先做，其它都依赖它**）

- **现状**：`apps/cuda_http.c`（2095 行）直接调 29 个 `sgl_cuda_*` 符号（47 处调用），
  其中真正在服务热路径上的只有约 12 个：`from_llama / forward_seg / forward_seg_rows /
  decode_batch / forward_seg_emb / copy_kv / ctx / vocab / prewarm / estimate_vram_mb /
  avail_vram_mb / set_lin_slot(s)`；剩下 13 个是 MTP/投机解码专用。
- **怎么做**（两条路，见 §9 决策 1）：
  - **A（推荐）**：新增 `include/sgl/accel.h`，定义 `sgl_accel` 函数指针表（上面 12 个必选 +
    13 个可选，NULL = 该特性关闭）；`cuda_http.c` → `accel_http.c`，只通过表调用；
    CUDA 与 MLX 各提供一张表。`sglc.c` 的 `--device` 分支变成「选表」。
  - **B（快但欠债）**：照 CUDA 的样子复制一份 `include/sgl/mlx.h` + `src/mlx_backend.cpp` +
    `apps/mlx_http.c`。首版能跑得快 1–2 天，代价是**第三套 HTTP server**（约 2000 行重复），
    以后每加一个端点要改三处。
- **验收**：走 A 时，5090 上 `cuda_test` / `cuda_qwen35_test` / `cuda_deepseek_test` 与
  HTTP 端到端（`prefix_e2e.ps1`、`stream_test.ps1`、`conc_test.ps1`）结果与重构前**逐字一致**；
  本机 CPU ctest 23/23 全绿。

### W2 构建接入（CMake + 依赖策略）

- **现状**：`CMakeLists.txt` 只有 `SGL_CUDA`；ROADMAP 明写「不依赖任何第三方库」——MLX 会是
  第一个例外，这条约定要改写并注明「仅 macOS、默认 OFF、不影响其它平台」。
- **怎么做**：
  - `option(SGL_MLX "Build MLX backend (Apple Silicon only)" OFF)`，`if(SGL_MLX AND NOT APPLE) → FATAL_ERROR`。
  - 定位 MLX：优先 `-DMLX_ROOT=...`；否则 `find_package(Python)` + `python -m mlx --cmake-dir`；
    然后 `find_package(MLX CONFIG REQUIRED)`，用 `MLX_INCLUDE_DIRS / MLX_LIBRARIES / MLX_CXX_FLAGS`。
  - 后端源文件是 **C++20**（`src/mlx_backend.cpp`，`extern "C"` 出口），和 `cuda_backend.cu`
    同一套路；`add_library(sgl_mlx STATIC ...)`，`target_link_libraries(... mlx)`。
  - **rpath**：`libmlx.dylib` 的 install name 是 `@rpath/libmlx.dylib`，必须
    `BUILD_RPATH`/`INSTALL_RPATH` 指到 wheel 的 `lib/`；`mlx.metallib` 要与 dylib 同目录（wheel 里本来就是）。
  - 链 Metal / Foundation / QuartzCore / Accelerate（`MLXTargets-release.cmake` 一般已带，实测确认）。
- **验收**：iMac 上（源码与 build 目录都在 `/Volumes/9100 PRO`）
  `cmake -G Ninja -DSGL_MLX=ON` + `cmake --build` 全绿；`otool -L sglc` 能看到 `@rpath/libmlx.dylib`；
  `./sglc --help` 正常；**同一份源码在 Windows/Linux 上 `-DSGL_MLX=OFF`（默认）构建不受影响**。

### W3 核心前向：llama 系 fp16（T1 的主体工作量）

- **现状**：CPU 侧 `src/llama.c`（1514 行）已覆盖 llama/qwen2/qwen3/mistral 滑窗/gemma 归一化；
  CUDA 侧 `sgl_cuda_from_llama` 复用同一个加载器作为权重源。MLX 侧可以照抄这个复用方式：
  **权重仍由 `llama.c` + `safetensors.c` 加载到主机内存，MLX 只负责算**（统一内存下没有 H2D 阶段）。
- **怎么做**：实现 `sgl_mlx_from_llama / forward_seg / forward_seg_rows / decode_batch / ctx /
  vocab / free`，算子按 §5 的映射表组合库函数；KV 用「预分配 + `slice_update`」，段基址语义
  与 CUDA 侧保持一致（`kv_base + p`），chunk 取 256 的倍数。
- **验收**：新增 `tests/mlx_test.c`（对标 `tests/cuda_test.c`）：
  ① 逐层/整体 prefill 与 CPU 的 maxerr（fp16 下预期 1e-3 量级，阈值需实测后定，不能沿用
  CUDA fp32 的 3e-5）；② qwen05 + tinyllama 上 greedy 生成 200 token 与 CPU **逐 token 相同**；
  ③ `decode_batch` B=1/4/8 与单流输出一致。数字写进 `docs/BENCH.md`。

### W4 量化权重（T2）

- **现状**：仓库自有 W8A8（dp4a IMMA，`k_gemm_w8a8*`）与 W4（`k_gemm_w4_*`），另有
  `src/qweight.c` 能解 AWQ/GPTQ/GGUF 到 f32。MLX 有自己的 affine 量化
  （`mx::quantize / quantized_matmul / dequantize`，group_size 32/64/128，4/8 bit），
  **位布局与仓库的 W8A8 不同**。
- **怎么做**：加载时把权重转成 MLX 量化格式（`mx::quantize`），走 `quantized_matmul`；
  不要为了复用 `SGL_CUDA_W8` 的格式而在 MLX 上手写反量化 GEMM（那样拿不到 Apple Silicon 的
  量化加速）。顺带打开一条新能力：**直接吃 mlx-community 的 4-bit 检查点**（见 §9 决策 3）。
- **验收**：同一 prompt 下 4-bit MLX vs fp16 MLX 的 greedy 输出差异如实记录；
  16 GB M1 上给出「能装下的最大模型」实测（预期 7B@4bit 边界）。

### W5 前缀缓存 / 多模态 / gpt2（T2 的其余部分）

- `copy_kv`（段级前缀复用）→ MLX 的 slice 拷贝；**验收**：同 prompt 重发的加速比与输出逐字一致
  （对标 CUDA 侧 G15 的 2.5x 判据）。
- `forward_seg_emb`（LLaVA 视觉 token 注入）→ `mx::concatenate` 后走同一前向；
  **验收**：`test_llava` 的 MLX 参考（`ref_llava_mlx.py`）与 C+MLX 后端 maxerr < 2e-4。
- gpt2 后端（`k_embed_pos / k_layernorm_b / k_gelu` + Conv1D 转置）→ `fast::layer_norm` +
  `mx::gelu_approx`；**验收**：`test_gpt2` 的 GPU 版对照。

### W6 GDN / MLA / MTP（T3，明确后置）

- GDN 的 9 个 kernel（`k_lin_conv/l2norm/ba/step/gnorm` 及 `_b` 变体、`k_lin_layer_persist`）
  与 MLA（`src/cuda_deepseek.cu` 1045 行）在 MLX 里**没有对应算子**，只能用
  `mx::fast::metal_kernel` 写自定义 Metal 源码，或先按「该模型族在 MLX 后端不支持」明确报错。
- MTP 投机解码（13 个 `sgl_cuda_mtp_*` / `spec_*`）依赖逐行状态快照与回滚，MLX 的不可变数组
  语义下要重新设计（快照 = 保留旧 array 引用，反而可能更简单，但需要实测）。
- **验收**：T3 之前，`--device mlx` 遇到 GDN/MLA 模型必须**明确报错并说明原因**，
  不能静默给出错误输出。

### W7 测试与 CI

- **现状**：`macos-arm64-mlx` job 只跑参考脚本；ctest 里没有 MLX 项。
- **怎么做**：`SGL_MLX=ON` 时注册 `mlx_test` / `test_mlx_ops`（逐算子 vs CPU 实现）；
  CI 里该 job 增加 `-DSGL_MLX=ON` 配置 + 构建 + `ctest -R mlx`；保留现有
  `grep -q 'MLX('` 的 Metal 可用性断言。**注意 runner 内存/时长**：MLX 测试要限模型大小。
- **验收**：CI 上 `macos-arm64-mlx` 从「CPU ctest + 参考脚本」变成「真编 MLX 后端 + MLX ctest 通过」，
  且 `macos-arm64`（不带 MLX）仍然全绿；iMac 真机跑一遍同样的 ctest 并留档
  （对标 `docs/g_macos_arm64.txt` 的格式，新增 `docs/g_macos_mlx.txt`）。

### W8 文档与 CLI 口径（做完必须一起改，否则仓库自相矛盾）

| 位置 | 现在写的 | 要改成 |
|---|---|---|
| `apps/sglc.c:179-180` | 「MLX 目前无后端（只有 macOS 的数值参考脚本），指定它会报错后回落 cpu」 | 实际能力 + 不支持的模型族 |
| `apps/sglc.c:333-343` | mlx 档只打日志跳过 | 真探针（`mx::metal::is_available` 等价物）+ 起 MLX 引擎 |
| `README.md:121,127,237,245-247` | 「no MLX backend in this repo」 | 后端矩阵加 MLX 行 |
| `README.zh-CN.md:45,51,98,136` | 同上中文版 | 同上 |
| `CHANGELOG.md:189-191` | 「MLX 本仓库没有实现」 | 新版本条目 |
| `docs/GAP_vs_upstream.md:22` | 平台列「x86-64 CPU、CUDA sm_75+、arm64 CPU（macOS 单线程）」 | 加「Apple Silicon GPU（MLX/Metal）」 |
| `ROADMAP.md` | 无 MLX 里程碑；「不依赖任何第三方库」 | 新增里程碑行 + 改写依赖约定（MLX 为可选例外） |
| `TODO_PROMPTS.md` | 无 | 按 G 系列格式补一条（现状证据/为什么/怎么做/验收） |

---

## 5. 算子映射表（`src/cuda_backend.cu` 的 54 个 kernel → MLX 0.29.3）

| CUDA kernel | MLX 对应 | 备注 |
|---|---|---|
| `k_embed` / `k_embed_h` | `mx::take` | 嵌入查表 |
| `k_embed_pos` / `k_embed_pos_h` | `take` + `add` | gpt2 绝对位置 |
| `k_rmsnorm` / `k_rmsnorm_p1` / `k_hrmsnorm` | `mx::fast::rms_norm` | 直接可用 |
| `k_layernorm_b` | `mx::fast::layer_norm` | gpt2 带 bias |
| `k_gemm_nt` / `_h` / `_mrow` / `_h_mrow` / `k_fill_bias` | `mx::matmul` / `mx::addmm` | fp16/bf16 原生 |
| `k_rope` / `k_rope_p` | `mx::fast::rope(x, dims, traditional, base, scale, offset)` | 标量 offset |
| `k_rope_rows` / `k_rope_p_rows` | `mx::fast::rope(..., const array& offset)` | **逐行位置重载，正好是 batch decode** |
| `k_attn` / `k_attn_long` / `k_attn_long_batch` / `k_attn_batch` | `mx::fast::scaled_dot_product_attention` | `mask_mode="causal"` |
| `k_attn_q` / `k_attn_q_flash` / `k_attn_q_batch` / `k_attn_q_batch_flash` / `k_attn_q_split_partial` / `k_attn_q_split_merge` | 同上 | 分块/split-KV 是 CUDA 侧为 8K prefill 做的工程优化；MLX 的 memory-efficient 路径能否等效**需实测**，不行就退回按块循环 |
| `k_silu_mul` | `mx::silu(gate) * up` | SwiGLU |
| `k_gelu` | `mx::gelu_approx` | gpt2 |
| `k_residual` | `add` | |
| `k_cat2_rows` | `mx::concatenate` | |
| `k_argmax` / `k_argmax_p` | `mx::argmax`（+ `take` 取概率） | 采样仍在主机（与 CUDA 侧一致） |
| `k_f2h` | `astype(float16)` | |
| `k_kv_write` / `k_kvq_write` | 预分配 + `mx::slice_update` | chunk 取 256 倍数 |
| `k_gemm_w8a8` / `_q` / `k_quant_rows` / `k_w8_dequant` | `mx::quantize` + `mx::quantized_matmul` | **格式不同，加载时转换**（W4） |
| `k_gemm_w4_q` / `k_gemm_w4_mrow` | `quantized_matmul(bits=4)` | 同上 |
| `k_qgate_split` / `k_qgate_apply` | elementwise | |
| `k_lin_conv` / `k_lin_l2norm` / `k_lin_ba` / `k_lin_step` / `k_lin_gnorm`（+ `_b` 变体、`k_lin_layer_persist`） | **无对应** | GDN，需 `mx::fast::metal_kernel` 自定义（T3） |
| `src/cuda_deepseek.cu`（MLA，1045 行） | **无对应** | T3 |
| CUDA Graph capture/replay（`sgl_cuda_prewarm`） | `mx::compile`（decode step） | 语义不同：MLX 靠图编译而非显式 capture；**是否需要、收益多少要实测** |
| `cudaMemGetInfo`（`avail_vram_mb` / `estimate_vram_mb`） | `mx::get_active_memory` / `set_memory_limit` / `set_cache_limit` / `set_wired_limit` | 统一内存下没有「空闲显存」概念，`cuda_http.c` 里那套显存预检 + 503 降级逻辑要重写判据 |

---

## 6. 分期（建议按此交付，每期都能独立验收）

> **状态（2026-11-04）**：T1 ✅ 完成、T2 的「性能」部分✅ 完成；
> T2 的量化/前缀缓存/多模态/gpt2 与 T3 未做。详见
> [NOTES_mlx_t2.md](NOTES_mlx_t2.md)（含与官方 mlx_lm 的基准对比）与
> [PROGRESS_mlx.md](PROGRESS_mlx.md)。

| 期 | 内容 | 交付判据 |
|---|---|---|
| **T1** | W1 接缝 + W2 构建 + W3 llama 系 fp16 前向/decode/batch + `--device mlx` 起服务（`/health` `/generate` `/v1/completions` `/v1/chat/completions` `/v1/models` `/flush_cache` + SSE 流式） | iMac 真机：qwen05 与 tinyllama greedy 与 CPU 逐 token 一致；HTTP 端到端 4 个脚本通过；`docs/BENCH.md` 有 M1 的 prefill/decode tok/s（对比当前 macOS 单线程 CPU） |
| **T2** | W4 量化（含 mlx-community 4-bit 检查点，若决策 3 选是）+ W5 前缀缓存 / 多模态 emb / gpt2 | 前缀重发加速比与输出一致性；`test_llava` vs MLX 参考 < 2e-4；16 GB 上「最大可跑模型」实测 |
| **T3** | W6 GDN（`metal_kernel`）/ MLA / MTP 投机 + `mx::compile` 调优 | Qwen3-Next 族在 MLX 上输出与 CPU 对齐；投机接受率与 tok/s 实测 |
| 贯穿 | W7 测试与 CI、W8 文档口径 | 每期合并时同步更新，不留「文档说没有、代码里有」的状态 |

---

## 7. 会咬人的语义差异（动手前必须先定，否则返工）

1. **惰性求值**：MLX 的数组是懒的，不 `mx::eval` 就什么都不算。每步只 eval 需要的东西
   （logits 那一行），并用 `mx::async_eval` 让下一步的图与主机侧采样重叠。
   eval 放错位置的两种失败形态：吞吐为 0（每步同步全图）或内存爆（图一直攒不求值）。
2. **数组不可变**：KV 必须预分配 + `slice_update`；「段」= 切片。`copy_kv` = 切片拷贝。
   不要试图用指针写进 MLX 数组的 buffer（`data_ptr` 只在 eval 之后、且不被图复用时才安全）。
3. **dtype 与容差**：MLX 默认 fp32 但推理常用 bf16/fp16；与 CPU fp32 金标准比对的阈值
   **不能沿用 CUDA fp32 的 3e-5**，要先测出 fp16 的实际量级再定阈值（预计 1e-3 级）。
4. **线程模型**：`cuda_http.c` 是「队列 + 工作线程」；MLX 的 stream 是线程局部的，
   跨线程要么单 stream 串行，要么 `new_thread_local_stream`。这一条不定清楚，并发测试会偶发崩。
5. **统一内存 ≠ 无限内存**：16 GB 机器上权重 + KV + 图缓存共享同一池；
   需要 `set_wired_limit` / `set_cache_limit` 与一个 MLX 版的「装不下就明确 503」判据。
6. **`mlx.metallib` 与 rpath**：可执行文件跑起来要能找到 `@rpath/libmlx.dylib`，
   且 dylib 要能找到同目录的 `mlx.metallib`。打包（CI 的 `dist/`）时这两条都要处理，
   否则「本机跑得过、产物跑不起来」。

---

## 8. 第一步：最小可验证闭环 —— **已完成（2026-10-29）**

冒烟程序已入库：`tools/mlx_smoke/{smoke.cpp,CMakeLists.txt,README.md}`（不碰仓库任何现有代码）。
在 iMac 上（源码与 build 都在 `/Volumes/9100 PRO/sgl-mlx-smoke`，经 `~/v2t` 符号链接避开路径空格）
`cmake -G Ninja -DSGL_MLX...` 配置 + 编译 + 链接 + 运行全通过，输出 `SMOKE_PASS`：

```
version      : 0.29.3          default_dev : type=1(gpu) index=0   metal_avail : 1
metal.device_name              : Apple M1
metal.architecture             : applegpu_g13g
metal.memory_size              : 17179869184   (16 GB)
metal.max_recommended_working_set_size : 12713115648  (11.84 GB)
metal.max_buffer_length        : 9534832640    (8.88 GB)
rms_norm  [4,64] fp32          maxerr vs host = 1.19e-07
rope(array offset) B=3         vs 逐行标量 offset maxerr = 0
sdpa(causal) B3 H2 S6 D8       maxerr vs host = 2.98e-08
fp16 rms_norm                  dtype=float16，无 NaN/Inf
otool -L                       @rpath/libmlx.dylib（LC_RPATH 指向 wheel 的 lib/）→ 解析成功
```

**由此确定的三件事**：① 工具链/rpath/metallib/Metal 全链路可用，W2 没有未知风险；
② `fast::` 三个关键算子精度在 1e-7~1e-8 量级，**fp32 下可以沿用 CUDA 侧的严格阈值**，
只有 fp16/bf16 档需要放宽容差（W3 的验收阈值按精度档分别定）；
③ 内存预算判据 = `max_recommended_working_set_size`（11.84 GB），W2/W4 的「装不下就明确报错」
按这个数写。

**下一步就是 W1（接缝）**，动手前需要 §9 的三个决策。

---

## 9. 需要拍板的三个决策

1. **接缝**：A 抽 `sgl_accel` 表、`cuda_http.c` 改成后端无关（推荐，多花约 1 天，但要重验
   5090 全绿）／ B 直接复制一份 `mlx_http.c`（快，永久留下 2000 行重复）。
2. **范围**：只做 T1（llama 系 fp16 + 服务）／ T1+T2（含量化、前缀缓存、多模态）／
   T1–T3 全量对齐 CUDA 后端（含 GDN、MLA、MTP，工作量约为 T1 的 3–5 倍）。
3. **权重格式**：只沿用现有 safetensors/GGUF/AWQ/GPTQ 加载器（转成 MLX 量化格式）／
   额外支持直接加载 mlx-community 的 4-bit 检查点（在 16 GB M1 上更实用，但要新增一条加载路径）。
