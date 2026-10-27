# G11–G19 执行进度（loop 自动续作）

> 本文件由定时任务「按 TODO_PROMPTS.md 做完全部」维护，每轮结束更新。
> 任务定义见 ../TODO_PROMPTS.md（v0.4）。顺序按文档附注建议：
> G11 → G17 → G19 → G16 → G14 → G12 → G15 → G13 → G18。

## 环境事实（2026-10-27 复核，**重要**）

- **真正的模型目录是 `C:/Users/ultra/Desktop/works/sglang/models`**（`CMakeLists.txt` 的
  `SGL_MODELS_DIR = ${CMAKE_CURRENT_SOURCE_DIR}/../models`）。里面的真权重都在：
  `qwen05`（949MB）、`qwen3-06`（1.45GB 含权重）、`tinyllama`（2.1GB）、`llama-160m`、
  `clip-vit-base-patch32`（914MB，含 ref/）、`test-lora*`、`gpt2`、`tiny-llama`，
  以及随机模型 `{mistral,gemma,gemma2,qwen3next,llava}-rand`。
  `sglang-c/models` 只是历史残留的一份子集（没有 qwen05/tinyllama/llama-160m）——
  查模型在不在**不要**只看 `sglang-c/models`。
- `models/mm-test` = `mistral-rand` 的 LM + `llava-rand` 的 clip_config/clip_model/projector
  + `test.png`/`test.jpg`（G11 验收用；在 `../models/mm-test`）。
- CPU HTTP 版 `--serve` 只有 MinGW 构建（`build/bin/sglc.exe`）；
  `build-cuda/bin/Release/sglc.exe` 会报「--serve not available in MSVC builds」。
- 本机 GPU：6GB，`cuda_http` 常驻 ~445MB（预检 `ctx=512 rows=64`）；WDDM 会溢出，
  性能结论必须同进程 A/B。

## 状态表

| 任务 | 状态 | 证据 |
|---|---|---|
| G11 GPU 收标准图片 | ✅ 完成 | 见下 |
| G17 G4 GPU 版复验 + soak 误报 | ✅ 完成（**找到并修掉真根因**） | 见下 |
| G19 CI 真跑模型族与多模态 | ✅ 完成 | 见下 |
| G16 W8 重测 + flash 泛化 | ✅ 完成 | 见下 |
| G14 投机正收益 | 已做完（未达 ≥1.0x，如实记“仍净减速”） | 见下 |
| G12 真实 llava-1.5 | ✅ 完成（两层 MLP + 占位插入 + 真实 CLIP/projector 分段对照） | 见下 |
| G15 前缀缓存 radix | ✅ 完成（租让覆盖生成 token + 带图 text；四组全 PASS） | 见下 |
| G13 Qwen3-Next full 层 | ✅ 完成（q 门控 / partial rope / (1+w) / MoE；递归状态按序列） | 见下 |
| G18 8K GPU-vs-CPU 绝对对照 | ✅ 完成（maxerr 1.47581e-04 < 1e-3，greedy SAME，PASS） | `docs/g18_8192.log` |

## G11 ✅ GPU 版收标准 JPEG/PNG

- 改动：`apps/cuda_http.c` 的 `vision_feats()` 按 data URL mime 分派——
  `data:image/{png,jpeg,jpg};base64,` 走 `sgl_base64_decode` → `sgl_clip_project_bytes`
  （stb_image 解码 + PIL 语义 resize/crop + CLIP + projector，与 CPU 版同一条路径）；
  其余载荷仍走 t10 的 raw float32 CHW 兼容分支。`CMakeLists.txt` 把 `src/image.c`
  加进 `sgl_cuda`。`tests/http_image_e2e.py` 新增 `--url` / `--images` 模式
  （指向已在跑的服务、不自起进程），输出改成 utf-8 replace（随机模型的 token 会让
  GBK 控制台炸 UnicodeEncodeError）。
- 验收（`cuda_http.exe ..\models\mm-test 30024` + `python tests\http_image_e2e.py
  --url http://127.0.0.1:30024 --images ..\models\mm-test`）：
  jpeg 200 / png 200 / text 200；jpeg≠png、有图≠无图；坏 base64 → 400 `bad image ...`。
- CPU 对照：同一套图（`build\bin\sglc.exe --model models/mistral-rand --serve --vision
  models/llava-rand`）输出逐 token 前缀与 GPU 版完全相同
  （png: `sports素材 mah passionately indicate Monroeจังหวัด合法…`）。
- 回归：`cuda_test.exe models\mistral-rand` 的 `[mm]` GPU 融合 vs CPU 融合
  **maxerr 1.19e-07、greedy SAME**（历史 1.2e-7）。
- 注：`models/mistral-rand` 这种 2 层随机模型 W8 greedy 会 7/10（小 logits 上量化误差
  翻 top1），历史 W8 验收用的是真 qwen05 —— 别当成回归；`cuda_test` 换 qwen05 跑即可。

## G17 ✅ GPU 版「/health 偶发卡死」根因 + soak 误报

**根因（不是 CPU 版 G4 那个 acceptor 饿问题）**：`cuda_http.c` 的 `read_request()`
在**没有 NUL 截断**的请求缓冲上用 `strstr(buf,"Content-Length:")`。负载一大，8KB 的
malloc 缓冲经常是从前一个 POST 请求回收来的，len 之后残留着 `Content-Length: 105`；
`cl` 被当成 105 后 body 循环一直等永不会来的字节 —— 于是无 body 的 `GET /health`
挂到客户端超时为止。空闲时堆刚拿是 0 才看不出来。

定位过程（可用 `SGL_HTTP_TRACE=1` 复现）：trace 打出 accept/conn enter/recv/hdr/sent/conn done
的 tick，看到「hdr 之后到 sent 之间的间隔恰好等于客户端超时」；12 并发 load 下
`g4_probe` 的 `timeout_read` 恒等于客户端超时值。

修复：每次 recv 后 `buf[len]=0`；Content-Length 只在**本请求 header 段内**做有界搜索
（与 `src/http.c` 的 t5 修法一致）。

验收：
- GPU：`g4_probe load 600 12 256 --port 30024` → **n=600 ok=600 timeout_read=0**
  （修复前同负载 0/600 全超时）；complete_ms med 14.9 / max 26.3。
- CPU：`g4_probe load 600 12 256 --port 30000` → **600/600 ok，0 超时**
  （requests 60/60、curl 60/60 也 0 超时）。
- soak：`soak_curve.ps1 -Exe build-cuda\bin\Release\cuda_http.exe -Model models\mm-test
  -Port 30033 -Minutes 2` → RSS 斜率 0.020 MB/min（0.0044%/min）、VRAM 0.000 MB/min，
  PASS，stderr 无 ERROR/FATAL（INFO 白名单生效，模型加载时那些「可选张量 not found」
  不再误报）。
- 顺手：`g4_probe.py` 支持 `--host/--port`（以前写死 30000）。

## G19 ✅ CI 真跑模型族与多模态

- `gen_random_{mistral,gemma,qwen3next}.py` 加 `--out/--tokenizer`，`gen_random_gemma.py`
  另有 `--variant dynamic`（gemma2），`gen_random_llava.py --out`；CI 的 Linux/Windows
  job 都先跑这 5 个生成 + `ref_llava.py`，让 CMake「模型存在才注册 test」的条件成立。
- 验收判据改成正向且更严：`test_family_{mistral,gemma,gemma2,qwen3next}.*Passed` ≥ 4、
  `greedy6=一致` 行数 ≥ 4、`test_llava: all OK` ≥ 1。
- 本地核验：新参数生成的 6 个 safetensors 与 `../models/*-rand` **逐字节 IDENTICAL**
  （所以与仓库里的 `_ref_*.json` 对得上）；`ctest -V` 本地 23/23 全绿，log 里
  `greedy6=` 22 处、`all OK` 23 处、无 SKIP。

## G16 ✅ W8 IMMA 后重测 + flash 泛化

数据全在 `docs/BENCH.md` 的「G16」节。要点：

- 三精度 prefill（qwen05，ctx=16384，rounds=2）：fp32 85.4/386.7/2748.9 ms、
  fp16 52.2/198.4/1971.9 ms、int8 48.6/184.5/1923.8 ms（512/2048/8192）。
- dp4a vs IMMA **没有拐点**：M=1 持平（decode 仍选 dp4a），M≥2 从 46.7x 输到 155x。
- **short 核阀值 4096 → 64**（`SGL_ATTN_FLASH_MIN`）：128..4096 tok 上 flash 比 short
  快 1.3x–20.2x；dispatch 收成单个 `attn_dispatch()`；未特化 D 行为不变。
- head_dim：新增 D=96 特化 + `D % 32` 守卫（D=80 不加守卫会静默只算前 64 维，
  实测 maxerr 0.83 → 加守卫后回退 7.15e-07）。
- 新模型：`../models/d96-rand`（E=192 H=2 KVH=1 L=2 D=96）、`../models/d80-rand`
  （E=320 H=4 D=80，验不整除 32 的回退）。
- 验收：`cuda_test ..\models\qwen05` → `test_cuda: all OK`；[long] 三路对比
  short(32)/default(flash@2000)/flash@4300 vs 两遍全 SAME；D=96 三长度 GPU vs CPU ≤8.4e-07。
- 工具：`tests/bench_prefill.ps1`、`long_vcpu` 新增 flash-vs-two-pass、
  `SGL_ATTN_DEBUG=1` 打实际用的核、`gen_random_mistral.py` 可指定维度。

### 新增（G14）
- **投机解码流水线（`src/cuda_spec.c`）**：预填 `np-1` 行，prompt 末 token 作为“挂起
  token”由下一轮 verify 第 0 行吞掉 → 每轮 verify 固定 n=K+1，reject/bonus 的 KV
  顺带被写掉，**去掉每轮一次 target 图步**。tinyllama+llama-160m（ctx=1024，new=32，
  best-of-3）：K=1 0.44x→0.73x（60.5 tok/s）、K=2 0.35x→0.64x、K=4 0.27x→0.40x。
  `cuda_test` 的两个 [spec] 例子（draft==target 100%、真实组合 13.3%）依旧逐 token
  等于非 spec greedy。**仍末达到 ≥1.0x**：按实测回本需 K=1 a>88% / K=2 a>69.5%，
  而实际只有 39.1% / 28.6%。
- **small-M GEMV（`k_gemm_nt_mrow` / `k_gemm_nt_h_mrow`，模板 MT=2/4/8）**：M=2..8 时
  权重行只读一遍、同时累 MT 行 query。verify：n=2 22.1→16.1ms、n=4 25.8→18.5ms；
  批量 decode B=2..8 与 2–8 token 短 prefill 同样受益。踩过的坑：动态 M 落 local
  memory、MT 轮同步归约让 n=5 反而变慢、MT>M 时不能读 A 的越界行。
- **诊断开关**：`SGL_SPEC_PROF=1`（每轮 draft/verify/补齐分解）、
  `SGL_ATTN_DEBUG=1`（用的哪个 attention 核）、`SGL_SMALLM_CUBLAS=1`（M=2..8 退回 cublas A/B）。

### 新增（G15）
- **前缀租让覆盖生成 token**（`apps/cuda_http.c`）：非视觉请求登记 `n_prompt+(n_gen-1)` 行，
  多轮对话/续写整段复用。实测命中 552 → **573 行**，同 prompt 重发 0.262s → **0.103s（2.5x）**。
- **带图请求也登记 text 部分**：同文本不同图命中 **135 行**（以前 0）。
- **`/health` prefix 字段**：slots/hits/misses/hit_rate/hit_rows/leased_rows/flushes；
  `SGL_PFX_MAX=0` 可全面关缓存，超容量请求会 flush 全部租让。
- **`tests/prefix_e2e.ps1` 四组全 PASS**（同前缀/续写/带图/显存紧张），每组都与
  「关缓存」逐字对比。任务里的「跳段两轮扫描」已实测无收益（扁平表上界就是最长的一条），
  改用「登记生成 token 行」拿到真实收益。
- 回归：`cuda_test ..\models\qwen05` → `test_cuda: all OK`（[prefix] reuse vs full
  maxerr 1.4e-05、greedy 6/6 SAME、prefill 33.4ms vs 68.8ms）；CPU ctest 23/23。

### 新增（G13）
- **Qwen3-Next full 层真实特性**（`src/llama.c`/`src/ops.c`）：q 门控（q_proj 2×、
  sigmoid 半幅门）、partial rotary（0.25 维）、(1+w) RMSNorm、MoE（4 专家 top-2 +
  共享专家 + routed_scaling）。config 同时支持 `layer_types` 与 `full_attention_interval`。
  验证：随机模型 prefill top5 maxerr **2.5e-08 – 5.8e-08**（要求 <1e-6）、greedy6 全一致、
  decode_batch 一致；`test_llama: all OK`；全量 ctest **24/24**。
- **递归状态按序列分配**：64 位 seq key（旧实现用首 KV 页 → radix 前缀共享时互踩）；
  新 ctest **test_lin_seq**：共享前缀 + 不同后缀并发 decode 与单独跑**逐位相同**，
  并且带反向对照（故意共用 key → 结果不同）证明测试有鉴别力。
- 新 API：`sgl_llama_forward_seq` / `sgl_llama_decode_batch_seq` /
  `sgl_llama_lin_release` / `sgl_llama_lin_slots_in_use`（调度器接入时要用 release 归还槽）。

### 新增（G12）
- **两层 MLP projector**（`sgl_projector`，自动判层数 + GELU erf）+ **视觉 token 插在
  `<image>` 占位位置**（scheduler `img_at`，尾巴文本位置顺延）。
- **tokenizer 修 bug**：added token 只在 pre-token 边界匹配 → `USER: <image> ...`
  根本产生不了 `<image>` id（测试第一轮就是这么失败的）；现在先按字面量切再 BPE。
- **真实权重分段对照**（任务允许）：真实 openai/clip-vit-large-patch14-336（392 张量）
  + 真实 liuhaotian/llava-v1.5-7b 的 mm_projector（1024→4096→4096），并支持
  `mm_vision_select_layer=-2`。与 torch/transformers fp32 对照：预处理 rel 1.15e-07、
  CLIP patch 行 rel **1.41e-05**、mm_projector rel **1.78e-05**。
- 新 ctest：**test_g12_place**（占位符插入 == 等价路径、≠ 旧行为）、**test_llava_real**
  （真实权重分段对照）。全量 **26/26**。
- 未做：真实 7B LM 的端到端（fp16 13.5GB，本机只剩 3.5GB 磁盘）；多图仍拼在同一占位位置
  （llava 是一图一占位）。

### 新增（G18）
- **8K GPU-vs-CPU 绝对对照通过**：`SGL_POOL_SLOTS=8208 long_vcpu ..\models\qwen05 8192`
  → GPU prefill **2.8s**、two-pass 66.1s、CPU prefill **3733.0s（62.2 分钟）**、
  **maxerr 1.47581e-04 < 1e-3**、greedy 5562 vs 5562 SAME、verdict PASS。
  原始输出 `docs/g18_8192.log`；空闲内存 23.3GB/32GB。
  - 顺带证实：CPU 侧是**单线程**（MSVC 构建没开 OpenMP，CPU 时间 ≈ 墙钟），
    这就是以前“40+ 分钟没跑完”的原因；开 OpenMP 预计能降到 ~10 分钟量级（未实测，
    不当结论）。
  - 干净环境下 flash 在 8K 行上比 two-pass 快 **23.6x**（G1 当时记录的 8.3x 是显存溢出/
    并发争用状态下的数，两个数字不要互相引用）。
### 修复
- **MSVC 构建的 CPU 路径一直只跑 SSE2（慢 10x）—— 已修**（`src/dot_avx2_msvc.c` +
  `src/ops.c` + CMake）：AVX2 内积原来只对 GCC 可用（`__attribute__((target("avx2,fma")))`），
  MSVC 没有函数级 target 属性，于是整条 CPU 路径（llama 的 GEMM 是「每行一个内积」）
  落在 SSE2 基线。实测 qwen05 930-token prefill：MSVC 326.2s vs GCC 单线程 31.9s（差 10.2x）。
  现在把 AVX2 版本放进单独编译单元（只给它 `/arch:AVX2`，其余代码仍 x86-64-v2 基线，
  避开 t12.2 的全程序 AVX2 自动向量化 bug），用 CPUID（AVX2+FMA+OSXSAVE）运行时分派，
  不支持就退回 SSE2。修后：930-token 326.2s → **33.6s（9.7x）**；8192-token CPU prefill
  3733.0s → **611.8s（6.1x）**（g18b_8192.log，maxerr 1.43409e-04 仍 PASS）。
  `cuda_test ..\models\qwen05` 仍 all OK（CPU-vs-GPU prefill maxerr 1.79e-07 → 2.15e-05，
  归约顺序变化属 fp32 噪声）；CPU 构建 ctest 26/26。
- 同时更正了自己上一轮的错判：把 CPU 慢归因于「没开 OpenMP」。实测 OpenMP 只值 **2.05x**
  （930-token：单线程 31.9s → 8 线程 15.6s），归因已改。

- 至此 **TODO_PROMPTS v0.4 的 G11–G19 全部完成**（只剩可选项未做：G12 的 7B LM 端到端
  因磁盘不够、G13 的 CUDA 路径、G15 的跳段共享——都有实测依据说明无意义/不可能）。

## macOS arm64 真机 + MLX（第 16 轮，2026-10-27）

用户提供了一台 iMac（`192.168.20.14`，用户 `wangchao`）。上一轮我写的「macOS 只能静态审查」
**因此作废**：这一轮在真机上把 `macos-arm64` 矩阵项跑通了，并按用户要求用 **MLX** 做数值参考。

### 环境（真机，非模拟）

| 项 | 值 |
|---|---|
| 系统 / 架构 | macOS 27.0.1 / arm64（8 核） |
| 编译器 | Apple clang 21.0.0 (clang-2100.3.34.2) |
| cmake / ninja | 3.31.6 / 1.13.1（官方预编译二进制，装在 `~/tools`，**没装 Homebrew、不需要 sudo**） |
| OpenMP | **不可用**（AppleClang 无 `-fopenmp`）→ 走单线程 fallback，正是 CMakeLists 里的 else 分支 |
| MLX | 0.29.3，`mx.default_device()` = `Device(gpu, 0)`（Metal） |
| python | 3.9.6（venv），numpy 2.0.2、tokenizers 0.22.2 |

### 结果

| 步骤 | 结果 |
|---|---|
| 解包 + 按 workflow 下载 gpt2(523MB) / qwen3-06 tokenizer | OK |
| 生成 5 个随机族 + `ref_llava.py` | 全部成功，`proj_checksum 1.072851`（与 x86 逐字节一致） |
| `cmake -G Ninja` + `cmake --build build` | **66 个目标全部编译链接成功** |
| `ctest -V --timeout 1800` | **24/24 通过**（`test_radix` 14.4s、`test_scheduler` 4.1s） |
| G19 gate（同一套判据） | **counts: fam=4 fam_ran=4 llava=1 → MACOS_GATE_PASS** |

摘要留档：`docs/g_macos_arm64.txt`。

### MLX 作为数值参考（用户要求「测试在 mac 使用 mlx」）

新增 `tests/ref_llava_mlx.py`：与 `ref_llava.py` 数学等价、实现独立，**计算全部在
mlx.core（默认 Metal GPU）上**，只借用 numpy 读 safetensors 和落 `.npy`。它覆盖写
`models/llava-rand/_ref_{proj,feats,imgemb}.npy`，于是 `test_llava` 的比对对象就从 numpy
换成了 MLX（测试本身一行没改）。

| 对照 | 数值 |
|---|---|
| MLX(Metal) vs numpy 参考（proj） | maxerr **1.49e-08**，max\|ref\|=0.057 → **rel 2.60e-07** |
| MLX vs numpy 参考（feats） | maxerr **9.54e-07**，max\|ref\|=3.12 → **rel 3.06e-07** |
| **C 实现 vs MLX 参考**（CLIP patch feats） | maxerr **4.77e-07** |
| **C 实现 vs MLX 参考**（两层 MLP projector） | maxerr **1.49e-08** over 16×64 |
| 换成 MLX 参考后重跑全量 ctest | 仍 **24/24**，`test_llava: all OK` |

结论：三个独立实现（C fp32 CPU、numpy fp32/f64、**MLX Metal fp32**）在这一层的相互差
都在 1e-7 量级，远低于测试阈值 2e-4。

### 期间修/发现的三件事

1. **`test_llava` 的输出文案过期**：三处写的是 `vs numpy`，但参考文件可能来自 MLX。
   已改成 `vs ref(_ref_*.npy; numpy 或 MLX)`。
2. **`test_llava.c` 一个 Clang 可见的未使用变量**（`ref`，G12 后比对改为读 `.npy`）：
   加 `(void)ref;` 并注明参数只为兼容旧调用。
3. **一次「假失败」是我自己的脚本造成的**：第一次解包时 `git archive` 的 tar 没有顶层目录，
   我 `tar -C ~/sglang-work` 把仓库解到了 `~/sglang-work/` 根上，于是遗留了
   `~/sglang-work/models/gpt2/`（只有 config、没有权重）。CMake 的
   `if(EXISTS ../models/gpt2)` 因此成立，ctest 的 `SGL_TEST_MODEL` 指向那个空目录，
   `test_radix` 报 `sched init failed` 而红。删掉遗留目录重新 configure 后 **24/24 全绿**。
   教训：CMake 那个「优先 ../models」的分支对**残缺目录**是敏感的。

### 仍未直接验证

`macos-x64`（Intel / Rosetta 交叉编译）：这台 iMac 是 arm64，用
`CMAKE_OSX_ARCHITECTURES=x86_64` 可以交叉编译，但那需要额外一轮；`release` job 仍需 push tag。

### 留在 iMac 上的东西（下轮可直接复用）

- `~/tools/cmake/CMake.app/...`、`~/tools/ninja`、`~/tools/venv`（含 **mlx 0.29.3**）
- `~/sglang-work/sglang-c`：已构建的树（`build/bin/` 里 24 个测试可执行），
  含 `models/gpt2`（523MB）与 5 个随机族；磁盘剩余由 6.3GB 降到 **3.6GB**
  （不再需要时直接删 `~/sglang-work` 即可回收）
- 网络：HF 可达（2.1s）；PyPI 直连太慢，**用清华镜像**才装得上 mlx

### 本轮未完成

- **ASan + UBSan 跑全量测试**：起点是找内存/UB 问题（上一轮 Clang 警告刚抓到过一个真 bug）。
  在 Docker 里已构建成功（clang + `-fsanitize=address,undefined`）、跑到 11/24 时 Docker
  守护进程失去响应（同时把本机 C: 撑到 **0 字节**），容器被清掉，**未重跑**。
  接着做的方式：`-DCMAKE_C_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"` 构建后
  `ASAN_OPTIONS=detect_leaks=0 ctest -V`，再 grep `ERROR: AddressSanitizer` / `runtime error:`。

macOS job 跑不了（无 macOS），但能做一件同等价值的事：**用 Clang 构建整个项目**。
`CMAKE_C_COMPILER_ID == Clang` 正是 macOS job 走的分支（Apple Clang 也是 CLANG），而且
Clang 不装 libomp 时走的是**非 OpenMP 单线程 fallback**——这条组合在 x86 上从未验过
（之前只有 GCC+OpenMP 和 MSVC 无 OpenMP）。

### 结果（ubuntu:24.04 容器，clang 18 + Ninja，`-DCMAKE_C_COMPILER=clang`）

| 项 | 结果 |
|---|---|
| configure | `OpenMP not available with Clang, single-threaded fallback`（如预期） |
| build | **65/65 全过，0 error**（`-march=x86-64-v2`、`__attribute__((target("avx2,fma")))`、`__builtin_cpu_supports` 在 Clang 下都成立） |
| ctest | **24/24 通过**；`test_llava` 真跑（`[llava] CLIP patch feats vs numpy maxerr = 4.77e-07`） |
| G19 gate | **counts: fam=4 fam_ran=4 llava=1 → PASS** |

SKIP 的全是 fresh clone 本该缺的夹具（qwen05/tinyllama/clip-vit-*/lora/g12-mlp），与 CI 一致。

### Clang 的警告抓到一条 GCC 没报的**真 bug**（已修）

```
/work/src/spec.c:61:7: warning: variable 'dl' is used uninitialized whenever 'if' condition is true
                        [-Wsometimes-uninitialized]   （tl / vl 同样三条）
```

`src/spec.c` 的 `sgl_spec_generate()` 里，`dl/tl/vl` 原本声明在 `sgl_llama_pages_alloc()`
**之后**，而 pages_alloc 失败会 `goto fail`，`fail:` 里又会 `sgl_xfree(dl/tl/vl)` ——
**释放三个未初始化的指针**（堆损坏 / 崩溃）。触发条件是「KV 池耗尽」，是现实路径而不是
理论路径。GCC 的 `-Wall -Wextra` 不报这个，MSVC 也不报，所以一直没被发现。

修法：把三个指针提到任何 `goto fail` 之前并初始化为 `NULL`（`sgl_xfree` 对 NULL 安全：
POSIX 版判空、MSVC 的 `_aligned_free(NULL)` 也是 no-op）；顺手把「主机内存不足」与
「页表分配失败」分开处理，并在 malloc 失败时不再走 `pages_unref`（那条路页表没填过）。

**验证**：
- MSVC/MinGW 本地：`cmake --build build` 0 error；`test_spec`（用 `../models/tinyllama`
  + `../models/llama-160m`）→ `spec: drafted=60 accepted=8（13%）`、`[spec==greedy ✓]`、
  `test_spec: all OK`；全量 `ctest` **26/26**。
- Clang：把修好的 `spec.c` 拷进容器重编 → **`-Wsometimes-uninitialized` 计数 = 0**，
  且再跑完整 `ctest` 仍 **24/24 通过**（116s）。

顺带核过同类模式：`src/cuda_spec.c` 的 `dl/tl/vl` 都在第一个 `goto done` 之前赋值，
`if (!seq || !dl || !tl || !vl) goto done;` 也做了判空，**没有同样的问题**。

### 仍未验证

`macos-arm64` / `macos-x64`（无 macOS；x64 还要 `CMAKE_OSX_ARCHITECTURES=x86_64` 交叉编译
+ Rosetta）。本轮把 Clang 前端与无 OpenMP 配置都验过了，剩下未覆盖的只有 Apple 平台
特有的东西（Apple ld、libSystem、`-mmacosx-version-min` 等运行时/链接器差异）。

## linux-arm64 矩阵项实跑（第 14 轮，2026-10-27）

上一轮我判定「arm64 跑不了」，**这个判断也不对**：Docker Desktop 带 qemu binfmt，
`--platform linux/arm64` 能跑 aarch64 容器（`uname -m` → `aarch64`）。于是把 CI 里
`linux-arm64` 那一项也用 ubuntu:24.04 + qemu 模拟跑了。

这一项值得单独验：arm64 上 `__SSE2__`/AVX2 全部不成立，`src/ops.c` 会退到**标量**内积路径，
`src/dot_avx2_msvc.c` 只对 MSVC 生效——也就是说 CPU 数值路径与 x86 是**两套不同代码**。

### 结果（qemu 模拟，耗时不可与原生相比）

| 步骤 | 结果 |
|---|---|
| `apt-get install build-essential cmake ninja-build ...` | OK；GCC 报 `OpenMP: enabled (compiler-driven)`（arm64 上 GNU 同样开） |
| 生成 5 个随机族 + `ref_llava.py` | 全部成功，`proj_checksum 1.072851`（与 x86 逐字节一致） |
| `cmake -G Ninja` + `cmake --build build -j` | **65/65 全部编译链接成功（BUILD_OK_ARM64）** |
| `ctest -V --timeout 1800` | **100% tests passed, 0 failed out of 24** |
| G19 gate（`bash -e -o pipefail`） | **counts: fam=4 fam_ran=4 llava=1 → ARM64_GATE_PASS** |

摘要留档：`docs/g_arm64_ctest.txt`。24 项里 `test_family_{mistral,gemma,gemma2,qwen3next}`
分别 11.40 / 11.27 / 11.11 / 51.84s（qemu 下），全 Passed。

### 过程中的一次「假失败」（我自己的脚本问题，已排除）

第一遍有 2 项失败：`test_lin_seq` 与 `test_family_qwen3next`，报
`tensor 'model.layers.3.self_attn.q_proj.weight' shape[0]=64 != expected 128`。

查明是**我的复刻脚本不对**，不是产品问题：workflow 会生成**全部 5 个族**，我为了省 qemu 时间
只生成了 2 个，其余靠 `cp -a /srcro/.` 从仓库工作目录拷——而 `sglang-c/models/qwen3next-rand`
是 **G13 之前的旧夹具**（q_proj 宽 64），新代码按 `2*q_dim`（=128）读，于是报形状不符。
按 workflow 补齐 5 个生成后重跑：**24/24 全过**。

顺带记一个**本地陷阱**（CI 不受影响）：`sglang-c/models/` 里还留着 G13 之前的
`qwen3next-rand`（q_proj `[64,64]`，真模型在 `../models/qwen3next-rand` 是 `[128,64]`）。
它已被 `.gitignore` 忽略，所以干净检出/CI 都没有；但如果有人只拿 `./models` 当模型目录
本地构建，`test_family_qwen3next`/`test_lin_seq` 会以「layer 3 weights missing / 形状不符」
的形式假红。用 `tests/gen_random_qwen3next.py --out models/qwen3next-rand` 重生即可。

### 仍未验证

- `macos-arm64` / `macos-x64`：Windows 上没有 macOS，无法等价复现（`macos-x64` 还牵扯
  `-DCMAKE_OSX_ARCHITECTURES=x86_64` 交叉编译 + Rosetta 转译）。
- `release` job：需 push tag，属对外发布，未做。

## Linux job 实跑（第 13 轮，2026-10-27）

之前我写「本机没有 Linux 环境」——**不准确**。WSL 里只有 `docker-desktop`（非通用发行版），但 Docker 可用（linux/amd64 29.6.2），于是能用 Ubuntu 24.04 容器把
`linux-x64` 矩阵项真跑一遍。

做法：容器里 `apt-get install build-essential cmake ninja-build python3 python3-pip`，
源码从只读挂载复制到 `/work`，**逐字**执行 workflow 的生成步 → 配置 → 编译 → Test 步 → gate。
只有两处替代：`models/gpt2/model.safetensors`（520MB）与 `models/qwen3-06` 的 4 个 tokenizer
文件改用本机同名文件的软链（CI 是 `curl` HuggingFace，内容相同）。

### 结果

| 项 | 结果 |
|---|---|
| 生成 5 个随机族 + `ref_llava.py` | 全部成功（与 msvc 那条输出逐字一致，proj_checksum 1.072851） |
| `cmake -G Ninja` + `cmake --build build -j` | BUILD_OK |
| `ctest --test-dir build -V --timeout 900` | **100% tests passed, 0 failed out of 24**（176s） |
| G19 gate | **counts: fam=4 fam_ran=4 llava=1 → LINUX GATE: PASS** |

24 项里有 `test_family_{mistral,gemma,gemma2,qwen3next}` 与 `test_llava`，说明模型族与
多模态在 Linux 上也真跑了（不是 SKIP）。`test_llama: all OK` 出现 4 次。

### 又抓到一个我自己引入的 gate bug（已修）

第 10 轮我把 `|| true` 改成 `|| echo 0`，本意是防「日志缺失 → 变量空 → `[ \"\" -lt 4 ]` 报错
→ if 当假 → 静默通过」。但**这个改法在「无匹配」这个更常见的情形下反而坏了**：
`grep -c` 无匹配时会**先打印 `0` 再返回 1**，于是 `|| echo 0` 又追加一个 `0`，变量成了两行的
`\"0\
0\"` → `[ \"0\
0\" -lt 4 ]` 报 integer 错 → `if` 当假 → **gate 静默通过**。
也就是说：**某个 family 测试真回归的时候，gate 恰好拦不住** —— 正是它存在的意义。

修法（`.github/workflows/build.yml` 147-157）：

- 计数改成 `grep -c ... | head -1 || true`（`head -1` 保证只取一个数字，多行被截断）；
- 补 `${var:-0}` 兜空；
- 再加一道 `case \"$fam$fam_ran$llava\" in *[!0-9]*) exit 1 ;;` 的非数字硬失败，
  任何意外（多行/空/乱码）都只能让 gate 变严、不能变宽。

验证（在容器里、**用 GitHub Actions 真实的 `bash -e -o pipefail` 旗标**）：

| 情形 | 期望 | 实测 |
|---|---|---|
| A 正常（4 族 Passed + 4 条 all-OK + llava OK） | 0 | **exit 0**（不会假红） |
| B 一个族回归（3 个 Passed、无 all-OK 行） | 1 | **exit 1** |
| C 日志不存在 | 1 | **exit 1** |
| D 日志里是非数字内容 | 1 | **exit 1** |

### 一个没能复现的现象（如实记录，不当结论）

用 `docker run --rm` 跑整套时，PowerShell 终端报过一次 `退出码 1`，但输出里 gate 明确是
`LINUX GATE: PASS`。改用常驻容器 + `docker exec` 复跑同一份脚本（并在脚本里逐步打印
`ctest|tee pipeline rc=$?`），得到 **rc=0、脚本自然退出 0**。所以：

- 能用工具确认的是：**gate 在通过时返回 0，脚本自然退出码 0**；
- `--rm` 那次报的 1 无法复现，怀疑是我自己的 PowerShell 包装层（`Tee-Object` 管道）在
  容器退出后取到的状态，但**没有证据，所以不写成 CI 的问题**。

### 仍未验证的

- `linux-arm64`、`macos-arm64`、`macos-x64` 三个矩阵项：本机没有 arm64 容器/macOS，
  跑不了（`macos-x64` 走的是 `-DCMAKE_OSX_ARCHITECTURES=x86_64 -DCMAKE_C_FLAGS=-march=x86-64-v2`
  交叉编译 + Rosetta，Windows 上无法等价复现）。
- `release` job：要 push tag，属对外发布，未做。

## msvc-cuda 矩阵项的断言本机复核（第 12 轮，2026-10-27）

前两轮只跑了 workflow 的 `msvc` job，`msvc-cuda` 那条从未验证。它在无 GPU 的 runner
上不跑 ctest，只做两件事，都能在本机用现成产物复核：

1. **打包步的 DLL 断言**：`Get-ChildItem $env:CUDA_PATH -Recurse -File -Include "cublas64_*.dll",
   "cublasLt64_*.dll"` 必须找到 ≥2 个，否则 throw。本机
   `CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3` → 找到 **2** 个：
   `bin\x64\cublas64_13.dll`（50.3MB）与 `bin\x64\cublasLt64_13.dll`（442.2MB）。
   递归查找确实是必要的——CUDA 13 把 DLL 放在 `bin\x64\` 子目录而不是 `bin\`。
2. **smoke 步**：要求 `build/bin/{cuda_cli,cuda_http,sglc}.exe` 都在，且 `sglc.exe` 无参
   退出码为 1。本机（VS 生成器，产物在 `bin\Release\`）三个 exe 均存在，
   `sglc.exe` 无参退出码 **= 1**，usage 文本确实打到 stderr（首行是 `usage: ... --model <dir>`）。

未验证的只剩：`linux-arm64` / `macos-arm64` / `macos-x64` 三个矩阵项（本机没有 arm64
容器与 macOS），以及真正打 tag 时的 `release` job（需 push tag，属对外发布，未做）。
~~[已推翻] `linux-x64` 当时写的是「本机无 Linux 环境」~~ → **第 13 轮用 Docker(Ubuntu 24.04)
真跑了 `linux-x64`：24/24 通过、gate PASS**，见上一节。

## README 对账（第 11 轮，2026-10-27）

发布质量（t13 要求 README/CHANGELOG 准确）直接吊在 README 上，所以把 README 逐句对了一遍
代码，改掉四处过期/误导说法（提交 `3180306`）：

| 原来的说法 | 实际 |
|---|---|
| LLaVA 行：“raw float base64；JPEG 解码不支持” | G11 已能直收 PNG/JPEG（同 CPU 版路径） |
| 能力表没有 Qwen3-Next | G13 已支持（q 门控/partial rope/(1+w)/MoE） |
| 未覆盖：真实 LLaVA 权重“仅结构已验证” | CLIP-L 与 mm_projector 已用真实权重对照，缺的只是 7B LM |
| M26：“≤4096 走 short 核、>4096 切两段式” | G1→长行走 flash，G16 又降到 64 tok；两段核默认只在 `SGL_ATTN_TWOPASS=1` 或无特化 D 时用 |

另外两处顺手修正：

1. **G1 的 8.3x 不能当头条**：那是显存吃紧时测的，BENCH 已写「两个数字不要互相引用」。
   README 现在两个都给并注明条件（8.3x 当时 / 23.6x 干净同进程 A/B）。
2. **核实过程中发现 GPT-2 不走 `attn_dispatch`**（它有自己的前向循环，仍然直接启
   shared-scores 核），所以“kernel 选择集中在一处”这句话只能限 llama 族；已在 README
   里限定，并在「已知问题」里记下 GPT-2 没拿到 G1/G16 的长上下文优化。

补了一节「缺口关闭 G11–G19」（与已有的 G1–G6 节同格），并给「已知问题」加了四条真限制：
MSVC 无 OpenMP（只值 2.05x）、前缀缓存是段级非 radix 树且多图仍共用一个 `<image>` 位、
CI 不跑 CUDA 与真实 llava 权重、GPT-2 缺新 attention 路径。

新写进 README 的每条事实都 grep 回代码核过（`SGL_*` 开关、`D % 32` 守卫、`img_at`、
`sgl_projector_init`、`moe_row`、`sgl_dot_f32_avx2`），表格改完也重数了每行的 `|` 个数。

## G19 补记：从**干净检出**按 CI 步骤实跑（2026-10-27，第 10 轮）

之前只验证了「生成器产出与夹具逐字节相同」和「YAML 能解析」，**没验证过那串步骤在 fresh
clone 上真能跑通**——而且夹具现在被 .gitignore 了，CI 必须自己生成。这一轮补上了：

做法：`git clone sglang-c` 到临时目录（138 个文件，`models/` 里只有 gpt2 的 config/tokenizer），
把 `models/qwen3-06` 的 4 个 tokenizer 文件 + `models/gpt2/model.safetensors` 用本地副本代替
CI 的 curl（**只替换下载，不动任何命令**），然后**逐字**执行 workflow 里 `msvc` job 的：

1. 生成步：`gen_random_{mistral,gemma,gemma2,qwen3next,llava}.py` + `ref_llava.py` → 全部成功，5 个族目录都在；
2. 配置 + 编译（Ninja + MSVC，vcvars64）：rc=0；`ctest -N` 注册 **24** 项，其中
   `test_family_{mistral,gemma,gemma2,qwen3next}` 4 项 + `test_llava` 均在列（qwen3 族需真实
   权重，CI 不下，故不注册——判据只要求 ≥4）；
3. `ctest --test-dir build -V` → **100% tests passed out of 24**；
4. 把 workflow 里那段验收 gate 抽出来用 bash 跑（与 `shell: bash` 同一套命令）→
   `family runs all-OK: 4, test_family_* Passed: 4, test_llava all-OK: 1` → **CI GATE: PASS**。

### 顺带发现并修掉的两个真问题

1. **`tests/_ref_llava.json` 已过期**（第 9 轮之前就存在，这一轮才发现）：它在仓库里但
   **没有任何代码读它**（真正被测的是 `ref_llava.py` 写进模型目录的 `_ref_{feats,proj}.npy`），
   而 G12 把 `gen_random_llava.py` 的默认 projector 改成两层 MLP 后就与生成结果不一致了
   （`proj_checksum` 1.18944 vs 1.07285，且缺 `projector: "mlp"` 字段）。按文档要求
   「`_ref_*.json` 必须与被生成的权重一致」已从干净检出的产物重生并提交。
2. **CI 验收 gate 自己有两个隐患**（都是这轮模拟时踩到的）：
   - 原来用 `grep -c 'greedy6=一致'`（中文）——能不能匹配取决于测试二进制的执行字符集。
     本机实测 MSVC + `/utf-8` 下确实是 UTF-8 字节，所以今天能工作；但换个不带 `/utf-8`
     的工具链就会变成系统代码页而默默匹配不到。已改成纯 ASCII 判据
     `grep -c 'test_llama: all OK'`（语义还更准：整轮零失败）。
   - 原来写的是 `|| true`：`ctest.log` 不存在时 grep 什么都不输出，变量为空 →
     `[ "" -lt 4 ]` 报 integer 错、被 `if` 当成假 → **gate 静默通过**。已改为
     `2>/dev/null || echo 0`。反向验证：把 ctest.log 改名后跑，现在正确输出
     `::error::fam_ran=0` 并 exit 1（改之前是 exit 0）。

临时克隆已删除，磁盘回到 3.7GB。

## 最终验收（v0.4 清零后的全量重跑，2026-10-27）

两个构建都从当前 HEAD 重编重跑，四项全绿：

| 验收项 | 命令 | 结果 |
|---|---|---|
| CPU 全量 ctest | `ctest --test-dir build --timeout 900` | **26/26 通过**（含 test_lin_seq / test_g12_place / test_llava_real / test_family_* 5 族） |
| GPU 全量 cuda_test | `cuda_test ..\models\qwen05` | **rc=0，test_cuda: all OK**；prefill CPU-vs-GPU 2.15e-05、[w8] imma==dp4a 0、[batch] 291.8 tok/s 一致、[long] 三路全 SAME、[prefix] 2.1x 且 6/6 SAME、[mm] 1.19e-07、[spec] 两例逐 token 等于非 spec |
| 前缀缓存 e2e | `tests/prefix_e2e.ps1` | **4 组 ALL PASS**（同前缀 552 行 / 续写 573 行 / 带图 text 130 行 / 超容量 flushes=1） |
| 多模态 e2e | `tests/http_image_e2e.py` | **E2E OK**（jpeg/png/text 全 200，图不同则输出不同，坏 base64 → 400） |

顺带修掉了一个一直让 `git status` 不准的噪音：`.gitignore` 以前只忽略 `models/*/model.safetensors`，
于是生成出来的 7 个夹具目录（*-rand / mm-test / g12-mlp / d96,d80-rand / llava-mlp）里的
config.json/tokenizer.json 全部显示为 untracked。现已整目录忽略（`models/gpt2` 仍跟踪，CI 需要它），
夹具仍在磁盘上、不影响任何测试。

## 现状（v0.4 已清零）

- TODO_PROMPTS v0.4 的 **G11–G19 全部完成**（每节都有实测数字和日志路径）。
- 未做的三项都是文档里允许的可选项，且有实测依据：
  - **G12 的 7B LM 端到端**：llava-1.5-7B 的 LM 是 fp16 13.5GB，本机磁盘只剩 3.4GB；
    已用任务允许的「分段对照」覆盖（真实 CLIP-L/14-336 + 真实 mm_projector）。
  - **G13 的 CUDA 路径**：任务写明「CUDA 可选」（cuda_backend.cu 只有 llama 族 attention）。
  - **G15 的跳段共享两轮扫描**：扁平表上可复用行数的上界就是最长的一条登记，拼接永远
    不可能更长（已论证 + 实测无收益），改用「登记生成 token 行」拿到真实收益。
- 以后的缺口写进 TODO_PROMPTS v0.5。