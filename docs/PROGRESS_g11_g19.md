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