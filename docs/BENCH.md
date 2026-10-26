# sglang-c 性能快照 (BENCH)

行 schema: 栈(cpu/cuda) | 精度(fp32/fp16/w8/int8) | 模型 | 场景(single/batch) | tok/s | 显存峰值 MB | 备注

由 `sglang-c/bench_all.ps1` 维护，每次运行整表重写（幂等）。固定 prompt 与 max-new，每项跑 2 次取较好值。
CUDA fp16 / int8 行：设环境变量 `SGL_CUDA_FP16=1` / `SGL_CUDA_W8=1` 后重跑本脚本即出数。显存峰值由 nvidia-smi 每 120ms 采样（仅 cuda 行）。

## 最新快照 (2026-10-27 终版，三轮实测取优：fp32 基线 / SGL_CUDA_FP16=1 / SGL_CUDA_W8=1；同 prompt 同参数 greedy 40 tok best of 2，GPU 空载时测)

CUDA 三精度：qwen05 单流 fp32 33.5 / fp16 36.3 / int8(W8A8 dp4a) 33.8 tok/s（decode 带宽瓶颈下持平）；
显存（cuda_cli 上传后 free VRAM）：fp32 剩 216MB / fp16 剩 3611MB / **int8 剩 3947MB**（最省：投影 0.55GB + wte fp16）；
tinyllama-1.1B fp16 39.4 tok/s（~2.2GB 权重装入 6GB 卡；GPU 被后台服务占用时会降到 ~28）。
W8 数值验收：16 步 top1 与 fp32 全同、gap 偏差±0.1 内无翻转（NOTES_t8.md）。
int8 IMMA vs dp4a：M=1 持平（选 dp4a）；**M≥2 已切 IMMA**，拐点与 prefill 实测见下方 G16 节。

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
**G14 后更新**：结构成本已改掉（流水线去掉每轮 target 图步 + small-M GEMV），
K=1 0.44x→0.73x、K=2 0.35x→0.64x、K=4 0.27x→0.40x；按实测回本需 K=1 a>88% /
K=2 a>69.5%，而实测只有 39.1% / 28.6% —— **仍未转正**，根因就是 draft 太弱（见下）。

## G15：前缀缓存（租让表）扩到「生成 token + 带图 + 可关」 （2026-10-27）

工具：`tests/prefix_e2e.ps1`（四组全 PASS）、`/health` 的 `prefix` 字段。
模型 qwen05（ctx=4096，conc=2，seg=2048），prompt = 40 句式 ≈ 558 token。

### 端到端（同 prompt 重发 vs 冷启）

| 场景 | 耗时 | 命中行数 |
|---|---|---|
| 冷启（无租让） | 0.262s | — |
| 完全同 prompt 重发 | **0.103s（2.5x）** | 552 / 558 行（单轮 prefill 行数的 99%） |
| 同前缀不同后缀 | 0.103s | 552 行 |
| 多轮续写（prompt = 上轮 prompt+生成结果+新后缀） | 0.103s | **573 行**（多 21 行 = 生成 token 的 KV） |

### 四组验收（prefix_e2e.ps1，全部输出与「SGL_PFX_MAX=0 关缓存」逐字一致）

| 组 | 内容 | 实测 |
|---|---|---|
| 1 同前缀 | A=前缀+S1 → B=前缀+S2 | 命中 552 行，leased_rows=565 |
| 2 续写 | A=前缀+S1（生成 16 tok）→ B2=前缀+S1+A 的结果+S2 | 命中 **573** 行 > 同前缀的 552（多的 21 行就是以前不登记的生成 token 行） |
| 3 带图 | 同文本、不同图片的两条视觉请求 | 命中 **135 行 text 部分**（以前带图请求完全不登记 ⇒ 0） |
| 4 显存紧张 | ctx=512/conc=8 ⇒ seg=64，发 200+ token 的 prompt | 请求被拒、`flushes=1`、随后小请求仍与关缓存逐字相同 |

### 做了什么

1. **租让的行数从「只有 prompt」改成「这段里真有有效 KV 的行」**：非视觉请求登记
   `n_prompt + (n_gen-1)`（最后一个生成 token 的那行不保证写过，保守少算一行）；
   典型的多轮对话可以整段复用上一轮。实测 552 → 573 行。
2. **带图请求也登记**（只到 text 末）：视觉行依赖图片、位置在 text 之后，不归这段
   登记管；所以同文本不同图片的请求能复用 text 部分（135 行），同文本同图片还能
   接着复用视觉行以外的部分。
3. **`/health` 新增字段**：`slots / hits / misses / hit_rate / hit_rows / leased_rows /
   flushes`（leased_rows 是 gauge：当前可复用行数）。
4. **运维杆 `SGL_PFX_MAX`**（0 = 彻底关缓存，1..16 = 限表大小）与 `pfx_flush_all()`
   （请求超过一段容量时主动丢掉全部租让，避免后续命中半截现场）。

### 关于「跳段共享（两轮扫描）」——实测无收益，未做

扁平表里每条登记都从第 0 行开始，所以一次请求能复用的行数**上界就是所有登记中
公共前缀最长的那条**；跨两个段拼接永远不可能更长（短的必然被长的那份覆盖）。
所以没去实现“两轮扫描”，而是把真正能多拿行数的方向（生成 token 的行）做了。
另外：本服务的段是按并发数**静态划分**的，租让不额外占显存（`leased_rows` 只是
可复用行数的计数，不是额外分配），所以「显存压力下回收租让」在这套结构里
退化为「超容量请求时把现场清干净」；已在组 4 验。

## G14：投机解码流水线化 + small-M GEMV（2026-10-27）

工具：`cuda_perf --spec <draft> --spec-k K`，`SGL_SPEC_PROF=1` 打各阶段耗时。

### 成本分解（tinyllama fp16 + llama-160m，ctx=1024，K=4，**改前**）

| 阶段 | ms/轮 | 占比 |
|---|---|---|
| draft 4 个图步 | 13.4 | 24% |
| verify（n=4，plain） | 25.8 | 46% |
| target 补齐 1 次（图步） | 12.8 | 23% |
| draft 补齐 1 次 | 3.9 | 7% |

### 两处改动

1. **small-M GEMV（`k_gemm_nt_mrow` / `k_gemm_nt_h_mrow`）**：M=2..8 时每输出行
   一个 block，权重行只读一遍，同时累 MT 个 query 行（MT 用模板实例化成 2/4/8）。
   踩到的两个坑：① 动态 M 的累加器落 local memory → M=2 比 M=1 还慢（22.1 vs 14.6ms）；
   ② 旧的 MT 轮同步归约让 n=5..8 比 n=4 慢（33ms vs 18.5ms），改 warp shuffle +
   单次同步后回到 27.7ms；③ MT=8 时不能读 A 的 i≥M 行（A 只分配 M 行）。
   verify 实测：n=1 14.6ms、n=2 16.1ms、n=3 17.2ms、n=4 18.5ms、n=5 27.7ms
   （改前分别是 14.6 / 22.1 / 24.0 / 25.8 / 33ms）。
2. **流水线（去掉每轮一次 target 图步）**：预填 `np-1` 行，prompt 末 token 当“挂起
   token”由下一轮 verify 第 0 行吞掉，于是每轮 verify 固定 n=K+1，reject/bonus
   的 KV 顺带被写掉。

### 结果（prompt 16 tok、new 32、best-of-3；plain = 82.4–82.8 tok/s）

| K | 接受率 | 改前 | 改后 | 每轮成本 |
|---|---|---|---|---|
| 1 | 39.1% | 0.44x（35.1 tok/s） | **0.73x（60.5 tok/s）** | 2×3.35 + 14.7 = 21.4 ms |
| 2 | 28.6% | 0.35x（28.6 tok/s） | **0.64x（53.1 tok/s）** | 3×3.35 + 17.2 = 27.2 ms |
| 4 | 14.8% | 0.27x（22.0 tok/s） | **0.40x（33.3 tok/s）** | 5×3.35 + 27.7 = 44.5 ms |

**结论：仍净减速（未达到 G14 的 ≥1.0x 门槛）。** 回本条件是
`(1 + K·a)·t > (K+1)·d + v`，代入 t=11.4ms、d=3.35ms：K=1 需 a>88%、K=2 需 a>69.5%、
K=4 需 a>72%，而实测 a 只有 39.1% / 28.6% / 14.8%。根因是 160M draft 的图步也有
3.35ms 固定开销（数学只占 ~1.9ms），d/t 仅 0.29，必须靠 70–90% 的接受率才回本 ——
那需要同源蒸馏 draft。详见 docs/NOTES_t7.md 的 G14 补记。

正确性（`cuda_test ..\models\qwen05` → `test_cuda: all OK`）：
`[spec] qwen05 draft==target 接受率 100% (20/20)、24 tok == 非 spec greedy ✓`；
`[spec] tinyllama+llama-160m 接受率 13.3% (8/60)、24 tok == 非 spec greedy ✓`
（接受率与改前完全一致，说明流水线没有改变输出语义）；`[batch]`（B=4 现在走
small-M GEMV）batched == single-stream OK；`[long]`/`[w8]`/`[fp16]`/`[gpt2]` 无回归。



工具：`tests/bench_prefill.ps1`（qwen05，ctx=16384，rounds=2 取优，GPU 空载）。

### 1) 三精度 prefill 绝对值（n = 512 / 2048 / 8192）

| 精度 | 512 | 2048 | 8192 | 备注 |
|---|---|---|---|---|
| fp32 | 85.4 ms（5994 tok/s） | 386.7 ms（5296 tok/s） | 2748.9 ms（2980 tok/s） | 默认路径（>64 tok 走 flash） |
| fp16 | 52.2 ms（9808 tok/s） | 198.4 ms（10324 tok/s） | 1971.9 ms（4154 tok/s） | SGL_CUDA_FP16=1 |
| int8 | 48.6 ms（10533 tok/s） | 184.5 ms（11100 tok/s） | 1923.8 ms（4258 tok/s） | SGL_CUDA_W8=1（M≥2 走 IMMA） |

同进程 A/B（fp32，`cuda_perf --ab`）：two-pass 383.2 / 3864.6 / 66501.5 ms vs
flash 44.8 / 200.0 / 1999.4 ms → **8.6x / 19.3x / 33.3x**。8192 行的倍率与 G1 记录的
8.3x 同量级（这次显存更松，两遍核 66.5s 而不是 72.4s）。

8192 行比 2048 慢 10 倍（2048→8192 只有 4x token）：flash 的 attention 本身仍是 O(n²)，
prefill 到 8K 时 attention 已经占主导（G1 的 8.3x 就是把 attention 从 66s 压到 2s 的结果）。

### 2) int8 dp4a vs IMMA：没有“拐点”，M≥2 全输

`cuda_perf --ab` 同进程切 `SGL_W8_DP4A`：

| n（=M） | dp4a | IMMA | 倍率 |
|---|---|---|---|
| 128 | 1570.6 ms | 33.6 ms | 46.7x |
| 256 | 3072.6 ms | 33.6 ms | 91.5x |
| 512 | 6177.2 ms | 40.8 ms | 151.6x |
| 1024 | 12376.4 ms | 79.7 ms | 155.3x |
| decode（M=1） | 0.0584 ms/GEMV | 0.0588 ms | 1.01x（持平，选 dp4a） |

dp4a 的耗时随 M 线性涨（每块独立重复量化同一行激活），所以除 M=1 外一直输，
而且越输越多（151–155x）。结论：**dp4a 只保留给 M=1 的 decode**，注释/选路已如此。

### 3) flash 泛化：head_dim 64/96/128 + 不支持时的回退

- 以前只有 D=64 / 128 两个特化，其它 head_dim 全部回退两遍核。现在新增 **D=96**
  特化（QPW=2、TK=32，shared 30.3KB）。内核要求 `D % 32 == 0`（ND=D/32），
  所以加了一道 `D % 32` 守卫：否则像 D=80 这种 32*2=64<80 会静默只算前 64 维。
- **D=96 数值验收**（`..\models\d96-rand`：L=2 H=2 KVH=1 E=192 D=96，tokenizer 用 qwen3-06）：

| n | flash vs two-pass | GPU(flash) vs CPU | greedy | 判定 |
|---|---|---|---|---|
| 37 | 2.53e-07 | 4.17e-07 | SAME | PASS（<5e-3） |
| 200 | 3.28e-07 | 5.33e-07 | SAME | PASS |
| 500 | 5.66e-07 | 8.42e-07 | SAME | PASS |

  命令：`SGL_FORCE_LONG_ATTN=1 SGL_ATTN_DEBUG=1 long_vcpu ..\models\d96-rand 200`
  （`SGL_ATTN_DEBUG=1` 会打出 `[attn] flash D=96 QPW=2 TK=32`，证明确实走了特化
  而不是静默回退）。
- **D=80（不整除 32）**：`..\models\d80-rand` 走回退 → `[attn] flash skipped: D=80
  not a multiple of 32 -> slow path`，GPU vs CPU maxerr 7.15e-07、greedy SAME。
  （加守卫前这个特化会算错：maxerr 0.83、greedy DIFF —— 所以这次没有把 80 当特化。）

### 4) 额外发现：flash 不只用于长行，short 核的阀值从 4096 降到 64

原设计 `<= 4096 行走 short 核（shared 分数），> 4096 走 flash`。实测（qwen05 int8，
同进程 best-of-3）：

| n | short 核 | flash | flash 快多少 |
|---|---|---|---|
| 16 / 32 | 31.7 / 34.3 ms | 31.7 / 34.5 ms | 持平 |
| 64 | 32.0 ms | 34.8 ms | short 快 8% |
| 128 | 44.1 ms | 33.4 ms | **1.3x** |
| 256 | 68.0 ms | 39.8 ms | **1.7x** |
| 512 | 169.1 ms | 46.2 ms | **3.7x** |
| 1024 | 583.0 ms | 75.7 ms | **7.7x** |
| 2048 | 2201.0 ms | 186.8 ms | **11.8x** |
| 4096 | 11445.3 ms | 567.6 ms | **20.2x** |

所以改成：`> 64 tok`（且 head_dim 有特化）就直接走 flash，`<= 64` 或没特化仍走
short（超过 4096 行再退两遍）。dispatch 逻辑从 prof/非 prof 两份抄写收成单个
`attn_dispatch()`。默认路径因此变快：fp32 512/2048 token 由 207.8/2385.9 ms →
44.8/200.0 ms（同进程 A/B 的 flash 列），int8 512/2048 由 169.1/2201 ms → 48.6/184.5 ms。
正确性：`cuda_test` 的 `[long]` 段现在三路都验 —— short(32) vs 两遍 1.08e-05 SAME、
default(2000=flash) vs 两遍 2.00e-05 SAME、flash vs 两遍（4300 行）1.75e-05 SAME，
加上 flash vs CPU（200 tok）2.62e-05 SAME；整套 `cuda_test ..\models\qwen05` 输出
`test_cuda: all OK`。

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
