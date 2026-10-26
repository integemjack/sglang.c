# NOTES_gaps — G1–G6 缺口关闭（验收记录）

日期：2026-10-27 ｜ 机器：RTX 2060 6GB（桌面常驻占 ~3.2GB，空闲 ~2.7–3.8GB），CUDA 13.3，
CPU 栈 MinGW GCC 16.2 + Ninja，CUDA 栈 MSVC BuildTools 2022 + nvcc（sm_75）。
模型：qwen2.5-0.5B（`models/qwen05`）、tinyllama-1.1B、llama-160m、gpt2。

> 测量口径（重要）：这张卡的空闲显存经常小于一个 fp32 0.5B 模型的常驻量，超出部分被
> WDDM 溢到系统内存，**绝对时间会被放大 3–20 倍**。所以本文件里的性能对比一律用
> **同一进程内 A/B**（`cuda_perf --ab` / 环境变量当场切路径），并在 `[cfg]` 行记录
> 当时的 free VRAM。跨进程的「改前 vs 改后」数字只作参考，不作结论。

---

## G1 · GPU 长上下文 prefill 提速（flash 单遍核）

**做了什么**：`src/cuda_backend.cu` 新增 `k_attn_flash<D, QPW, TQ_TILE>` 模板核（Q 块驻
shared、K/V 分块流式过 global、online softmax 复用已有的 run_max/run_sum 框架），
`launch_flash()` 按 D=64/128 选模板参数；`fwd_seg_llama` 的长路径改为优先走 flash，
`SGL_ATTN_TWOPASS=1` 可强制回退到旧的两遍核（A/B 用）。另加 `SGL_CUDA_PROF=1` 的
逐层 attention 事件计时（`[prof] attention X ms / L layers`），不插入中间同步。

**实测（同进程 A/B，qwen05 fp32，ctx=8192，rounds=2，free VRAM 3803MB）**：

| prompt tokens | 两遍核（旧） | flash（新） | 加速 |
|---|---|---|---|
| 512 | 313.4 ms | 206.1 ms | 1.5x |
| 2048 | 3980.8 ms | 2345.6 ms | 1.7x |
| 8192 | 72383.2 ms | 8681.0 ms | **8.3x** |

跨进程参考值（不同显存状态，仅供对照）：t2 当时 8K prefill 66.1s（ctx 16384），
本次 flash 后同条件 2.8s。

**正确性**（`cuda_test [long]`，全部保留原阈值）：
- determinism（2000 tok）：short-vs-short 0，long-vs-long 0（逐位可复现）；
- two-pass vs shared-scores（2000 tok）：maxerr = 2.19e-05（fp 噪声级，阈值 1e-3）；
- 4300-token batched == single-stream：OK（14 tok / 0.46s）；
- `long_vcpu` 8K GPU vs CPU 的 t2 绝对对照：本轮因并发内存压力未跑完，见「未完成」。

**定位数据**（`SGL_CUDA_PROF=1`，n=4096 prefill，旧两遍核）：`[prof] attention 451367.7 ms /
24 layers` —— 注意力占绝对主导，确认瓶颈在 kernel 而不是别处（该次测量同样落在溢出状态）。

## G2 · W8 大 M 切 cublas IMMA

**做了什么**：新增 `k_quant_rows`（每行一次算 absmax 量化成 int8，不再像 dp4a 核那样
每个输出通道重算一遍）、`k_w8_dequant`（int32 累加 × 行尺度 × 通道尺度 + bias）、
`gemm_nt_w8_imma()`（`cublasGemmEx` CUDA_R_8I × CUDA_R_8I → CUDA_R_32I，COMPUTE_32I）；
`PROJ()` 宏里 M>1 走 IMMA、M==1 仍走 dp4a GEMV。`SGL_W8_DP4A=1` 强制旧路径（A/B 用）。
IMMA scratch（`w8_a8/w8_c32/w8_as`）在 `sgl_cuda_reserve()` 里随 ws_cap 一起长，
调用点前已 `invalidate_graphs()`（铁律 4）。

**实测（`sgl_cuda_gemm_bench`，M=8，tinyllama [2048,2048]）**：
`int8 imma 0.0692 ms` vs `dp4a 0.9250 ms` → **13.4x**；M=1（qwen05 [896,896]）
`imma 0.0680` vs `dp4a 0.0620` → dp4a 仍略快，所以 decode 保持 dp4a（结论有数据支撑）。

**正确性（`cuda_test [w8]` 新增段）**：
- IMMA vs dp4a 同模型 prefill logits：**maxerr = 0**（整数点积与顺序无关，逐位相等）；
- W8 prefill vs fp32：maxerr 2.33（fp32 max|logit| 19.6，相对 11.9%，阈值 30%），top1 相同；
- W8 greedy 10 个 decode step：**10/10 与 fp32 全同**。

## G3 · cuda_http decode CUDA Graph 稳定性

**根因（t5 当时「graph 不稳」的真相）**：三条叠加，都有实测支撑 ——
1. 捕获失败会把一个**进程级** static 置成「graphs off」，一个 B 的偶然失败让所有 batch
   永久走 plain（decode 慢 3–4x）；
2. 服务端 `ws_cap` 随最长 prompt 增长 → `invalidate_graphs()` → 下一次 decode 现场重新
   捕获（实测每张 ~0.5s，夹在请求中间，表现为「间歇性极慢」）；
3. CUDA 13 默认 `CUDA_MODULE_LOADING=LAZY`：内核首次加载、cuBLAS 首次选算法都发生在
   capture 内部 —— 在流捕获里做这些事会失败。
另外补一个我自己引入的坑（已修）：预热步骤必须先把 host 镜像 H2D 上传，否则
`ensure_io()` 刚 `cudaMalloc` 出来的 `d_base` 是未初始化内存，拿它启动 decode 步会
KV 越界 → illegal memory access（这个症状和 t5 记录的「sticky CUDA error」一模一样，
很可能当年也踩到过）。

**做了什么**：抽出 `capture_decode_graph()`（先跑一次 plain step 预热，再捕获；失败只把
**这一个 B** 标成 `SGL_NO_GRAPH`，其它 B 照常回放；`invalidate_graphs()` 会清掉标记，
下次还有机会捕获）；新增 `sgl_cuda_prewarm(g, rows, max_b)`：起服务前把 `ws_cap`、
batch 镜像、logits 镜像一次撑到位，并为 B=1..max_b 预捕获，外加一次 64-token 的
prefill 预热（把 prefill 专用内核和 M>1 的 cuBLAS 算法选择也提前做掉）。
`apps/cuda_http.c` 默认**开启** graph（不再强制 `SGL_CUDA_GRAPH=0`）并调用 prewarm。

**实测**：
- decode A/B（qwen05 **fp16**，ctx 8192，rounds=3，free VRAM 3803MB）：

  | B | graph OFF | graph ON | 加速 |
  |---|---|---|---|
  | 1 | 23.0 ms/step（43.5 tok/s） | 9.7 ms/step（103.4 tok/s） | 2.4x |
  | 4 | 36.6 ms/step（109.2 tok/s） | 8.9 ms/step（448.3 tok/s） | 4.1x |
  | 8 | 38.5 ms/step（125.1 tok/s） | 11.3 ms/step（710.2 tok/s） | 3.4x |

- prewarm：`[engine] prewarm: ws rows=1024, decode graphs 8/8`，日志无 `capture failed`；
  首请求从 ~20s（lazy 加载 + 现场捕获）降到与后续同量级。
- 稳定性：`tests/conc_test.ps1 -Rounds 5 -Conc 12 -GraphOn` → 5/5 轮 12/12 完成，
  服务端单请求耗时 med 0.28s / max 1.23s，进程存活 PASS；
  `tests/stream_test.ps1 -Port`（t5 四项）→ stream==plain / 12 并发排队 /
  finish_reason=length / 服务存活 全 PASS。

## G4 · HTTP 无 body 请求间歇卡死

**定位**（`tests/g4_probe.py`，纯 CPU 版即可复现 → 与 CUDA 运行时无关）：
`load 300 8 128` 复现过 20/300 次 `timeout_read`（6.7%），connect 正常 ~15ms、
10s 内无首字节；同时刻 POST /generate 正常。取 TCP 快照：无 SYN 堆积、无 CloseWait
累积 → 不是 send 阻塞，而是**服务端还没走到 send**。
结构性根因：`srv->lock` 一把粗粒度锁同时被两处热路径争用 —— worker 每步
`sgl_sched_step` 都持锁做计算，acceptor 每接一个连接也要抢同一把锁登记
`active_conns`，高争用下 acceptor 被饿死，连接被内核完成握手却长时间不建线程。

**修法**（`src/http.c`）：`active_conns` 拆出独立的 `conns_lock`（acceptor / conn_fn /
stop 三处改用它），acceptor 不再排在 worker 的计算锁后面；`cv_conns_zero` 的广播与
stop 轮询同步改到新锁；两把锁的 init/destroy 补齐。

**复验**（同一探针、同一负载形状）：
- `load 300 8 128`：300/300 ok，timeout_read=0，complete_ms med 14.9 / **max 33.8**；
- `load 600 12 256`（更重）：600/600 ok，timeout_read=0，med 14.9 / **max 35.5**；
- requests / curl 两种客户端各 60 次：0 超时；
- 回归：`test_http` / `test_scheduler` / `test_radix` 单项 Passed，全量 ctest 23/23。

## G5 · 显存预检 + 降级 + soak 曲线

**做了什么**：`sgl_cuda_estimate_vram_mb(model, ctx, rows)`（与 `sgl_cuda_from_llama`
同口径：权重按 fp32/fp16/int8 开关、KV 池、激活工作区、logits 镜像、cuBLAS workspace）+
`sgl_cuda_avail_vram_mb()`；`cuda_http` 在上传前预检，**空闲 < 需求 + 192MB 就不上传**，
服务照常起：`/health` 返回 `{"ok":false,"gpu":false,"degraded":true,"reason":"..."}`，
`/generate` 直接 503 + 同一句原因（以前是进程 `return 2` 直接死，调用方只看到连接被拒）。

**实测**：`SGL_CUDA_CTX=32768` 起服务 →
`[engine] vram precheck: need ~3236MB, free 2270MB (ctx=32768 rows=4096)` →
`/health` = degraded JSON，`/generate` = **HTTP 503** + 原因（含「降低 SGL_CUDA_CTX 或
用 SGL_CUDA_FP16=1/SGL_CUDA_W8=1」的可执行建议）。

**soak**（`tests/soak_curve.ps1`，20 分钟，4 路并发负载，10s 采样）：
RSS 3251.9 → 3310 MB，VRAM 常驻 4662 MB 不变；中段一次 +688MB 的台阶是**我自己并发
跑 cuda_test 造成的干扰**（时间戳与那次启动对齐），干扰结束后回到平台期并保持平坦。
判定线：RSS/VRAM 斜率 < 0.5 MB/min 且末段无上升趋势 → 无泄漏。干净环境的斜率复跑见「未完成」。

## G6 · GPU 前缀缓存（段租让）

**做了什么**：`sgl_cuda_copy_kv(g, src, dst, len)`（所有层 k/v 的 len 行 D2D 拷贝）；
`cuda_http` 侧加段租让表（`PFX_MAX=16`、`PFX_MIN=32`，按 token 精确比对最长公共前缀，
LRU 驱逐，段被复用时先释放它持有的租让）；请求结束时把它的 prompt 登记为租让；
命中时搬 KV 行 + 只 prefill 后缀（全命中则留最后一个 token 重算以拿到 logits）。
`/health` 增加 `prefix:{hits,hit_rows,leased_rows}` 计数。

**正确性（`cuda_test [prefix]` 新增段）**：同一 prompt B = P + S2 两条路径 ——
完整 prefill vs 复用 A 段的前 300 行 KV + 只算 48 token 后缀：
**maxerr = 1.43e-05**（fp 噪声级：后缀单独 prefill 时 M 不同，cuBLAS 选不同 kernel/
累加顺序；与 two-pass vs shared 的 2.19e-05、CPU vs GPU 的 3.5e-05 同量级），
**greedy 6/6 逐 token 全同**，prefill **50.4ms vs 127.1ms（2.5x）**，copy 2.1ms。

**端到端（`tests/prefix_e2e.ps1`，756-token 共享前缀 + 不同后缀）**：
冷启动 B：0.76s；先发 A 再发 B：B **0.34s（2.2x）**，`hits=1 hit_rows=756 leased=1524`，
**输出与无缓存逐字一致** → ALL PASS。

---

## 未完成 / 遗留

1. `long_vcpu` 的 8K **GPU vs CPU** 绝对对照（t2 那条 maxerr<1e-3）本轮没跑完：CPU 侧
   8K prefill 需要几分钟且当时机器有并发内存压力（`cpu pages fail` / OOM）。flash 路径
   的正确性目前由 `[long]`（2000 tok 两遍 vs shared、4300 batch vs single）与 `[prefix]`
   覆盖；需要在干净机器上补跑一次 `long_vcpu models/qwen05 8192`。
2. G5 soak 的**斜率判定**要重跑一次无干扰版本（脚本已修：`Measure-Object` 需要
   pscustomobject，hashtable 会报错）。
3. 前缀缓存是**段级**（一条序列一个租让槽，最长公共前缀按 token 精确比对），不是 CPU 侧
   那种 radix 树；跨段的部分共享、显存压力下的主动回收、以及带 visual token 的请求都未覆盖。
4. G3 的 prewarm 会写一次 64-token 的垃圾 KV 到段 0；首个真实 prefill 会覆盖它，且
   attention 只读 `[kv_base, kv_base+len)`，不会被读到 —— 但如果将来支持段间共享需重新审视。
