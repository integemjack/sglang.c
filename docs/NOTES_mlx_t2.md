# MLX 后端 T2：性能实测与定位

> 环境：iMac Apple M1 / 8 核 / 16 GB 统一内存 / macOS 27.0.1 / Metal 4，
> AppleClang 21，MLX 0.29.3（pip wheel），模型 Qwen2.5-0.5B-Instruct fp16
> （L=24 H=14 KVH=2 E=896 D=64 V=151936 tie=1，942 MB 权重）。
> 时间：2026-11-04。

## 结论先说

| 阶段 | decode B=1 | decode B=4（合计） |
|---|---|---|
| T1 完成时 | 69.0 ms/步 · 14.5 tok/s | 79.7 ms/步 · 50.2 tok/s |
| + `fast::rope` 快路 | 62.7 ms/步 · 15.9 tok/s | 76.6 ms/步 |
| + q/k/v 与 gate/up 权重合并 | 45.4 ms/步 · 22.0 tok/s | 59.6 ms/步 |
| + 去掉每层 `krows.eval()` | **23.2 ms/步 · 43.2 tok/s** | **33.5 ms/步 · 119.6 tok/s** |
| **官方 mlx_lm 0.29.1 同机同模型（基准）** | **23.89 ms/步 · 41.9 tok/s** | 未测 |

**最终比官方实现略快**（43.2 vs 41.9 tok/s），B=4 合计 119.6 tok/s。
HTTP 端到端（290 token 一条请求）：37.8 tok/s，其中 decode 23.5 ms + 调度 gap 3.3 ms。
并发 4 条各 64 token：0.86 s 完成 ≈ 297 tok/s 合计。

## 关键发现：真正的瓶颈是「每层强制同步」，不是 kernel 数

T1 时我在 `decode_batch` 里写了

```c
krows.eval();
vrows.eval();
for (int b = 0; b < B; b++) { ... slice_update ... }
```

当时想的是「先把这一行的 K/V 物化，再写进池」。但 MLX 是惰性求值的：
`slice_update` 和 `krows` 之间本来就有数据依赖，MLX 自己会排序，
**这两行 eval 唯一的效果是把图切成 24×2 = 48 段，每段都要等 GPU 跑完**
（`eval` 是同步点）。整步耗时因此从 23 ms 涨到 45 ms。

删掉这两行后，整个 decode 步（24 层 × 约 30 个 op）在一次 `eval` 里提交，
MLX 得以跨层融合 kernel、异步流水 —— 相当于拿到了 `mx.compile` 的大部分收益，
却不用重构函数签名为纯函数、也不用把 KV 池当输入输出传。

**教训**：在 MLX 里，除了最后取结果那一次，中间不要 eval。

## 试过但没用（都是实测，不是猜）

| 尝试 | 结果 | 说明 |
|---|---|---|
| Smax 对齐到固定档位（`SGL_MLX_BUCKET`=1/8/16/32/64/128） | 69.0 → 69.3 / 69.8 / 70.1 / 69.3 / 74.2 | 原假设「kernel 按 shape 特化、每步重编」不成立 |
| `mx::fast::scaled_dot_product_attention` 代替手写 softmax | 55.1 → 57.7（**反而慢**），且 fp16 下与手写路径舍入差 2.2e-3 | 本后端 shape（Smax<200、D=64、H=14）下 sdpa 没优势。默认关闭，留 `SGL_MLX_SDPA=1` 开关 |
| GQA 用 `broadcast_to` view 代替 `repeat` | 无明显变化 | repeat 的拷贝不是瓶颈 |
| 不写 KV（跳过 slice_update） | 74.1 vs 74.7（噪声内） | KV 写不是瓶颈（真正贵的是同步，不是写） |
| 换线程跑（主线程 vs 独立线程） | 43.19 vs 42.20 ms/步 | 线程不是原因 |
| **`mx::full` 型 AB 矩阵** | 数值自相矛盾（52.9 / 65.6 / 68.8 无规律） | 测量噪声 ±10 ms，单次对比不可信；改用多次取最小后才有意义 |

## 仍然有效的两项优化

**1. `mx::fast::rope` 快路径（省 6.3 ms/步）**

手写 rope 每层 q/k 各约 10 个 op（slice×2、multiply×4、add、subtract、astype×2、
concatenate），24 层 ~480 个 op；`fast::rope` 是 1 个 Metal kernel。

判定条件（加载时一次算好 `rope_base[l]`）：
- `rot == D`（全维旋转，不是 partial rotary）
- `pos_scale == 1 && attn_factor == 1`（没折过）
- 逐项校验 `invf[i] == base^(-2i/D)`（守住 gemma-2 那种每层不同 invf 的单纯形）

用它的两条硬约束（都是实测报错逼出来的）：
```
[rope] offset must be a scalar or vector with 1 elements but has shape (8).
[rope] offset must be an integer but got type float32.
```
→ offset 必须是 **int**、且只能是标量或长度 1 的向量。**所以只有「整次调用共用一个位置」
才能走快路**：prefill 的 n>1 和批量 decode 里位置各行不同，都回退手写。
prefill 是一次性大计算，op 数不是瓶颈，无所谓。

**2. q/k/v 与 gate/up 权重合并（省 12 ms/步）**

加载时把三份权重 `concatenate` 成一份（`wt_on()` 时沿 axis=1，否则沿 0），
前向一次 matmul 再 `slice` 切段。每层 matmul 从 7 次降到 4 次（168 → 96 次/步），
少 3 次 kernel launch + 3 次权重读取。

## 诊断工具（留在代码里）

| 开关 | 作用 |
|---|---|
| `SGL_MLX_PROF=1` | 每次 `decode_batch` 打 `[mlxprof] B= Smax= base0= lens0= ms` |
| `SGL_MLX_AB=1/2/4` | 不算 lm_head / 不写 KV / GQA 用 broadcast |
| `SGL_MLX_BUCKET=N` | Smax 对齐档位（默认 32；1 = 关） |
| `SGL_MLX_SDPA=1` | 走 `fast::scaled_dot_product_attention`（默认 0） |
| `SGL_MLX_FP16=0` | 权重/KV 走 fp32 |
| `SGL_MLX_CTX=N` | KV 池行数 |

## 测量方法上的坑

`tests/mlx_test.c` 原来用 `clock()/CLOCKS_PER_SEC` 计时间 —— 那是 **CPU 时间**，
MLX 等 GPU 的时间完全不计入，于是报出 42 ms 而实际墙钟是 66 ms（差 1.5 倍）。
已改成 `clock_gettime(CLOCK_MONOTONIC)`。**性能数字必须用墙钟。**

另外随机权重的 fixture（`models/*-rand`，L=2 E=64）输出分布近乎均匀，
top1-top2 概率差只有 1e-6~1e-8，此时 top-1 一致是噪声 —— 测试里加了
`top_gap()` / `top1_meaningful()`，只在分布够尖时才断言 top-1。
真模型（Qwen2.5-0.5B）的 gap 到 0.05~0.7，是有效的判据。

## 还没做（下一步的依据）

1. **`mx::compile`**：现在靠「一次 eval」拿到了大部分收益，但整步仍有 ~30 op/层
   的图构造在 host 侧。要再降，得把 decode 重构为纯函数（KV 池当输入输出传），
   用 `mx::compile` 把整步编译成一条 command buffer。风险中等、收益未知（估计 10~20%）。
2. **批量 KV 写**：B 个散点行 → B 次 `slice_update`。B=4 时 33.5 ms/步 vs B=1 的 23.2，
   增量 10 ms 基本就是多出来的 3×24×2 次写入。`metal_kernel` 可以一次写完。
3. **lm_head 的 GEMV**：[1,896]@[896,151936] 每步读 272 MB 权重。B=1 时是纯带宽
   受限（理论 4 ms @68GB/s），实测约 9 ms（AB=1 对比）。可以试 `mx::quantized_matmul`
   或把 lm_head 存 8-bit。
4. **调度 gap 3.3 ms**：`accel_http.c` 的调度轮询。B=1 时占 12% 的端到端时间。