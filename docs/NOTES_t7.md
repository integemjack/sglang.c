# NOTES_t7 — Speculative Decoding v2 收尾验收

日期：2026-10-27 ｜ models/tinyllama（target 1.1B）+ models/llama-160m（draft，同 Llama-2 词表）

## 验收结果（SGL_DRAFT_DIR=models/llama-160m test_spec）

- **接受率 13%**（drafted=60 accepted=8）——与 M18/M19 结论一致（未蒸馏同源小模型的上限）
- **数学保证保持**：spec greedy == target-only greedy 逐 token ✓（test_spec 断言）
- **端到端（CPU）**：spec 1.0 tok/s vs target-only 3.6 tok/s——**净减速**

## 结论

- v2 的 n-outer 批量 verify（sgl_llama_forward_n，权重驻留、K 候选一次过）已落地：
  verify 成本 ≈ 单步×1.2 而非 ×K（spec.c 调用，test_spec 通过）。
- CPU 的负收益瓶颈在 **draft 的 K 次串行前向**（160m 每步 ~0.1s），不在 verify。
  spec 的正收益场景是 **GPU**（draft 前向 <1ms 时接受率 13% 即可回本）——
  留作 CUDA spec 的后续工作（TODO_PROMPTS 任务 7 的 GPU 部分未在本轮展开）。
- 修复记录：v2 初版 kv_catch_up 把 1 行 logits 缓冲传给 forward_n 的 n>1 prefill
  导致堆溢出（已修：借 n 行临时缓冲末行拷回）；test_llama 增加
  decode_batch==forward 断言防回归。

## G7 补记：GPU 侧实现（2026-10-28，src/cuda_spec.c + sgl_cuda_forward_seg_rows）

- 后端新增多行 logits 出口 `sgl_cuda_forward_seg_rows`（lm_head 一张 M=n GEMM）；
  ensure_lgb 增长路径补 invalidate_graphs（图里绑的是 h_lgb/d_lgb 裸指针，铁律 4）。
- 正确性（cuda_test [spec]，全绿）：
  - qwen05 draft==target（同实例双 KV 段 0/8192）：接受率 100% (20/20)，24 tok 与非 spec greedy 全同；
  - tinyllama(fp16)+llama-160m：接受率 13.3% (8/60)，24 tok 与非 spec greedy 全同（与 CPU 版 13% 一致）。
- 性能（RTX 2060，free VRAM 2881MB，prompt='The meaning of life is'，new=32，best-of-3，
  非 spec 基线 = prefill + decode_batch B=1 图步）：
  - plain 80.3-82.9 tok/s；spec K=1 35.1 (0.44x, acc 39.1%) / K=2 28.6 (0.35x, acc 28.6%) /
    K=4 22.0 (0.27x, acc 14.8%)——**仍净减速**。
  - 成本实测分解（K=4）：每轮 ≈ 135ms（修补齐前）：4 次 draft 图步 ~20ms + verify n=4
    plain 路径 ~30ms + reject/bonus 补齐各 1 次 plain forward_seg ~45ms。补齐改走
    decode_batch 图步后降到 ~97ms/轮（11.9→22.0 tok/s）。
  - 根因：① 接受率上限（未蒸馏同源小模型 13-15%，每轮只产 1.6 tok）；② verify/catchup
    的 plain 路径在 WDDM 桌面 GPU 下不能图化（n 可变），固定吃 2-3 倍图步成本。
    正收益条件：接受率≥ 50%（同源蒸馏 draft）或 verify 图化（固定 n）+ 批量 spec。
- temperature>0 的修正采样（speculative sampling）GPU/CPU 均未实现，只支持 greedy。

## G14 补记：方向 1「固定 n verify + 流水线」落地（2026-10-27）

按 TODO_PROMPTS v0.4 的 G14 做的；结论：**结构成本砍掉了 1.5 倍，但仍净减速**（诚实记录）。

### 先量了成本分解（SGL_SPEC_PROF=1，K=4，旧结构）

| 阶段 | ms/轮 | 占比 |
|---|---|---|
| draft K=4 图步 | 13.4 | 24% |
| verify（n=4，plain forward_seg_rows） | 25.8 | 46% |
| target 补齐 1 次（decode_batch 图步） | 12.8 | 23% |
| draft 补齐 1 次 | 3.9 | 7% |

两个可攻点：verify 的 cublas 小 M 路径、以及每轮固定一次 **target 补齐**。

### 改动

1. **`k_gemm_nt_mrow` / `k_gemm_nt_h_mrow`（small-M GEMV）**：每输出行一个 block，
   权重行只读一遍、同时累 2/4/8 个 query 行。关键点：**MT 必须是编译期常量**
   （模板），动态 M 会让累加器落到 local memory —— 实测 M=2 反而比 M=1 慢
   （22.1ms vs 14.6ms），模板化后 16.1ms。跨线程归约改成 warp shuffle（原 MT 轮
   同步归约让 n=5..8 比 n=4 还慢）；并且 i≥M 的行不能读（A 只分配 M 行，越界读
   让 n=5 白多花 6ms）。
2. **流水线化（去掉 target 补齐）**：预填 `np-1` 行，把 prompt 末 token 当“挂起
   token”，下一轮 verify 的第 0 行负责吞它。于是每轮 verify 固定 n=K+1 行，
   reject/bonus token 的 KV 由下一轮 verify 顺带写掉 —— 每轮省一次 target 图步。

### 结果（tinyllama fp16 + llama-160m，ctx=1024，prompt 16 tok，new=32，best-of-3）

| K | 接受率 | 改前 | 改后 | 每轮成本（draft + verify） |
|---|---|---|---|---|
| 1 | 39.1% | 0.44x（35.1 tok/s） | **0.73x（60.5 tok/s）** | 2×3.35 + 14.7 = 21.4 ms |
| 2 | 28.6% | 0.35x（28.6 tok/s） | **0.64x（53.1 tok/s）** | 3×3.35 + 17.2 = 27.2 ms |
| 4 | 14.8% | 0.27x（22.0 tok/s） | **0.40x（33.3 tok/s）** | 5×3.35 + 27.7 = 44.5 ms |

plain 基线 82.4–82.8 tok/s（decode B=1 图步 11.4 ms/token）。

### 为什么还是上不了 1.0x（可算的结论）

每轮成本 = (K+1)·d + v，产出 1+K·a 个 token（d=draft 图步 3.35ms，v=verify 成本，
含固定启动开销）。要赢只需 `(1+K·a)·t > (K+1)·d + v`，代入实测的 t=11.4、d=3.35：

| K | 需要的每候选接受率 a | 实测 a |
|---|---|---|
| 1 | **88%** | 39.1% |
| 2 | **69.5%** | 28.6% |
| 4 | **72%** | 14.8% |

所以：**在 WDDM 桌面 GPU 上，draft 图步本身就有 ~3.3ms 的固定开销（160M 模型的
数学只要 ~1.9ms），而 target 一步 11.4ms，d/t 只有 0.29 —— 必须靠接受率到 70–90% 才回本。
未蒸馏的 llama-160m 对 tinyllama 只有 13–39%，所以依然净减速。**

要真正转正，只剩两条路（都超出本轮范围）：
1. 同源蒸馏 draft（K=2 时只需 ~70% 接受率，同族蒸馏通常能到 70–85%）—— 需要新权重；
2. 把 draft 的 K 步合成一次图启动 + 设备端 argmax（去掉每步的 host 同步），
   把 d 压到 ~2ms，同时给 verify 做定长图（v→11.4）：K=1 门槛会降到 ~60%。

方向 2（cuda_http 批量 spec）和方向 3（temperature>0 修正采样）本轮未做：前者需要动

调度器（batch 里混 spec/非 spec 请求），后者的价值是功能而不是速度。

---

## 补记 II：方向 3「temperature>0 修正采样」落地（2026-11-04）

`sampling.c` 抽出单一来源的分布构建（`sgl_dist_transform`：penalty → T → min_p
→ top_k → top_p；`sgl_sample_ex` 抽的就是它，行为逐位不变），`spec.c` /
`cuda_spec.c` 在原 greedy 轮次结构上加标准拒绝采样：draft 从 q̃ 抽 K 候选 →
target 逐位 `u < min(1, p̃(x)/q̃(x))` 检验 → 拒绝处从 norm((p̃−q̃)₊) 重采样 →
全接受再从末位 p̃ 直抽 bonus。数学保证：产出 token 的边缘分布 == 直接从
target 同参数抽样，与 draft 好坏无关（draft 只决定接受率/速度）。

验收（全部是测试内断言，不是目测）：

| 检验 | CPU（tinyllama 1.1B，test_spec） | GPU（tinyllama+llama-160m fp16，cuda_test） |
| --- | --- | --- |
| T=0 退化 == greedy spec（逐 token） | ✓ | ✓ |
| draft==target 接受率（q̃==p̃ ⇒ 应 100%） | **100.0%（20/20）** | **100.0%（20/20）** |
| 一号 token 分布对拍 TV(spec,plain) ≤ 1.6×TV(plainA,plainB)+0.02 | N=96：**0.385 ≤ 0.737** ✓ | N=400：**0.292 ≤ 0.508** ✓ |

关键点：两处的 draft 都很弱（GPU 组合接受率 13.3%；CPU C 段接受率 0%——几乎
每个 token 都从 (p̃−q̃)₊ 修正重采样产生），分布仍然精确重合——这正是修正
数学的不变量在起作用，不是「草稿准」造成的巧合。服务层（cuda_http 的 MTP、
sglc CLI）尚未接采样版，仍 greedy；库层 API 已可用
（sgl_spec_sample_generate / sgl_cuda_spec_sample_generate，penalty 的 history
与非投机路径一致；stop_ids 仍由服务层检查）。

另记：全量 cuda_test 在桌面占 ~1GB 显存时，会在 gemm-bench 之后的 pure-decode
段 AV 崩溃（stash 掉本次改动后在 HEAD 上同样复现，非本次改动引起；显存空闲
时全量可过）。为此给 cuda_test 加了 `SGL_CUDA_ONLY_SPEC=1` 开关，显存紧张时
仍可单独验证 spec 段。
