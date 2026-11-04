# G20：GPU 页级 radix（slot 零拷贝挂页 + 页池动态准入）

> 2026-11-04 完成并验收。任务卡：TODO_PROMPTS.md G20（侦察在 004db81）。
> 相关：G6（f32 池段级 copy_kv 前缀复用）、G15（租让表）、M2（kvq 量化分页 KV）。

## 设计（按实现落地的形态）

- **后端原语**（`include/sgl/cuda.h` + `cuda_backend.cu`，仅 kvq 恒等页表模式：
  `SGL_KVQ=1` 且非 M6 换页、非 host 副本；其余返回 -2，服务层回落静态段）：
  - `sgl_cuda_kvq_page_size()`：页大小（不支持 = 0）；
  - `sgl_cuda_pages_map(dst,src,npg)`：dst 起那组页的页表项改指 src 同位置页的
    slot——**零 D2D**，只改 `h_pt` + 增量上传 `d_pt`（一次 H2D）；MTP 自有 f32
    段照 `copy_kv` 语义拷（服务投机路径全副作用保持）；
  - `sgl_cuda_pages_unmap(base,npg,cow)`：还原恒等页表。`cow=1` 先把共享页数据
    逐页 D2D 拷回本段（写时复制落地），`cow=0` 直接弃。
  - 「引用计数」在 M2 恒等模式（n_slots==n_pages、pt 恒等、无 slot 分配器）下
    **不需要设备侧 refcount**：共享生命周期完全由服务层的租让表/页状态机管理
    （free/active/leased）。M6 换页模式不支持，明确挡掉。
- **服务层**（`apps/accel_http.c`，ABI 3 新增可选 `page_size/pages_map/pages_unmap`）：
  - 请求的 KV 基址与并发槽解耦：`r->base/pg0/npg`（页区间分配器 first-fit，
    两级准入：prompt+max_new 全量拿不到就退 prompt+1，decode 顶到容量提前
    length 截断）；非 pagemap 维持 `seg*seg_len` 静态段，行为不变；
  - 租让表（G15）升级页粒度：登记=前 ceil(len/P) 页转 leased；驱逐/COW 由
    `pfx_drop`/`pfx_cow_mappings` 统一处理（驱逐时先 COW 还在映射的活跃请求，
    再还页）；
  - 命中=页对齐截断（尾半页丢弃，≤P-1 行重算）+ `pages_map` 零拷贝挂页；
  - `/health` 暴露 `pagemap`、`pages{size,total,free,leased,mapped,cow,allocs,fails}`
    与 prefix 命中率。

## 验收（全部脚本/断言内，日志可复跑）

后端级（`cuda_test` `[kvq]` 段，`SGL_KVQ=1 SGL_KVQ_TEST=1 SGL_CUDA_ONLY_SPEC=1`，
qwen05 fp16，ctx 8192，页 64）：

| 检验 | 结果 |
| --- | --- |
| map vs copy（G6 语义基准）logits | **maxerr = 0（逐位）**；操作耗时 0.10ms vs 102-118ms（**190-1296×**） |
| map vs 同段分块重算（512+64） | 逐位一致（greedy 同） |
| COW：unmap(cow=1) → 写坏源段 → 续 decode | token 不受污染，6 步续算 6/6 SAME |
| unmap(cow=0) 后重写本段 | 接口正常 |
| （记录）一次算全 576 vs 分块/复用 | **maxerr 3.50002**，greedy 654 vs 406 —— pre-existing，见 G21 |

服务级（`tools/kvq_serve_check.py`，cuda_http，ctx 8192 conc 8 静态段=1024）：

| 检验 | kvq（pagemap） | f32（对照，copy_kv） |
| --- | --- | --- |
| (2) 同前缀两请求输出 | 64 vs 85 chars，差异=G21（分块 vs 一次算） | **逐字一致（86=86）** |
| (3) 命中与挂页 | hits=1，**mapped=8 页**，cow=8 | hits=1（copy 路径） |
| (4) 单请求 ~3600 tok >> 静态段 1024 | **200 OK 正常生成**（动态准入；静态模式会 400） | （跳过，会 400） |
| (5) 页回收 | churn 后 unexplained-busy=0（无泄漏） | — |

两模式脚本均 `all OK`。

## 过程中发现并修复的 pre-existing bug

**`sgl_cuda_lin_slots` 语义错位（W1/t11 层面，已修）**：字段是「槽数配置」
（dense 默认 8），服务层却拿 `pfx_ok = (lin_slots <= 0)` 当「模型有无线性注意力」。
后果：**CUDA 服务的前缀缓存（G6/G15 整套）自 t11 起 silently 关闭**（pfx_ok=0、
hits 恒 0、无任何报错）。修法：dense（lin_n==0）返回 0，hybrid 返回槽数不变。
f32 对照脚本验证 copy 复用随即恢复（hits=1、输出逐字一致）。

## 遗留（G21）→ 已定位并闭环（2026-11-04 同日，dump 实证）

**现象**：kvq 池下「分块 prefill / 复用」与「一次算全」不自洽（logits maxerr
~3.5、greedy 会变；f32 池同场景 1.4e-05）。

**根因（cuda_test `[kvq]` 段 sgl_cuda_kvq_dump 逐字节对比，576 行全量 dump）**：
- 页 0-7（512 行，两路径都在同一块内算）**逐位一致**；
- 差异全部集中在**页 8（尾块）**：54228/1880064 字节（2.9%），第一处在
  **L0 page8 tok1 off132 —— V 的量化 scale 区**（tok_stride=136 的 [128,136)）；
- 机制：M=576 与 M=64 的 GEMM（PROJ/cuBLAS）选不同 kernel → 行输出
  fp 级微差 → 尾块的块 amax/scale 跳档，被 4-bit 量化【固化成离散差异】→
  逐层放大 → greedy 分叉。f32 池无量化，同样微差只传播成 1.4e-5。
- copy_kv / G20 pages_map / 分块重算三者逐位一致（同一份量化字节），
  与 G20 无关；kvq 分支自 M2 起从未做过「vs 一次算全」的验收（默认关是对的）。

**修法取舍**：a) 锁定 GEMM 单 kernel（治本，但固定 algo 跨卡/跨版本不稳、
prefill 性能回退，高风险）；b) 量化前确定性预整（无数学保证的 hack）；
c) **接受 + 语义边界明确化**（上游 sglang/vLLM 对量化 KV + prefix cache 同样
不保证逐位）→ 选 c。

**落地的服务一致性语义（kvq_serve_check.py (2) 断言）**：「未命中 vs 命中」
不保证逐位（不同执行路径，已文档化）；**「同路径」必须逐字一致**——两个都
命中的请求实测 85=85 chars 逐字一致 ✓；f32 池下命中与否仍逐字一致（86=86）✓。
排障工具：sgl_cuda_kvq_dump/dump_size（正式导出，页对齐区间字节比对）。
