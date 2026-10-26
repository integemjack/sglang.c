# NOTES_t6 — 多 LoRA 运行时热切换（方案 B：多实例池）

## 设计

LoRA 原为加载时合并（`SGL_LORA=<dir>` 环境变量，`llama.c` 在量化前 `W += B·A·(alpha/r)`，
一份权重只能带一个 adapter）。本改动改为**运行时多适配器**，CPU 栈采用方案 B（多实例池）：

- 调度器（`sgl_scheduler`）持有实例数组 `insts[]`：`insts[0]` 是 base（干净权重），
  每个 adapter 通过 `sgl_sched_add_lora(name, model_dir, lora_dir)` 各建一份
  「base+合并 delta」的完整 `sgl_model` 副本（内部临时设 `SGL_LORA=lora_dir` 走现有
  `sgl_model_load` 加载+合并路径，完成后恢复原环境）。**复用全部现有合并代码，CUDA 侧不动**
  （方案 A「运行时算 delta、多 adapter 共享一份 base 权重」留给后续 CUDA 实现）。
- 每实例独立分页 KV 池 + 独立 RadixCache（页号属于各自实例的池，不能混用）。
- 请求带实例下标（`sgl_req.inst`，默认 0）：
  - prefill 本来就是单请求一步，直接用该请求的实例；
  - 批量 decode **逐实例分组**（不同实例权重不同，不能混进同一 batch），
    每实例各组成一批各做一次 `decode_batch`，实现简单且正确。
- 提交时按名字解析实例：`sgl_sched_submit_lora / submit_stream_lora(..., lora)`，
  `lora=NULL` 走 base；未知名返回 -1。旧 API `sgl_sched_submit/submit_stream` 语义不变（=base），
  **不带 --lora 时不建任何额外实例，行为与改动前完全一致**（15 套旧测试全绿回归验证）。

## 用法

### CLI（服务与单发均可）

```
sglc --model models/tiny-llama --serve --port 30000 --lora s1=models/test-lora-s1,s2=models/test-lora-s2
```

`--lora name=dir[,name2=dir2,...]`：每个 adapter 预先各建一份合并副本。
不带 `--lora` 且无 `SGL_LORA` 时行为与以前完全一致。

### HTTP

`/generate`、`/v1/chat/completions`（及 `/v1/completions`）JSON body 顶层可选字段
`"lora":"name"`，按名路由到对应实例；不带该字段走 base 实例；未知名返回 400：

```
{"error":{"message":"unknown lora 'nope' (load adapters at startup with --lora name=dir)"}}
```

## 内存公式（方案 B 的代价）

每 adapter 一份**整模型权重副本**（不是只存 delta）：

```
N 个 adapter 总内存 ≈ (N+1) × 单份合并权重 + N × 独立 KV 池
                     （KV 池每实例一份，页数与 base 相同）
```

实测参考：tiny-llama（2 层随机，f32）单份 ≈ 4.4 MB；qwen05 f32 ≈ 3.8 GB、
w8 ≈ 1 GB。例：qwen05 + 4 个 adapter（w8）≈ 5 × 1 GB = 5 GB。
对比方案 A（共享 base、运行时算 delta）只需 delta 显存（adapter 本身仅 MB 级），
那是 CUDA 侧后续工作；CPU 栈内存宽裕时方案 B 换实现简单与正确性。

## 测试

- `tests/gen_lora.py` 加可选 `--seed N`（默认 42，旧 `models/test-lora` adapter 逐字节可复现）；
  用 seed 1/2 生成 `models/test-lora-s1`、`models/test-lora-s2`（基于 models/tiny-llama）。
- `tests/ref_lora.py` 传第 3 参输出路径即可生成各自参考（已生成
  `tests/ref_lora_s1.json`、`tests/ref_lora_s2.json`）。
- 新增 `tests/test_lora_multi.c`（进 ctest，CMake 环境变量 SGL_TEST_LORA_S1/S2、
  SGL_TEST_LORA_REF_S1/S2）：同调度器内 s1/s2 请求输出各自 == 单独加载该 adapter 的
  numpy greedy6 参考；并发混跑（2×2 交错提交）与串行参考一致；未知名提交返回 -1；
  重复名 add 被拒；base 实例仍可用。
- ctest 16/16 全绿（原 15 套 + 新增 test_lora_multi，无回归）。

## curl 验证记录（sglc --serve --lora s1=…,s2=…，端口 18099）

```
== /generate lora=s1 ==  completion "ј спо ду Arist Bedeutung turn"   == ref_s1 greedy6 解码文本，一致
== /generate lora=s2 ==  completion "ј споçosMillнал suitable"        == ref_s2 greedy6 解码文本，一致（s2≠base，路由生效）
== /generate 无 lora ==  走 base 实例，正常返回
== lora=nope ==          HTTP 400 {"error":{"message":"unknown lora 'nope' (load adapters at startup with --lora name=dir)"}}
== /v1/chat/completions + "lora":"s1" == 正常返回（ChatML 前缀下路由同生效）
```

注：tiny-llama 是随机小模型，s1 adapter（scale=2）对该 prompt 的 greedy6 恰与 base
全等（参考 `ref_llama.py` 输出验证），因此 s1 与 base 文本相同属巧合；路由正确性由
s2≠base 与 C 测试的逐 token 断言共同证明。
