# MLX 后端：4-bit / 8-bit 量化检查点

> 目标：直接加载 `mlx-community` 的量化检查点（`<base>.weight` 是 U32 打包，
> 配 `.scales`/`.biases`），不做成 fp16。Qwen2.5-0.5B-4bit 实测：
> 磁盘 276 MB（fp16 是 951 MB），decode **75 tok/s**，同机官方 `mlx_lm` 80.7。

## 1. 检查点长什么样

`model.safetensors` 里每个稠密矩阵是三个张量：

| 名字 | dtype | shape | 含义 |
|---|---|---|---|
| `<base>.weight` | U32 | `[N, K/8]` | 每 32 位打包 8 个 4-bit 值 |
| `<base>.scales` | F16 | `[N, K/64]` | 每个 group 一个缩放 |
| `<base>.biases` | F16 | `[N, K/64]` | 每个 group 一个零点偏移（不一定为 0） |

`config.json` 里有 `"quantization": {"group_size": 64, "bits": 4}`。
MLX 的语义是 `w = q * scale + bias`，喂给
`mx::quantized_matmul(x, w, scales, biases, transpose=true, group=64, bits=4)`
即可，**不要自己解包**。

两个容易搞错的点（都实测过）：

1. **每个 U32 打包 8 个值，不是 2 个**。`[N, K/8]` 里第 `K/8` 维是「K 方向
   每 8 个 4-bit 一包」，展开后才是完整的 `[N, K]`。手写拆包时写成 2 个会让
   形状对不上（`cannot reshape 200704 into (896,896)`）。
2. **scales 可以是负数**。这批检查点里 q_proj 的 scales
   `min=-0.1357 / max=+0.1426`，这是正常的（`mlx_lm` 读出来一模一样），
   别以为读错了文件。

## 2. 为什么不能走 `sgl_st_get`

`src/safetensors.c` 的 `sgl_st_get` 只认 F32/F16/BF16/FP8，遇到 U32 直接
`ERROR ... dtype=U32, only F32/F16/BF16/F8_E4M3/F8_E5M2 supported now`。
而 `sgl_llama_load()` 因此拿不到 `embed_tokens`，整个 CPU 侧加载返回 NULL。

**服务层的降级路径**（`apps/accel_http.c`）：`sgl_llama_load` 失败时改用
`sgl_llama_load_full(dir, /*config_only=*/1)` 只拿 config 与 `m->st`，
然后交给后端自读权重。此时 `m->layers` 全是 NULL，所以后端**不能**用
`m->layers` 里的 `sgl_tensor`——MLX 后端的做法是：权重一律按名字从
st 或 MLX 自己的 safetensors 表里取（`load2d` / `st1` 都先查 `g_mxw`）。

`cfg_only && strcmp(A->name, "mlx") != 0` 时直接给出可读的错误，而不是让
CUDA 后端去读 NULL 的 `m->layers`。

## 3. U32 字节 → `mx::array`：走过的四个坑

这是整个 T2 最耗时的部分。**结论：交给 `mx::load_safetensors`，不要自己碰字节。**

| 写法 | 结果 |
|---|---|
| `mx::array(ptr, shape, dtype)` | 独立 probe 里正常；链进 `sglc`/`mlx_q4_test` 后在 `array::init` 的 `std::copy` 里 **SIGBUS**（uint32 与 float32 两个分支都崩） |
| `mx::zeros(sh,dt).eval()` + `memcpy(a.data<T>(),...)` | 不崩，但 **数据被丢**：zeros 带 primitive，MLX 把 `zeros → take` 常量折叠成 0，实测 logits 全 0（top10 全 `0.0000`） |
| `allocator::malloc` + memcpy + `array(Buffer,...)` | **SIGSEGV** |
| `mx::load_safetensors(file)` | ✅ 用这个 |

另外两个附带的坑：

- **`mx::array` 没有默认构造函数**，`mx::array a; a = ...;` 编不过，
  必须用三目/lambda 一次初始化（`decode_batch` 的 K/V 分支就栽在这）。
- **`mx::array(Shape, Dtype)` 这个重载不存在**：写成 `mx::array(sh, dt)` 会选到
  `array(It data, Shape, Dtype)`，把 Shape 当迭代器，报
  `no viable overloaded '='`（`std::copy` 到 `complex64_t`）。MLX 也没有
  `empty()`，只有 `zeros`/`full`。

## 4. 性能：两处关键优化

### 4.1 lm_head 保留量化形式（47.5 → 76.9 tok/s）

`tie_word_embeddings: true` 时 lm_head 就是 embed_tokens。第一版把 wte 解量化成
fp16 `[V,E]`（151936×896 = 272 MB）再转置给 lm_head 用，**每生成一个 token 都要
把 272 MB 扫一遍**。

改法：`g->lm_head_q` 额外存一份量化三件套（≈87 MB），`head()` 优先用它走
`quantized_matmul`；`g->wte` 仍保留解量化版（`take` gather 需要稠密）。
wte 解量化那份只有 272 MB 的一半浪费在「显存里放了两份等价权重」上，
但对 16 GB 统一内存不构成压力，换来 1.6x 提速。

### 4.2 并发时把活跃 KV 行紧凑化（25.9 → 111 tok/s）

`accel_http` 把 KV 池按 `seg = ctx / conc` 切段，请求按段分配。`decode_batch`
原本直接 `slice(kc[l], 0, Smax)`，而 `Smax = max(base_pos[b]+lens[b]+1)`——
**段的跨度**决定它，不是序列的实际长度。

实测（4-bit，B=4，lens≈100）：

| seg | Smax | 每步 | 合计吞吐 |
|---|---|---|---|
| 256 | 896 | 25.5 ms | 111.9 tok/s |
| 4096 | 12416 | 137 ms | **25.9 tok/s** |

差 5.4 倍，全花在白扫的行上（12384 行里只有 ~100 行有内容）。

改法：`span > 2*maxlen + 64` 时用 `mx::take(kc[l], kidx, 0)` 把每个序列的活跃行
gather 成 `[B, maxlen, KVH, D]`，mask 的坐标系同步换成「紧凑索引」。
搬运量 `B*maxlen*KVH*D*2` ≈ 100 KB/层，而扫 span 要 ≈ 1.6 MB/层。

改完 seg=4096 也从 25.9 → **110.8 tok/s**，两种段大小无差别。
`SGL_MLX_COMPACT=0` 可关掉对照。

## 5. 8-bit（同一条代码路径，四个新坑全被对拍抓出）

8-bit 与 4-bit 走同一条链路，只是 `bits` 从形状现场推断。四个坑：

| 坑 | 症状 | 修法 |
|---|---|---|
| bits 推断公式漏除 group | 加载直接报「只支持 4/8-bit（推出 bits=512）」 | `bits = 32*kp / (scc*group)`，group 取 64 |
| `cat2` 拼接权重时丢量化参数 | forward 时 `quantized_matmul` 报形状不兼容（1152×224 vs bits=4） | `QW` 携带 `group/bits/mxfp`，拼接时检查并传递 |
| `emb_lookup` 只 take 了 w/scales，biases 全量传 | **8-bit Abort**（dequantize 形状不匹配）；4-bit 是 MLX 恰好没查 —— 别赌 | biases 也按行 take |
| lm_head 检查只看 `lm_head` 非空 | tie+量化时加载报「没有 lm_head 且 tie=0」 | 同时检查 `lm_head_q.quant()` |

### 5.1 对拍：与官方逐位一致，以及 mlx_lm 自己的 1D 路径差异

排查过程中逐层对过 sum：**embed、每一层 hidden、最终 hf，C++ 与官方原装层
全部逐位一致**（hf sum=248.23296，hf[0,:4]=7.32812/0.20398/0.14160/13.74219）。
最后确认：

- **官方 2D 输入**（`model(mx.array([[9707]]))`）与我们**逐位一致**：
  `lg[271]=12.95312 lg[198]=11.46094 lg[7426]=9.12500`。
- 官方 **1D 输入**（`mx.array([9707])`）会走出**另一条数值路径**
  （top2 变成 7426:9.1250，198 掉到 10 名开外）——batch=1 语义上应一致，
  这是 mlx_lm 侧的差异。**对拍统一用 2D 输入。**

另外两个实测事实：拼接后一次 `quantized_matmul` 再 `split`，与逐个 qmm
**maxdiff=0**；`quantized_matmul` 对 2D/3D 输入也 maxdiff=0。

### 5.2 8-bit 性能

| 路径 | 4-bit | 8-bit |
|---|---|---|
| 同尺子 bench（`mlx_q4_test`，greedy 64 步墙钟） | 13.30 ms/步 = **75.2 tok/s** | 17.16 ms/步 = **58.3 tok/s** |
| 官方 `mlx_lm.generate`（带 KV cache） | 80.7 tok/s（12.40 ms/步） | 65.3 tok/s（15.3 ms/步） |

与官方一致地存在 ~1 ms/步 的固定 gap（我们的 logits 全词表 qmm + 拷贝），
量化位宽不改变这个差值。

诊断开关：`SGL_MLX_TRACE=1` 逐层打 emb/每层/hf 的 sum（与 Python 对照定位
数值分叉；强制同步，只在排查时开）。

## 6. MXFP4（无现成检查点，合成数据验证）

mlx-community 没有 0.5B 级的 mxfp4 检查点，用 Python 手写 safetensors 容器
（mlx 0.29 的 `save_safetensors` 不收 U32/U8，报 `bad_cast`）合成一个
2 层 Qwen2 形状的检查点（weight U32 + scales U8 + 无 biases）验证：

- `load_q3` 的 U8/无 biases 分支正确推出 group=32/bits=4/mxfp=1；
- 逐层 hidden 与 Python 参照**逐位一致**（L0 sum=-35101、L1 sum=155518）；
- logits 统计量全同（mean=-2.83055 std=23.70873 min=-104.75 max=+66.062）、top1 一致；
  top10 中后段在合成数据上排序不齐（随机权重幅度大、分布平坦，属预期；
  真实检查点验证待有货源）。

**顺带抓出 MLX 0.29.3 的一个 kernel bug**：mxfp4 的 `quantized_matmul` 对
**f16/bf16 输入给出垃圾值**（同一输入，f16 → absmax 2e23，f32 → 1659，差 20
个量级；与 dequantize+matmul 对比，只有 f32 输入数学正确）。另外
`dequantize(mxfp4)` 输出是 **bfloat16**（affine 是 f16）。

修法：`proj()` 里 mxfp 权重强制 f32 输入、输出转回原 dtype；`emb_lookup`
的 mxfp4 dequantize 结果也 astype 回 wdt。affine 4/8-bit 不受影响
（f16 输入与 mlx_lm 逐位一致已验）。

合成数据还有个坑：scales 不能用随机 byte —— e8m0 指数 255 解成 2^128，
激活直接溢出 fp16 → NaN；要用合法指数（如 127 = 2^0）。

## 7. 验收（4-bit）

| 项 | 结果 |
|---|---|
| logits vs `mlx_lm`（token 9707） | top10 顺序与数值**逐位一致**（14.2578 / 11.2578 / 10.9219 …），logits mean/std/min/max 对上（-2.14518 vs -2.14576） |
| 生成质量 | 中文、英文、代码各 prompt 全部正常 |
| 单条 decode（128 tok） | 75.0 tok/s（1.71 s）；官方 mlx_lm 80.7（1.59 s），差 7% |
| 并发 4 条 | 111.1 tok/s 合计 |
| 加载 | 0.4 s（config-only + 惰性 safetensors） |
| fp16 回归 | 6 个模型 12~13 项对拍全绿（紧凑化没碰坏单/多序列路径） |

## 8. 没做的

- **多分片**（`.index.json`）：`mxw_open` 的多分片支持已写（逐 shard
  `load_safetensors` 合并），正在用 Qwen2.5-7B-4bit（4 个 shard）验收中。
- **mxfp4 真实检查点**：合成数据已验证链路，待 mlx-community 出 0.5B 级
  mxfp4 后补端到端对拍。
- **embed_tokens 的 gather_qmm**：现在 wte 是解量化的 272 MB，7B 级模型上
  embed 会到 1 GB。MLX 有 `gather_qmm` 可以避免，属于 T3。