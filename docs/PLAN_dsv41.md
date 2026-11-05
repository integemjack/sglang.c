# DeepSeek-V4.1-Flash × Strata 卸载技术：设计（T4/dsv41）

> 目标：在 sglang-c 的 MLX 后端跑 DeepSeek-V4.1-Flash（552B MoE，16GB M1 iMac）。
> 参考：官方 `inference/model.py`（已在 `tools/dsv41_ref/`，**语义以它为准**）、
> Strata 的 [DEEPSEEK41.md]（专家卸载技术来源）、真实 config.json。
> 状态：设计定稿，实现开始。模型 2.00bpw 集（248GB）下载中。

## 1. 架构（text path，全部已从 model.py 核实）

残差流是 **hc_mult=4 份并行拷贝（Hyper-Connections）**，不是单流：

```
embed(129280→5120) → 复制4份 → Engram(L1) → Block(L0) → … → Engram(L14) → Block(L14)…
→ hc_pre 收敛 → RMSNorm → head[129280,5120]
```

### Block（model.py:909）
- `hc_mixes(x; fn,scale,base)`：flatten 的 hc 流 RMSNorm → linear[(2+hc)·hc, hc·d] →
  Sinkhorn 归一（20 迭代, eps 1e-6）出三组系数 pre/post/comb[b,s,hc,hc]
- attention 输入 = hc_pre(x, **上一块 FFN 的** pre_mix)；FFN 输入 = hc_pre(x, **本块 attention 的** pre_mix)
- hc_post：`post·y + Σ_c comb·residual`。参数全 fp32（hc_attn_fn/hc_ffn_fn/base/scale）

### Attention（MLA 变体，model.py:614）——每层两条 KV 源拼接
1. **窗口**：`kv_norm(wkv(x))[·,512]`，RoPE 末 64 维，fp8 量化，**环形缓冲 128**（window_size=128）；
   每个query 看全部窗口槽（ring 未满时 -1 屏蔽）
2. **压缩**（仅 compress_ratio>0 的层）：`index_topk=512` 个压缩位置。
   压缩由 **kv_source_layers={2,8,14,20}** 的 Compressor 做：softmax(score) 门控池化 ratio 个 token →
   1 个 latent（**RoPE 前**，索引器要用未旋转形式）→ norm → RoPE（**compress theta 160000, YaRN
   orig 65536, factor 16**）→ fp4 量化（组16，e4m3 scale）→ 压缩缓存 [S/ratio, 512]。
   ratio=1 的层是纯投影无池化。其余层只读最近前驱 source 的缓存。
- **q**：wq_a(5120→1280)→RMSNorm→wq_b(1280→64×512)，RoPE 末 64（窗口层 base theta 10000 无 YaRN；
  压缩层用压缩 YaRN 频率表——**每层两张不同的 freqs**）
- **topk 选择**：index_source_layers={2,8,14,20,24,28,32,36} 各自跑 Indexer，其余层复用其结果；
  index_k 缓存只有 kv_source 层拥有（wk(512→128)+norm+RoPE+fp4）
- **Indexer**（model.py:488）：32 头 ×128 维小注意力；q 来自 attention 的 qr；score=Σ_h relu(q·k)·weights_proj，
  top-512 可见压缩位（可见性=块尾已过）；candidate 预筛（layer 20，块 8，top 2048 块）只在超长上下文起作用
- **sparse_attn**：q 对 topk_idxs gather 的 kv 做点积，**每头加 attn_sink**（logit 加性参数）；
  输出做**逆 RoPE**（末 64 维）再 o 投影
- **o 投影分组低秩**：64 头分 8 组，块对角 wo_a（每组 64×1024）→ 拼接 → wo_b(8192→5120)

### MoE（model.py:856）
- Gate：`s = sqrt(softplus(x·W^T / gate_temp))`；**topk 用 s+bias 选，权重用原始 s**
  （noaux_tc）；weights 归一化 `w/=(Σw+1e-20)`（**不是 norm_eps**）再 ×route_scale 1.5
- 384 路由专家（fp4 e2m1，SwiGLU inter 2304）+ **1 个共享专家**（稠密）；swiglu_limit=10：
  up 双侧 clamp，gate 只 clamp 上限
- 40 层同构（get_moe_config 只在 MTP 层不同，backbone 全一样）

### Engram（model.py:328，层 1、14）
- n-gram 哈希表：每 token 查 n_hash_cols=(4-1)×8=24 行，两张表各 384M 行 × 256 维 fp8（+逐 32 块 scale）
- wkv(24×256 → 5120×5)：4 份 key + 1 份 value；门控 = sigmoid(signed_sqrt(归一化点积))，
  `h += gate·value`（对 4 份 hc 拷贝各自做）
- 表驻 SSD mmap，每 token 只碰 24 行（Strata 同款）

### 其余
- head/emb bf16（转换后 q8）；norm_eps=1e-20；yarn 只作用于压缩 KV 的 theta
- 量化：官方 dense fp8(32×32, ue8m0)+专家 fp4(e2m1×2 打包)；GGUF 集是 IQ2_XXS/IQ4_NL
- DSpark/MTP 3 层草稿、vision：**v1 不做**（输出质量无影响，属加速/多模态）

## 2. Strata 技术落地（我们引擎里的对应物）

| Strata 机制 | 我们的实现 |
|---|---|
| 冷专家驻 SSD 就地计算 | 专家权重按层分文件存 safetensors，`mx::load_safetensors` mmap 惰性加载；每 token 只 `take` 被选中的 240 个专家切片（MLX 惰性图只在 eval 时读那几块） |
| 热专家缓存 + 命中画像 | 每层热专家数组（预算可配，默认 0=v1 全走 mmap）；画像文件后续接（Strata profile-2.00.bin 格式待逆向） |
| Engram 驻 SSD | 同 mmap take，行级 gather |
| 压缩 KV（省内存大头） | 窗口 128 环形 + 压缩缓存 [S/ratio]，40 层的 KV 常驻 ≈ 128×512×40×2B ≈ 2.6MB/seq + 压缩部分 |
| Faster 模式（top-5） | `SGL_DS41_EXPERTS=5` 环境开关（引擎侧只是 topk 少一个） |
| DSpark 草稿 | v2（投机解码，接口已留） |

### 16GB 内存预算（q4 稠密 + fp4 专家，估算）
- 常驻：稠密 ~3.6GB（q4）+ hc 流/激活 ~1.5GB + 热专家（可配）→ **放得下**
- 驻盘：专家 384×40×12.6MB(fp4) ≈ 164GB；Engram ~196GB（IQ4 是 111GB）
- 每 token 专家 I/O：240×12.6MB ≈ 3GB → 外置 NVMe ~2GB/s → **~0.7 tok/s 起步**（慢但真能跑；
  热缓存加上去后显著回升）

## 3. 实现拆步（每步可独立验收）

1. **dsv41-mini 合成模型 + Python 参考**：小尺寸（dim 64, 2 层, 8 专家 top2, hc4, 含 Engram/压缩/索引器全机制）
   → `tools/dsv41_ref/make_mini.py`（torch 参考跑通、存 npz 输入输出）
2. **src/mlx_ds41.cpp**：按本文档实现前向（先 dense-expert 版验证数学，再接 mmap 卸载），
   对拍 mini 的 logits（目标：fp32 路径逐位级一致，fp16 ~1e-3）
3. **格式转换器** `tools/dsv41_ref/convert_gguf.py`：GGUF IQ2_XXS/IQ4_NL → MLX safetensors
   （稠密 q4_0→affine4、专家→mxfp4、engram 行→fp8+scales）。llama.cpp 的 dequant 逻辑移植
4. **接入 accel 表**：`sgl_accel_mlx` 加 dsv41 分派（config model_type=deepseek_v41）
5. **真模型验收**：下载完成后转换（~1-2h）、加载、对拍 mlx_lm 不适用（官方无 MLX 版）→
   对拍官方 inference/model.py（CPU fp8/bf16 跑不动 552B，用我们的 mini+真权重前 2 层对拍）
6. 文档 NOTES_mlx_ds41.md + CHANGELOG

## 4. 已知风险

- **IQ2_XXS 转换损失**：IQ2→mxfp4 是重量化（2bit→4bit 只会更好），但 dequant 必须与
  llama.cpp 的 IQ2_XXS 逐位一致才无损——转换器要用 llama.cpp 的查找表（iq2_data）
- Indexer 的 fp4 量化（e2m1 组 32 e8m0）在 v1 里可先跳过（我们全精度算 indexer，选择结果
  与官方在边界 case 可能差 1-2 个位置 → 长上下文 output 会分叉；≤16K 上下文影响面小，文档标注）
- Engram 哈希函数在 engram.py（NgramHashState），需要 tokenizer 配合；mini 对拍先行
- gate_temp 真实值待从 inference/config.json 核对（默认 1.0）
