# G8 — 真实 CLIP 权重对齐 + 图片解码（CPU 路径）

## 1. 真实权重来源

目标目录 `works/sglang/models/clip-vit-base-patch32/`（openai/clip-vit-base-patch32）。

| 文件 | 大小 (B) | SHA256 | 来源 |
|---|---|---|---|
| config.json | 4186 | — | hf-mirror `resolve/main/config.json` |
| vocab.json / merges.txt | 862328 / 524657 | — | hf-mirror |
| preprocessor_config.json | 316 | — | hf-mirror |
| pytorch_model.bin | 605247071 | `A63082132BA4F97A80BEA76823F544493BFFA8082296D62D71581A4FEFF1576F` | ModelScope `openai-mirror/clip-vit-base-patch32`（`resolve/master/`） |
| model.safetensors | 351422734 | `CFCEE916836645BFFD7C4247492517EAE2145F0179EA81620920318116B7E5DB` | **本任务转换产物**（视觉塔 201 个张量，F32） |
| test.png / test.jpg | 305992 / 16390 | — | `tests/clip_real_ref.py` 生成 |

要点：

- 该仓库**没有 safetensors 变体**：hf-mirror 的 `resolve/main/model.safetensors`
  返回 404（`Cannot find the requested files in the cached path`…实际 body 是
  `Entry not found`，15 B），所以按 `pytorch_model.bin` 取权重。
- hf-mirror 直下 `.bin` 只有 ~200 KB/s（12 分钟没下完 1/3），改用 ModelScope 镜像
  53 s 下完（10.7 MB/s）。
- `model.safetensors` 是 `tests/clip_real_ref.py` 从 `.bin` 抽
  `vision_model.*` + `visual_projection.weight` 转成 F32 写的（C 侧 safetensors
  loader 只认这个格式）。抽查 4 个张量（q_proj[0]、class_embedding、
  post_layernorm.weight、visual_projection.weight）与原 `.bin` **逐字节相同**
  （maxdiff 0.0）。文本塔 / text_projection / logit_scale 未搬（C 侧用不到）。

config（vision_config）：`hidden_size=768 num_heads=12 layers=12 patch=32
image_size=224 intermediate=3072 layer_norm_eps=1e-5 hidden_act=quick_gelu
projection_dim=512`。

## 2. C 视觉塔与真实 CLIP 的结构性差异（已修正）

对照 HF `transformers/models/clip/modeling_clip.py`（本机 transformers 5.20.2）：

| # | 旧 t10 实现 | 真实 CLIP | 处理 |
|---|---|---|---|
| 1 | MLP 用 erf GELU | `quick_gelu` = x·sigmoid(1.702x) | `sgl_clip_act()`，按 config `hidden_act` 选（QUICK/GELU_ERF/GELU_TANH） |
| 2 | post_layernorm 作用**全部行**，再丢 class 行 | `last_hidden_state` = encoder 输出（**不过** post_ln）；`pooled_output` = post_ln(class 行) | `sgl_clip_encode_full()`：patch 行直接输出，pooled 单独算 |
| 3 | `visual_projection` 作用 patch 行 | 只作用 pooled → `image_embeds`[512] | 同上 |
| 4 | `encode` 输出 `[NP][projection_dim]` | LLaVA 取 `last_hidden_state[:,1:,:]` = `[NP][768]` | `proj_dim` 字段改为恒等于 E（保留字段给 cuda_http 算 buffer），新增 `img_proj_dim` |
| 5 | 张量名 `embeddings.position_embedding`、`embeddings.pre_layernorm.*` | `embeddings.position_embedding.weight`、`pre_layrnorm.*`（HF 的 typo 名） | 两套名字都试 |
| 6 | 要求 `patch_embedding.bias` 必须存在 | 真实 ckpt **没有**（conv bias=False） | 缺失时合成 0（`sgl_clip.synth` 标记，`sgl_clip_free` 回收） |
| 7 | eps 硬编码 1e-5 | 恰好一致，但应可配 | 读 config `layer_norm_eps` |
| 8 | 只认 `clip_config.json` + `clip_model.safetensors` | HF 是 `config.json`(vision_config) + `model.safetensors` | 都支持（按 config 来源决定先试哪个权重文件，避免刷 open failed 日志） |

随机权重测试同步更新：`gen_random_llava.py` 改成 HF 命名（`position_embedding.weight`
/ `pre_layrnorm.*` / `hidden_act=quick_gelu`）、projector 输入维从 PROJ 改 E；
`ref_llava.py` 的 numpy 参考同步 quick_gelu + pooled 语义。

## 3. 分层对照（真实权重，torch 2.14.1+cpu fp32 vs C fp32）

参考：`tests/clip_real_ref.py` 用 `CLIPVisionModel` 逐层导出
（`manual_vs_forward_last_maxdiff = 0.0`，即手工逐层与模型自身 forward 完全一致；
`last_eq_encoder_out_only = true` 证实 last_hidden_state 就是 encoder 输出）。
C 侧：`tests/test_clip_real.c`（`ctest -R test_clip_real`，输入 `ref/pv.npy`）。

| 层 | maxerr |
|---|---|
| pre_layrnorm 输出（encoder 输入） | 7.15e-06 |
| layer 0..8 | 5.48e-06 / 2.64e-06 / 2.24e-06 / 2.03e-06 / 1.59e-06 / 2.74e-06 / 3.28e-06 / 3.87e-06 / 4.17e-06 |
| layer 9 / 10 / 11 | 1.24e-05 / 1.14e-05 / 1.14e-05 |
| last_hidden_state[1:]（喂 LLaVA 的视觉 token） | 1.14e-05 |
| pooled_output（post_ln(cls)） | 8.11e-06 |
| image_embeds（visual_projection(pooled)，\|ref\|max=8.27） | 6.68e-06 |

阈值依据：两边都是 fp32、同一套权重，差异只来自算子求和顺序与 LN 的
mean/var 公式（C 用 double 累加），数值尺度 \|x\| 到 ~30，实测逐层最大 1.24e-05，
取任务给的 1e-3（约 80× 余量）作逐层阈值，最终投影 1e-2（实测 6.68e-06）。
随机权重侧 `test_llava` 与 numpy 参考仍逐位级：feats 7.15e-07、
image_embeds 4.33e-08、projector 1.04e-07。

## 4. 图片解码

- `third_party/stb/stb_image.h`（nothings/stb master，276.3 KB，MIT），
  只在 `src/image.c` 里 `#define STB_IMAGE_IMPLEMENTATION`。
- `sgl_image_decode_rgb()`：JPEG/PNG → RGB（不做 YCbCr 之外的色彩管理，
  16-bit PNG 降 8-bit，alpha 丢弃）。
- `sgl_clip_preprocess()`：复刻 PIL `Image.resize(BICUBIC)` + CLIP 的
  resize→center_crop→/255→(x-mean)/std。关键细节（都是实测出来的）：
  - 核 = Catmull-Rom（a=-0.5），support=2，`center=(o+0.5)*scale-0.5`，
    `lo=floor(center-2+0.5)`，权重**不夹负**、除以 Σ（PIL 用系数和归一，
    不是纯核值）；
  - 两 pass 的中间结果是 **8-bit**（`ImagingResampleHorizontal_8bit` 写回 UINT8）：
    按 float 中间结果与 PIL 在强边缘差到 9 个像素级，改回 8-bit 后降到 1 级；
  - CLIP 4.x 的 `size=224` 是直接缩到 224×224；transformers 5.x 的 processor 是
    短边 224 + center_crop。**方形输入两者等价**，测试图用方形；非方形时
    两者差 3.7（归一化空间），已在 ref 脚本里注明。
- 与 PIL 实测（320×320 图 → 224×224）：

| 用例 | maxerr | 阈值 | 说明 |
|---|---|---|---|
| test.png 解码 vs PIL 解码 | **0 像素级** | 必须为 0 | PNG 无损，逐字节全等 |
| test.png 全链 → 归一化 CHW | **0.015** | 0.03 | = 1 个像素级（1/255/std≈0.0146），resize 定点舍入 |
| test.jpg 解码 vs PIL(libjpeg) | **2 像素级** | 只报数 | stb 与 libjpeg 的色度/IDCT 重建差 |
| test.jpg 全链 → 归一化 CHW | **0.03** | 0.30 | 由上面 2 级主导 |
| base64 往返（305992 B） | 全等 | — | `sgl_base64_decode` |

  测试图内容：PNG 用白噪声（压 resize 高频），JPEG 用平滑渐变图（噪声经
  4:2:0+DCT 后内容本身就变了，无法比解码器）。

## 5. HTTP 多模态入口（只动 CPU 路径）

- `include/sgl/http.h`：`sgl_http_vision_fn` + `sgl_http_set_vision()`。
- `src/http.c`：`/v1/chat/completions` 的 `content[]` 里支持
  `{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,..."}}`
  → base64 解码 → 回调（`sgl_clip_project_bytes`：解码→预处理→CLIP→projector）
  → 多张图按行拼接 → `sgl_sched_submit_img()`；文本里插 `<image>` 占位。
  明确拒的组合返 400（未注册视觉塔、坏 base64、stream+图、n>1+图、带 lora+图）。
- `src/scheduler.c`：`sgl_req` 加 `feats/n_img`；带图请求**不走 RadixCache**
  （KV 里混了图片 emb，前缀语义不同），prompt token prefill 后接
  `sgl_model_prefill_emb()`，首 token logits 取最后一个视觉 token；
  feats 在 prefill 完即 `sgl_xfree`。
- `src/model.c/h`：`sgl_model_prefill_emb()` → llama 走
  `sgl_llama_forward_emb`，gpt2 返 -1。
- `apps/sglc.c`：`--vision <clip目录> [--projector <projector.safetensors>]`
  （默认 `<clip目录>/projector.safetensors`），E_lm 从模型 config.json 的
  hidden_size/n_embd 读。
- **未动**：`apps/cuda_http.c`、`src/cuda_backend.cu`、`tests/cuda_test.c`。
  GPU 侧接线时要把 `src/image.c` 加进 CMakeLists 的 `SGL_CORE_SRC`
  并给 `sgl_cuda` 加 `third_party/stb` include（现在没加，避免碰你在用的 CUDA 栈）。

e2e（`tests/http_image_e2e.py`，sglc --serve --vision + requests）：
JPEG/PNG/纯文本三种请求都 **200**，同 seed greedy 下输出随图片改变
（jpeg: `' Aboveเกินไป HdMex…'`，png: `')):\n supporters…'`，无图: `' location芠…'`），
坏 base64 → `400 {"error":{"message":"image_url base64 decode failed"}}`。

## 6. 验收

- `ctest --test-dir build-g8`：**23/23 Passed**（原 20 + test_clip_real + G9 的
  gemma2/qwen3next 等）。
- `cmake --build build-cuda --target sgl_cuda`（MSVC）：通过，
  `Release/sgl_cuda.lib` 生成（clip.c 已 MSVC-clean：无 statement-expression、
  无 `sgl_tensor x = 0` 之类，`false` 不再出现）。
- 全仓 `.c/.h/.cu` 的 UTF-8 BOM 已补齐（曾发现 31 个文件丢 BOM，含我没碰过的
  tests/*、include/sgl/cuda.h，已统一补回 EF BB BF）。

## 7. 未尽事项

1. ~~GPU 侧（cuda_http / cuda_backend）没接图片解码~~ → **G11 已接**（cuda_http 按 mime
   分派 PNG/JPEG 走 `sgl_clip_project_bytes_pj`，见 CHANGELOG 的 G11）。
2. ~~projector 仍是单层 linear~~ → **G12 已支持两层 MLP**（`sgl_projector`，从权重形状
   自动判层数，GELU 用 erf 版）；真实 llava-1.5 的 mm_projector 已直接对上（见 §8）。
3. ~~`<image>` 占位没有做「按占位位置插入视觉 token」~~ → **G12 已做**（scheduler 把
   视觉 token 插在占位符位置，尾巴文本位置顺延；占位符 token 本身不进 KV）。
   多图仍拼在**同一个**占位位置（llava 是一图一占位，本实现在多图时与它不同）。
4. `test_clip_real` 只在 `models/clip-vit-base-patch32/ref/` 存在时注册
   （fresh clone 不会红）；重跑参考要 `python tests/clip_real_ref.py <clip目录>`
   （需 transformers 5.20.2，本任务已 pip 安装）。
5. 真实 LLaVA **端到端**（CLIP 336 + llava-1.5 权重 + 语言模型）仍未做：llava-1.5-7B 的
   LM 是 fp16 13.5GB，本机剩余磁盘放不下（见 §8）。已改用任务允许的「分段对照」：
   真实 CLIP-L/14-336 + 真实 mm_projector 全链路对齐。
6. ctest 首轮曾有 9 项 `0xc0000409`（alloc.c out-of-memory 后 fastfail）：
   本机当时被其他 agent 的测试占满内存（Free 11 GB、单测要 mmap 数百 MB），
   单独跑与重跑全绿，非代码问题。

## 8. G12：真实权重分段对照（CLIP-L/14-336 + 真实 mm_projector）

权重来源（hf-mirror，2026-10-27）：

| 文件 | 来源 | 大小 |
|---|---|---|
| `models/clip-vit-large-patch14-336/pytorch_model.bin` | openai/clip-vit-large-patch14-336 | 1.71 GB |
| → `clip_model.safetensors`（392 个 vision_model.* 张量，fp32） | tests/convert_real_llava.py | 1.21 GB |
| `models/llava-real/mm_projector.bin` | liuhaotian/llava-v1.5-7b | 42 MB（fp16） |
| → `models/llava-real/projector.safetensors`（linear_1/linear_2，fp32） | 同上 | 84 MB |

真实 llava 的 config 关键字段：`mm_projector_type=mlp2x_gelu`、`mm_hidden_size=1024`、
`mm_vision_tower=openai/clip-vit-large-patch14-336`、**`mm_vision_select_layer=-2`**
（取倒数第二层 hidden state，不是最后一层）、`image_aspect_ratio=pad`。

C 侧新增：`sgl_clip` 多了 `sel_layer`（读 config 的 `mm_vision_select_layer` / `select_layer`，
也接受 `SGL_CLIP_LAYER` 覆盖，负数按 `L+sel` 归一）；`sgl_clip_encode` 在那一层拷贝
patch 行。已验证 select_layer=22（L=24）。

对照（同一张 336×336 图，torch 2.14.1+cpu fp32 vs C fp32，`tests/llava_real_ref.py`
生成 `_ref_pv/_ref_vis/_ref_real_proj.npy`）：

| 阶段 | 绝对 maxerr | 参考最大绝对值 | 相对 |
|---|---|---|---|
| CLIPImageProcessor 预处理（CHW 归一化） | 2.38e-07 | 2.07 | 1.15e-07 |
| CLIP 视觉塔 patch 行（576×1024，第 22 层） | 2.78e-03 | 197 | **1.41e-05** |
| mm_projector（576×4096，1024→4096→4096 + GELU） | 6.60e-04 | 37.1 | **1.78e-05** |

`ctest -R test_llava_real`：Passed（17.5s，主要花在 mmap 1.2GB 权重 + 576×1024×4096
的 CPU GEMM）。全量 ctest **26/26**。

**仍未做**：真实 7B 语言模型的端到端（磁盘不够）；换成一个小真实 LM（如
TinyLlama-1.1B）与 llava 的 CLIP+projector 拼接的混合实验也没做——那是另一个模型，
不是 llava-1.5 的真实行为，所以没当验收。
