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
| G16 W8 重测 + flash 泛化 | 未开始 | qwen05 在 `../models/qwen05` |
| G14 投机正收益 | 未开始 | tinyllama + llama-160m 都在 `../models` |
| G12 真实 llava-1.5 | 未开始 | 需下 ~4GB 权重（`../models/llava-1.5-7b` 之类） |
| G15 前缀缓存 radix | 未开始 | 纯代码 + 随机模型可验 |
| G13 Qwen3-Next full 层 | 未开始 | 纯代码 + 随机模型可验 |
| G18 8K GPU-vs-CPU 绝对对照 | 未开始 | qwen05 在，需 40–60 分钟干净时段 |

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

## 下一轮从这里继续

1. **G16**：qwen05 在 `../models/qwen05`。按文档：先跑一遍 W8 IMMA 后的 BENCH 行
   （`SGL_CUDA_W8=1`），再看 flash 在 gemma2/qwen3next/mistral 随机模型上的泛化
   （阈值不变、逐族记 maxerr）。
2. **G14**：tinyllama（target）+ llama-160m（draft）都在 `../models`，直接能复跑
   接受率与 tok/s。
3. G15 / G13 是纯代码任务，随机模型就能验；G12 需要先下 llava-1.5 权重；
   G18 要一段 40–60 分钟的干净时段。