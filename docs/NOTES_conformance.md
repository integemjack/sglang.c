# 与上游 sglang 的语义对拍（conformance）— 套件与三处修正

2026-11-05。对应 GAP_vs_upstream §5 第 7 项的「语义」半边（性能基准仍开）。

## 套件：`tests/conformance_upstream.py`

对照物两类（都不需要跑上游服务）：

1. **HF transformers（本机 pip）当真值**：`/tokenize`==`encode`、`/detokenize`==`decode`
   （4 基准 + 16 毒样本：多空格/数字串/缩写/中英混排/悬挂空白）；
   chat 模板用**行为等价法**验证——`/v1/chat/completions` 的输出必须 ==
   「HF `apply_chat_template` 渲染出的 prompt + `/v1/completions` 贪心续写」。
2. **上游源码语义编成断言**：stop 截断（不含停止串、finish=stop、裸字符串等价、
   流式拼装==非流式、长词跨 token 不吐前缀）、finish_reason（max_tokens→length）、
   SSE 帧序（恰一个 finish 帧 + `[DONE]`）、`stream_options.include_usage` 的
   空 choices usage 帧、`/generate` 原生 meta_info、usage 数字对得上 HF 编码。

用法：`python tests/conformance_upstream.py`（自己起 CPU 服务、跑完自己收；
`--model/--port/--keep` 可调）。**退出码 0=全过**。当前 35 项全绿。

## 顺带抓出的三个真 bug（都已修）

1. **Qwen 系预分词错**（`src/bpe.c`）：我们只实现了 GPT-2 经典款；Qwen2/2.5/3
   的 Split Regex 是另一族——单个前导符号可挂字母段（`"(x"` 一个块）、数字按
   单码点切（`123`→3 块）、符号串吞尾部换行（`"):\n"` 一块）、`\s*[\r\n]+`。
   后果：缩进代码 / `def f(x):` 一类文本 token 序列与 HF 不同（实测
   `def f(x):\n    return x + 1\n`：13 个 token vs HF 12 个，`(x`/`):\n` 被拆散）。
   修法：加载时按 tokenizer.json 里 Split Regex 的 `[^\r\n\p{L}\p{N}]?\p{L}+`
   签名派发到新写的 `scan_next_qwen`（码点区间表实现 \p{L}/\p{N}，冷门文字落
   符号类——切分合法但可能与 HF 不同，注释已声明）；GPT-2 经典/Metaspace/
   Unigram 路径零改动。16 条毒样本逐 id 对齐 HF。
2. **流式路径完全没应用 stop**（`src/http.c`）：非流式 `apply_stops` 截断正确，
   流式把全部 token 原样吐出（实测 stop=`the`：非流式 113 字符截断，流式 227
   全吐）。修法：流式分支加累积缓冲——全文最早命中即截（含 `sgl_sched_abort`
   提前掐掉后台生成）；**尾部悬挂回扣**：缓冲尾部凑出某停止串真前缀的字符扣住
   不发（`th` 可能变 `the`），不命中再放行。
3. **流式两个协议缺口**：finish 帧硬编码 `"stop"`（达到 max_tokens 应为
   `"length"`）；`stream_options.include_usage` 不支持（上游在 `[DONE]` 前发
   空 choices 的 usage 帧）。均已实现（usage 的 prompt_tokens 用
   `sgl_sched_encode` 现算，completion_tokens 按流出 token 计数）。

## 回归

串行（本机 ctest 严禁 `-j`，见记忆 m23471）：test_bpe / test_http / test_unigram /
test_gguf / test_scheduler / test_spec / test_beam / test_tinyllama /
test_family_qwen3 / test_family_qw_marlin8 / test_family_mistral — 11/11 绿。
BPE 修复同时惠及 GPU 栈（共享同一 tokenizer 库）。

## 已知剩余（GPU 服务面，归到后续 cuda_http 工作）

`apps/accel_http.c`（CUDA/MLX 服务的 HTTP 栈，独立于 src/http.c）仍缺
**stop 字符串**与 **stream_options.include_usage**——与批量 spec 一并补
（同文件），届时用 `tests/conformance.py`（cpu vs gpu）跑双栈对拍。
