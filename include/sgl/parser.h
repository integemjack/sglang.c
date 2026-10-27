/* sgl/parser.h — 输出侧解析（对应上游 sglang srt/parser/）
 *
 * 两块能力：
 *   1. reasoning 分离：DeepSeek-R1 / Qwen3 的 `思考...正文`，
 *      以及 Qwen3 的 `<|end_of_thinking|>` 变体；
 *   2. tool call 提取：Qwen/Hermes 风格的 `<tool_call>{"name":...}</tool_call>`
 *      （也认 ```json 包裹与裸 JSON 数组）。
 *
 * 都是纯字符串处理，不改动 text（结果用 sgl_xmalloc，调用方 sgl_xfree）。
 */
#ifndef SGL_PARSER_H
#define SGL_PARSER_H

#ifdef __cplusplus
extern "C" {
#endif

/* 从 text 里分离 reasoning。
 * 命中分隔符（默认 `` 或 `<|end_of_thinking|>`）时：
 *   *reasoning = 分隔符之前的文本（拷贝，调用方 sgl_xfree；可能为空串）
 *   *content   = 分隔符之后的正文（text 内部的指针，不拥有）
 * 未命中：*reasoning = NULL，*content = text。
 * 返回 1 = 分离成功，0 = 没有 reasoning 段。 */
int sgl_parse_reasoning(const char *text, char **reasoning, const char **content,
                        const char **finish_marker);

/* 提取 tool call。识别 <tool_call>{...}</tool_call>（可多个），
 * 也接受 `<tool_call>[{...},{...}]</tool_call>`。
 * 返回 NULL（没有调用）；否则返回 JSON 数组文本 `[{"name":..,"arguments":..}, ...]`，
 * 调用方 sgl_xfree；*n_calls 给出个数。 */
char *sgl_parse_tool_calls(const char *text, int *n_calls);

#ifdef __cplusplus
}
#endif

#endif