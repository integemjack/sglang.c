/* sgl/grammar.h — 约束解码 / 结构化输出（grammar engine）
 *
 * 对应上游 sglang 的 srt/constrained（xgrammar / outlines）：把 JSON Schema 或简单正则
 * 编译成「字节级自动机」，在采样时屏蔽掉会把输出带出合法前缀的 token（token 级掩码）。
 *
 * 典型用法：
 *   sgl_grammar *g = sgl_grammar_from_json_schema(schema, len);
 *   if (!g) { fprintf(stderr, "%s\n", sgl_grammar_error(g)); ... }
 *   sgl_grammar_set_vocab(g, pieces, lens, npieces);      // 每次换词表重绑
 *   for (;;) {
 *     sgl_grammar_mask(g, cand_ids, n, allowed);          // 1=允许 0=禁止
 *     int tok = 从 allowed 里采样;
 *     if (sgl_grammar_accept(g, tok) != 0) continue;      // 0=合法并推进，状态不变则拒绝
 *     if (sgl_grammar_can_finish(g)) break;
 *   }
 *   sgl_grammar_free(g);
 *
 * 实现要点（详见 grammar.c 顶部注释）：
 *   - schema / 正则创建时编译成不可变的节点树，运行时状态是一个「帧栈」（下推自动机）；
 *   - 栈顶帧决定下一个字节是否合法；非法字节不会改变状态（内部先复制状态再提交）；
 *   - token 级掩码 = 复制状态 + 逐字节喂入，任一字节非法该 token 即被禁止；
 *   - 复杂度：accept 单字节 O(栈深)；mask 每个候选 token O(token 字节数 × 栈深)，
 *     栈深只随 JSON 嵌套层数增长（数组/重复项不加深，重复用同一个 REP 帧）。
 *
 * 支持的 JSON Schema 子集：
 *   type（含 type 数组）/ object+properties+required+additionalProperties /
 *   array+items+minItems+maxItems / string+minLength+maxLength+pattern /
 *   number+integer+minimum+maximum / enum / const / anyOf / oneOf（按 anyOf 处理）
 * 不支持的：allOf、$ref、not、if/then/else、exclusiveMinimum/Maximum、patternProperties、
 *   propertyNames、uniqueItems、multipleOf、format —— allOf/$ref 会让
 *   sgl_grammar_from_json_schema 返回 NULL 并给出原因，其余关键字静默忽略（不收紧约束）。
 *
 * 已知限制（有意为之，见 grammar.c 里各自位置的注释）：
 *   - additionalProperties 不是 false（含缺省）时退化为「任意键名 + 任意 JSON 值」，
 *     且此时不强制 required；
 *   - REP / CLASS / NUM 的「继续还是结束」按贪心处理，不再回溯（如正则 [0-9]+[0-9] 走贪心路径）；
 *     ALT（enum / anyOf / type 数组）则是真并集语义：同首字节的多个分支逐个回溯尝试，
 *     所以 enum:["x","y"] 这类分支能正确区分；
 *   - 同一对象里禁止重复键（used 位集），属性数上限 60；
 *   - minimum/maximum 对整数按「位数可行性」精确收紧；对小数只校验整数部分可行性；
 *   - 字符串长度按「字符」计（UTF-8 序列算 1 个、\uXXXX 与 \n 等转义算 1 个）。
 */
#ifndef SGL_GRAMMAR_H
#define SGL_GRAMMAR_H

#include <stddef.h>

typedef struct sgl_grammar sgl_grammar;

/* 由 JSON Schema 文本创建文法；schema_text 不需要以 NUL 结尾（按 len 读取）。
 * 失败返回 NULL，原因见 sgl_grammar_error 无法取（此时文法不存在）——直接返回 NULL。 */
sgl_grammar *sgl_grammar_from_json_schema(const char *schema_text, size_t len);

/* 由简单正则创建文法（整串匹配）。支持：字符、转义、.、[...] 字符类（含 ^ 取反、a-z 区间、
 * \d \w \s \D \W \S）、分组 (...)、量词 * + ? {m} {m,} {m,n}、分支 |、锚点 ^ $ 忽略。
 * 失败返回 NULL。 */
sgl_grammar *sgl_grammar_from_regex(const char *pattern);

void sgl_grammar_free(sgl_grammar *g);

/* 绑定词表：token id -> UTF-8 字节片段。piece_bytes[id] 为 NULL 表示该 token 不可用
 * （如特殊 token）；piece_len 为 NULL 时按 strlen 处理。指针由调用方持有，文法不拷贝。 */
void sgl_grammar_set_vocab(sgl_grammar *g, const char *const *piece_bytes, const int *piece_len,
                           int n_pieces);

/* 对 n 个候选 token 给出 1=允许 0=禁止。允许「只是合法前缀」的 token，但完整 token 的每个
 * 字节都必须合法。词表外的 id、piece 为 NULL 的 id 一律置 0。 */
void sgl_grammar_mask(const sgl_grammar *g, const int *token_ids, int n, unsigned char *allowed);

/* 追加一个 token：0=合法并推进；非 0=非法（状态不变）。 */
int sgl_grammar_accept(sgl_grammar *g, int token_id);

/* 追加原始字节：0=合法并推进；非 0=非法（状态不变）。 */
int sgl_grammar_accept_bytes(sgl_grammar *g, const char *bytes, int len);

/* 1=当前状态是完整合法输出，可以停止生成。 */
int sgl_grammar_can_finish(const sgl_grammar *g);

/* 最近一次失败的原因（可能为空串）。 */
const char *sgl_grammar_error(const sgl_grammar *g);

/* 回到初始状态，复用同一个文法（省去每次请求重新解析 schema）。 */
void sgl_grammar_reset(sgl_grammar *g);

#endif /* SGL_GRAMMAR_H */