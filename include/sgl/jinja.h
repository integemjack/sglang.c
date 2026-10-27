/* sgl/jinja.h — Jinja2 子集渲染器（只为 chat_template 服务）
 *
 * 上游 sglang 用 HF tokenizers 的 apply_chat_template（Jinja2 渲染 tokenizer_config.json 的
 * chat_template）。本项目零依赖，故自带一个小型解释器，覆盖主流模板用到的语法：
 *   {{ expr }} / {%- if/elif/else/endif -%} / {% for x in expr %}...{% else %}...{% endfor %}
 *   {% set x = expr %} / {% set ns.attr = expr %} / namespace(...)
 *   表达式：属性/下标/切片、比较、and/or/not、in、is defined/string/false、算术、
 *           字符串方法（startswith/endswith/split/strip/rstrip/lstrip/lower/upper/replace）、
 *           过滤器（tojson/length/default/trim/string/first/last/join/upper/lower/int/list）
 * 不支持：宏、include/import、set 块、自定义过滤器、模板继承。
 */
#ifndef SGL_JINJA_H
#define SGL_JINJA_H

#include <stddef.h>

#include "sgl/json.h"

typedef struct sgl_jinja sgl_jinja;

/* 读 <dir>/tokenizer_config.json 的 chat_template（字符串，或 [{name,template}...] 取第一个）。
 * 没有模板 / 解析失败返回 NULL（调用方应退回内置 ChatML 拼接）。 */
sgl_jinja *sgl_jinja_load_dir(const char *dir);

/* 直接给模板文本（测试用） */
sgl_jinja *sgl_jinja_from_text(const char *tpl, size_t len);

void sgl_jinja_free(sgl_jinja *j);

/* 渲染。messages: JSON 数组（role/content/tool_calls...，可为 NULL）；
 * kwargs: 额外变量（如 tools / enable_thinking），可为 NULL。
 * add_generation_prompt 注入同名变量（true/false）。
 * 成功返回 0 并把 malloc 的字符串写到 *out（*out_len 为长度，不含结尾 NUL），失败返回非 0。 */
int sgl_jinja_render(const sgl_jinja *j, const sgl_json *messages, int add_generation_prompt,
                     const sgl_json *kwargs, char **out, size_t *out_len);

/* 模板源文本（调试/日志用），非 NULL */
const char *sgl_jinja_source(const sgl_jinja *j);

/* tokenizer_config.json 里的 bos/eos（没有则为空串）*/
const char *sgl_jinja_bos(const sgl_jinja *j);
const char *sgl_jinja_eos(const sgl_jinja *j);

#endif