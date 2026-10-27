/* sgl/json.h — 最小 JSON 解析器（tokenizer.json / config.json / safetensors header 用）
 * 递归下降，只解析不改写；字符串按 UTF-8 原样保留（\uXXXX 转成 UTF-8）。 */
#ifndef SGL_JSON_H
#define SGL_JSON_H

#include <stddef.h>

typedef enum {
  SGL_JSON_NULL,
  SGL_JSON_BOOL,
  SGL_JSON_NUMBER, /* 统一 double；整数值在 2^53 内无损 */
  SGL_JSON_STRING,
  SGL_JSON_ARRAY,
  SGL_JSON_OBJECT,
} sgl_json_type;

typedef struct sgl_json sgl_json;

struct sgl_json {
  sgl_json_type type;
  /* 标量 */
  double num;
  int boolean;
  char *str; /* SGL_JSON_STRING：以 NUL 结尾的 UTF-8，调用方不释放 */
  /* 容器 */
  sgl_json **items;      /* ARRAY / OBJECT 的值数组 */
  char **keys;           /* 仅 OBJECT：与 items 等长 */
  int count;
};

/* 解析失败返回 NULL（错误信息打到日志）*/
sgl_json *sgl_json_parse(const char *text, size_t len);
void sgl_json_free(sgl_json *j);

/* 取值辅助（找不到返回 NULL / def）*/
const sgl_json *sgl_json_get(const sgl_json *obj, const char *key);
const char *sgl_json_get_str(const sgl_json *obj, const char *key, const char *def);
double sgl_json_get_num(const sgl_json *obj, const char *key, double def);
int sgl_json_get_int(const sgl_json *obj, const char *key, int def);
/* JSON 布尔（true/false）；也给数字（非 0 为真）。JSON 里开关常是布尔值。*/
int sgl_json_get_bool(const sgl_json *obj, const char *key, int def);

#endif
