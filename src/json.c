/* sgl json 实现：递归下降解析器 */
#define _CRT_SECURE_NO_WARNINGS
#include "sgl/json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sgl/log.h"

/* 本模块内部一律用普通 malloc/realloc/free（无需对齐），与 sgl_json_free 一致 */

typedef struct {
  const char *p;
  const char *end;
  int depth;
} parser;

static sgl_json *parse_value(parser *ps);

static void skip_ws(parser *ps) {
  while (ps->p < ps->end &&
         (*ps->p == ' ' || *ps->p == '\t' || *ps->p == '\n' || *ps->p == '\r'))
    ps->p++;
}

static sgl_json *jnew(sgl_json_type type) {
  sgl_json *j = (sgl_json *)calloc(1, sizeof(sgl_json));
  if (!j) abort();
  j->type = type;
  return j;
}

void sgl_json_free(sgl_json *j) {
  if (!j) return;
  free(j->str);
  for (int i = 0; i < j->count; i++) {
    sgl_json_free(j->items[i]);
    if (j->keys) free(j->keys[i]);
  }
  free(j->items);
  free(j->keys);
  free(j);
}

/* 把 \uXXXX 解成 UTF-8 追加到 sb */
static void utf8_append(char **sb, size_t *len, size_t *cap, unsigned cp) {
  unsigned char buf[4];
  int n;
  if (cp < 0x80) { buf[0] = (unsigned char)cp; n = 1; }
  else if (cp < 0x800) {
    buf[0] = (unsigned char)(0xC0 | (cp >> 6));
    buf[1] = (unsigned char)(0x80 | (cp & 0x3F));
    n = 2;
  } else if (cp < 0x10000) {
    buf[0] = (unsigned char)(0xE0 | (cp >> 12));
    buf[1] = (unsigned char)(0x80 | ((cp >> 6) & 0x3F));
    buf[2] = (unsigned char)(0x80 | (cp & 0x3F));
    n = 3;
  } else {
    buf[0] = (unsigned char)(0xF0 | (cp >> 18));
    buf[1] = (unsigned char)(0x80 | ((cp >> 12) & 0x3F));
    buf[2] = (unsigned char)(0x80 | ((cp >> 6) & 0x3F));
    buf[3] = (unsigned char)(0x80 | (cp & 0x3F));
    n = 4;
  }
  if (*len + n + 1 > *cap) {
    *cap = (*cap * 2) + 16 + n;
    *sb = (char *)realloc(*sb, *cap);
    if (!*sb) abort();
  }
  memcpy(*sb + *len, buf, n);
  *len += n;
  (*sb)[*len] = 0;
}

static void sb_push(char **sb, size_t *len, size_t *cap, char c) {
  /* 原样字节直接写入（UTF-8 多字节序列不经重新编码）；
   * 误用 utf8_append 会把 >=0x80 的字节二次编码成 2 字节 */
  if (*len + 2 > *cap) {
    *cap = (*cap * 2) + 16;
    *sb = (char *)realloc(*sb, *cap);
    if (!*sb) abort();
  }
  (*sb)[(*len)++] = c;
  (*sb)[*len] = 0;
}

/* 解析字符串字面量（含转义），返回 malloc 的 UTF-8 */
static char *parse_string_raw(parser *ps) {
  if (ps->p >= ps->end || *ps->p != '"') return NULL;
  ps->p++;
  size_t cap = 32, len = 0;
  char *sb = (char *)malloc(cap);
  if (!sb) abort();
  sb[0] = 0;
  while (ps->p < ps->end && *ps->p != '"') {
    char c = *ps->p++;
    if (c == '\\') {
      if (ps->p >= ps->end) goto fail;
      char e = *ps->p++;
      switch (e) {
      case '"': sb_push(&sb, &len, &cap, '"'); break;
      case '\\': sb_push(&sb, &len, &cap, '\\'); break;
      case '/': sb_push(&sb, &len, &cap, '/'); break;
      case 'b': sb_push(&sb, &len, &cap, '\b'); break;
      case 'f': sb_push(&sb, &len, &cap, '\f'); break;
      case 'n': sb_push(&sb, &len, &cap, '\n'); break;
      case 'r': sb_push(&sb, &len, &cap, '\r'); break;
      case 't': sb_push(&sb, &len, &cap, '\t'); break;
      case 'u': {
        if (ps->end - ps->p < 4) goto fail;
        char hex[5] = {ps->p[0], ps->p[1], ps->p[2], ps->p[3], 0};
        char *endp = NULL;
        unsigned cp = (unsigned)strtoul(hex, &endp, 16);
        if (endp != hex + 4) goto fail;
        ps->p += 4;
        /* 代理对 */
        if (cp >= 0xD800 && cp <= 0xDBFF && ps->end - ps->p >= 6 &&
            ps->p[0] == '\\' && ps->p[1] == 'u') {
          char hex2[5] = {ps->p[2], ps->p[3], ps->p[4], ps->p[5], 0};
          char *endp2 = NULL;
          unsigned lo = (unsigned)strtoul(hex2, &endp2, 16);
          if (endp2 == hex2 + 4 && lo >= 0xDC00 && lo <= 0xDFFF) {
            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
            ps->p += 6;
          }
        }
        utf8_append(&sb, &len, &cap, cp);
        break;
      }
      default: goto fail;
      }
    } else {
      sb_push(&sb, &len, &cap, c); /* UTF-8 原样字节 */
    }
  }
  if (ps->p >= ps->end) goto fail;
  ps->p++; /* 结尾引号 */
  return sb;
fail:
  free(sb);
  return NULL;
}

static void jpush(sgl_json *arr, char *key, sgl_json *val) {
  sgl_json **ni = (sgl_json **)realloc(arr->items, (arr->count + 1) * sizeof(sgl_json *));
  if (!ni) abort();
  arr->items = ni;
  if (key || arr->keys) {
    char **nk = (char **)realloc(arr->keys, (arr->count + 1) * sizeof(char *));
    if (!nk) abort();
    arr->keys = nk;
    arr->keys[arr->count] = key;
  }
  arr->items[arr->count] = val;
  arr->count++;
}

static sgl_json *parse_object(parser *ps) {
  sgl_json *obj = jnew(SGL_JSON_OBJECT);
  ps->p++; /* { */
  skip_ws(ps);
  if (ps->p < ps->end && *ps->p == '}') { ps->p++; return obj; }
  for (;;) {
    skip_ws(ps);
    char *key = parse_string_raw(ps);
    if (!key) goto fail;
    skip_ws(ps);
    if (ps->p >= ps->end || *ps->p != ':') { free(key); goto fail; }
    ps->p++;
    sgl_json *val = parse_value(ps);
    if (!val) { free(key); goto fail; }
    jpush(obj, key, val);
    skip_ws(ps);
    if (ps->p < ps->end && *ps->p == ',') { ps->p++; continue; }
    if (ps->p < ps->end && *ps->p == '}') { ps->p++; return obj; }
    goto fail;
  }
fail:
  sgl_json_free(obj);
  return NULL;
}

static sgl_json *parse_array(parser *ps) {
  sgl_json *arr = jnew(SGL_JSON_ARRAY);
  ps->p++; /* [ */
  skip_ws(ps);
  if (ps->p < ps->end && *ps->p == ']') { ps->p++; return arr; }
  for (;;) {
    sgl_json *val = parse_value(ps);
    if (!val) goto fail;
    jpush(arr, NULL, val);
    skip_ws(ps);
    if (ps->p < ps->end && *ps->p == ',') { ps->p++; continue; }
    if (ps->p < ps->end && *ps->p == ']') { ps->p++; return arr; }
    goto fail;
  }
fail:
  sgl_json_free(arr);
  return NULL;
}

static sgl_json *parse_value(parser *ps) {
  if (++ps->depth > 200) { SGL_ERROR("json too deep"); return NULL; }
  skip_ws(ps);
  if (ps->p >= ps->end) return NULL;
  sgl_json *out = NULL;
  char c = *ps->p;
  if (c == '{') out = parse_object(ps);
  else if (c == '[') out = parse_array(ps);
  else if (c == '"') {
    char *s = parse_string_raw(ps);
    if (s) {
      out = jnew(SGL_JSON_STRING);
      out->str = s;
    }
  } else if (c == 't' && ps->end - ps->p >= 4 && strncmp(ps->p, "true", 4) == 0) {
    out = jnew(SGL_JSON_BOOL);
    out->boolean = 1;
    ps->p += 4;
  } else if (c == 'f' && ps->end - ps->p >= 5 && strncmp(ps->p, "false", 5) == 0) {
    out = jnew(SGL_JSON_BOOL);
    out->boolean = 0;
    ps->p += 5;
  } else if (c == 'n' && ps->end - ps->p >= 4 && strncmp(ps->p, "null", 4) == 0) {
    out = jnew(SGL_JSON_NULL);
    ps->p += 4;
  } else {
    /* 数字 */
    char *endp = NULL;
    double d = strtod(ps->p, &endp);
    if (endp == ps->p) return NULL;
    out = jnew(SGL_JSON_NUMBER);
    out->num = d;
    ps->p = endp;
  }
  ps->depth--;
  return out;
}

sgl_json *sgl_json_parse(const char *text, size_t len) {
  parser ps = {text, text + len, 0};
  sgl_json *j = parse_value(&ps);
  if (!j) {
    char ctx[80];
    int cn = 0;
    const char *q = ps.p;
    for (int i = 0; i < 64 && q < ps.end && cn < 70; i++, q++) {
      unsigned char c = (unsigned char)*q;
      if (c >= 0x20 && c < 0x7F) ctx[cn++] = (char)c;
      else cn += snprintf(ctx + cn, sizeof(ctx) - cn, "<%02X>", c);
    }
    ctx[cn] = 0;
    SGL_ERROR("json parse failed near byte %td: ...%s", (ptrdiff_t)(ps.p - text), ctx);
    return NULL;
  }
  skip_ws(&ps);
  if (ps.p != ps.end) {
    SGL_ERROR("json trailing garbage at byte %td", (ptrdiff_t)(ps.p - text));
    sgl_json_free(j);
    return NULL;
  }
  return j;
}

const sgl_json *sgl_json_get(const sgl_json *obj, const char *key) {
  if (!obj || obj->type != SGL_JSON_OBJECT) return NULL;
  for (int i = 0; i < obj->count; i++)
    if (strcmp(obj->keys[i], key) == 0) return obj->items[i];
  return NULL;
}

const char *sgl_json_get_str(const sgl_json *obj, const char *key, const char *def) {
  const sgl_json *j = sgl_json_get(obj, key);
  return (j && j->type == SGL_JSON_STRING) ? j->str : def;
}

double sgl_json_get_num(const sgl_json *obj, const char *key, double def) {
  const sgl_json *j = sgl_json_get(obj, key);
  return (j && j->type == SGL_JSON_NUMBER) ? j->num : def;
}

int sgl_json_get_int(const sgl_json *obj, const char *key, int def) {
  const sgl_json *j = sgl_json_get(obj, key);
  if (!j || j->type != SGL_JSON_NUMBER) return def;
  return (int)(j->num >= 0 ? j->num + 0.5 : j->num - 0.5); /* 四舍五入，含负数 */
}
