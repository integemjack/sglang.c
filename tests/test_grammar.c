/* test_grammar.c — 约束解码引擎（JSON Schema / 正则 → token 掩码）测试
 *
 * 用一张手工小词表（每个 token 一两个字节）驱动 token 级掩码与 accept / can_finish。
 */
#include <stdio.h>
#include <string.h>

#include "sgl/grammar.h"

static int fails = 0;
#define CHECK(cond, ...)                          \
  do {                                            \
    if (!(cond)) {                                \
      printf("FAIL %s:%d: ", __FILE__, __LINE__); \
      printf(__VA_ARGS__);                        \
      printf("\n");                               \
      fails++;                                    \
    }                                             \
  } while (0)

/* 手工词表：覆盖单字节结构字符、键名、多字节 token */
static const char *const VOCAB[] = {
    "{",       "}",     "\"a\"", "\"b\"", "\"c\"", "\"x\"", "\"y\"", ":",   "1",
    "2",       "0",     "11",    "10",    ",",     "null",  "[",     "]",   "true",
    " ",       "}{",    "\"a\":1", "\"ab\"", "a",   "b",     "5",     "9",   "//"};
#define NV ((int)(sizeof(VOCAB) / sizeof(VOCAB[0])))

static int vidx(const char *s) {
  for (int i = 0; i < NV; i++)
    if (strcmp(VOCAB[i], s) == 0) return i;
  return -1;
}

/* 单个 token 的掩码位 */
static int allow1(sgl_grammar *g, const char *tok) {
  int id = vidx(tok);
  unsigned char a = 0;
  if (id < 0) return -1;
  sgl_grammar_mask(g, &id, 1, &a);
  return a;
}

/* 用一个 token 推进；0=成功 */
static int acc(sgl_grammar *g, const char *tok) {
  int id = vidx(tok);
  if (id < 0) return -99;
  return sgl_grammar_accept(g, id);
}

/* 用原始字节推进；0=成功 */
static int ab(sgl_grammar *g, const char *s) {
  return sgl_grammar_accept_bytes(g, s, (int)strlen(s));
}

static const char *SCHEMA1 =
    "{\"type\":\"object\","
    "\"properties\":{\"a\":{\"type\":\"integer\"},"
    "\"b\":{\"type\":\"string\",\"enum\":[\"x\",\"y\"]}},"
    "\"required\":[\"a\"],\"additionalProperties\":false}";

static void test_object(void) {
  int lens[NV];
  int ids[NV];
  unsigned char allowed[NV];
  for (int i = 0; i < NV; i++) {
    lens[i] = (int)strlen(VOCAB[i]);
    ids[i] = i;
  }

  sgl_grammar *g = sgl_grammar_from_json_schema(SCHEMA1, strlen(SCHEMA1));
  CHECK(g != NULL, "schema1: %s", sgl_grammar_error(NULL));
  if (!g) return;
  sgl_grammar_set_vocab(g, VOCAB, lens, NV);

  /* 起始状态：'{' 允许、'}' 与键名禁止 */
  sgl_grammar_mask(g, ids, NV, allowed);
  CHECK(allowed[vidx("{")] == 1, "start: '{' must be allowed");
  CHECK(allowed[vidx("}")] == 0, "start: '}' must be forbidden");
  CHECK(allowed[vidx("\"a\"")] == 0, "start: key must be forbidden");
  CHECK(allowed[vidx("}{")] == 0, "start: '}{' must be forbidden");
  CHECK(allowed[vidx("\"a\":1")] == 0, "start: '\"a\":1' must be forbidden");
  CHECK(sgl_grammar_can_finish(g) == 0, "start: cannot finish");

  CHECK(acc(g, "{") == 0, "accept '{'");

  /* '{' 之后：只允许键名（a/b），不允许 '}'（required a 未出现） */
  sgl_grammar_mask(g, ids, NV, allowed);
  CHECK(allowed[vidx("\"a\"")] == 1, "after '{': key a allowed");
  CHECK(allowed[vidx("\"b\"")] == 1, "after '{': key b allowed");
  CHECK(allowed[vidx("\"c\"")] == 0, "after '{': unknown key forbidden");
  CHECK(allowed[vidx("}")] == 0, "after '{': '}' forbidden (required a)");
  /* 多字节 token：前面部分合法但整段超前的 token 必须被禁止 */
  CHECK(allowed[vidx("\"a\"")] == 1, "after '{': key a allowed");
  CHECK(allow1(g, "\"a\":1") == 1, "after '{': multi-byte token '\"a\":1' is a legal prefix");
  CHECK(allow1(g, "}{") == 0, "after '{': '}{' forbidden");

  /* 非法：'{' 后直接 '}' */
  CHECK(acc(g, "}") != 0, "'{' followed by '}' must be rejected");
  CHECK(sgl_grammar_can_finish(g) == 0, "after rejected '}': cannot finish");
  /* 失败的 accept 不能改变状态 */
  CHECK(acc(g, "\"a\"") == 0, "state survives rejected token: key a accepted");

  CHECK(acc(g, ":") == 0, "accept ':'");
  sgl_grammar_mask(g, ids, NV, allowed);
  CHECK(allowed[vidx("1")] == 1, "value of a: digit allowed");
  CHECK(allowed[vidx("11")] == 1, "value of a: '11' allowed");
  CHECK(allowed[vidx("0")] == 1, "value of a: '0' allowed");
  CHECK(allowed[vidx("null")] == 0, "value of a: 'null' forbidden (integer)");
  CHECK(allowed[vidx("\"x\"")] == 0, "value of a: string forbidden (integer)");
  CHECK(allowed[vidx("}")] == 0, "value of a: '}' forbidden before value");

  CHECK(acc(g, "1") == 0, "accept '1'");
  CHECK(sgl_grammar_can_finish(g) == 0, "{\"a\":1 cannot finish (object not closed)");
  sgl_grammar_mask(g, ids, NV, allowed);
  CHECK(allowed[vidx("}")] == 1, "after value: '}' allowed");
  CHECK(allowed[vidx(",")] == 1, "after value: ',' allowed");
  CHECK(allowed[vidx("\"b\"")] == 0, "after value: key needs ',' first");

  /* 完整合法输出：{"a":1} */
  CHECK(acc(g, "}") == 0, "accept '}'");
  CHECK(sgl_grammar_can_finish(g) == 1, "{\"a\":1} must be finishable");

  /* 复用：reset 后状态复位 */
  sgl_grammar_reset(g);
  CHECK(sgl_grammar_can_finish(g) == 0, "after reset: cannot finish");
  CHECK(allow1(g, "{") == 1, "after reset: '{' allowed");

  /* 两个键的完整路径：{"a":11,"b":"x"} */
  CHECK(acc(g, "{") == 0, "seq: '{'");
  CHECK(acc(g, "\"a\"") == 0, "seq: key a");
  CHECK(acc(g, ":") == 0, "seq: ':'");
  CHECK(acc(g, "11") == 0, "seq: 11");
  CHECK(acc(g, ",") == 0, "seq: ','");
  CHECK(acc(g, "\"b\"") == 0, "seq: key b");
  CHECK(acc(g, ":") == 0, "seq: ':'");
  CHECK(acc(g, "\"x\"") == 0, "seq: enum x");
  CHECK(sgl_grammar_can_finish(g) == 0, "{\"a\":11,\"b\":\"x\" cannot finish yet");
  CHECK(acc(g, "}") == 0, "seq: '}'");
  CHECK(sgl_grammar_can_finish(g) == 1, "{\"a\":11,\"b\":\"x\"} must be finishable");

  /* b 的 enum 收紧 */
  sgl_grammar_reset(g);
  CHECK(acc(g, "{") == 0, "enum: '{'");
  CHECK(acc(g, "\"a\"") == 0, "enum: key a");
  CHECK(acc(g, ":") == 0, "enum: ':'");
  CHECK(acc(g, "1") == 0, "enum: 1");
  CHECK(acc(g, ",") == 0, "enum: ','");
  CHECK(acc(g, "\"b\"") == 0, "enum: key b");
  CHECK(acc(g, ":") == 0, "enum: ':'");
  CHECK(allow1(g, "\"x\"") == 1, "enum: \"x\" allowed");
  CHECK(allow1(g, "\"y\"") == 1, "enum: \"y\" allowed");
  CHECK(allow1(g, "1") == 0, "enum: digit forbidden");
  CHECK(acc(g, "\"y\"") == 0, "enum: accept \"y\"");
  CHECK(acc(g, "}") == 0, "enum: '}'");
  CHECK(sgl_grammar_can_finish(g) == 1, "enum: finishable");

  /* required 缺失：{"b":"x"} 不能收尾 */
  sgl_grammar_reset(g);
  CHECK(acc(g, "{") == 0, "req: '{'");
  CHECK(acc(g, "\"b\"") == 0, "req: key b");
  CHECK(acc(g, ":") == 0, "req: ':'");
  CHECK(acc(g, "\"x\"") == 0, "req: \"x\"");
  CHECK(acc(g, "}") != 0, "required key a missing: '}' must be rejected");

  /* 未知键被禁止 */
  sgl_grammar_reset(g);
  CHECK(acc(g, "{") == 0, "unknown: '{'");
  CHECK(acc(g, "\"c\"") != 0, "unknown key c must be rejected");

  /* 值类型错误：a 是 integer，给 null 必须失败 */
  sgl_grammar_reset(g);
  CHECK(acc(g, "{") == 0, "type: '{'");
  CHECK(acc(g, "\"a\"") == 0, "type: key a");
  CHECK(acc(g, ":") == 0, "type: ':'");
  CHECK(acc(g, "null") != 0, "integer property must reject null");

  /* 词表外的 id / 负 id 一律禁止 */
  {
    int weird[3] = {-1, NV + 5, 9999};
    unsigned char wa[3] = {1, 1, 1};
    sgl_grammar_mask(g, weird, 3, wa);
    CHECK(wa[0] == 0 && wa[1] == 0 && wa[2] == 0, "out-of-vocab ids must be masked out");
    CHECK(sgl_grammar_accept(g, -1) != 0, "accept(-1) must fail");
  }

  sgl_grammar_free(g);
}

static void test_regex(void) {
  /* 用例 2："[0-9]{2,4}" */
  sgl_grammar *g = sgl_grammar_from_regex("[0-9]{2,4}");
  CHECK(g != NULL, "regex: [0-9]{2,4}: %s", sgl_grammar_error(NULL));
  if (g) {
    CHECK(sgl_grammar_can_finish(g) == 0, "regex: empty input cannot finish");
    CHECK(ab(g, "1") == 0, "regex: accept '1'");
    CHECK(sgl_grammar_can_finish(g) == 0, "regex: 1 digit cannot finish");
    CHECK(ab(g, "2") == 0, "regex: accept '2'");
    CHECK(sgl_grammar_can_finish(g) == 1, "regex: 2 digits can finish");
    CHECK(ab(g, "3") == 0, "regex: accept '3'");
    CHECK(ab(g, "4") == 0, "regex: accept '4'");
    CHECK(sgl_grammar_can_finish(g) == 1, "regex: 4 digits can finish");
    CHECK(ab(g, "5") != 0, "regex: 5 digits must be rejected");
    CHECK(sgl_grammar_can_finish(g) == 1, "regex: still finishable after rejected byte");
    sgl_grammar_free(g);
  }

  g = sgl_grammar_from_regex("[0-9]{2,4}");
  if (g) {
    CHECK(sgl_grammar_can_finish(g) == 0, "regex2: 0 digits cannot finish");
    CHECK(ab(g, "1") == 0, "regex2: accept '1'");
    CHECK(sgl_grammar_can_finish(g) == 0, "regex2: 1 digit cannot finish");
    CHECK(ab(g, "a") != 0, "regex2: non-digit must be rejected");
    sgl_grammar_free(g);
  }

  /* 分支 / 分组 / 量词 / 字符类 */
  g = sgl_grammar_from_regex("(ab|cd)+e?");
  if (g) {
    CHECK(ab(g, "ab") == 0, "regex3: 'ab'");
    CHECK(sgl_grammar_can_finish(g) == 1, "regex3: 'ab' can finish");
    CHECK(ab(g, "cd") == 0, "regex3: 'cd'");
    CHECK(sgl_grammar_can_finish(g) == 1, "regex3: 'abcd' can finish");
    CHECK(ab(g, "e") == 0, "regex3: optional 'e'");
    CHECK(sgl_grammar_can_finish(g) == 1, "regex3: 'abcde' can finish");
    CHECK(ab(g, "x") != 0, "regex3: 'x' must be rejected");
    sgl_grammar_free(g);
  }

  /* 字符类 + 转义 + 区间 */
  g = sgl_grammar_from_regex("[A-Z][a-z]?\\d+");
  if (g) {
    CHECK(ab(g, "Ab12") == 0, "regex4: 'Ab12'");
    CHECK(sgl_grammar_can_finish(g) == 1, "regex4: can finish");
    sgl_grammar_free(g);
  }
  g = sgl_grammar_from_regex("[A-Z][a-z]?\\d+");
  if (g) {
    CHECK(ab(g, "a") != 0, "regex4: lowercase start must be rejected");
    sgl_grammar_free(g);
  }
  g = sgl_grammar_from_regex("[A-Z][a-z]?\\d+");
  if (g) {
    CHECK(ab(g, "Zx") == 0, "regex4: 'Zx' is a legal prefix");
    CHECK(sgl_grammar_can_finish(g) == 0, "regex4: digits are mandatory");
    sgl_grammar_free(g);
  }

  CHECK(sgl_grammar_from_regex("[abc") == NULL, "regex: unterminated class -> NULL");
  CHECK(sgl_grammar_from_regex("(ab") == NULL, "regex: unbalanced group -> NULL");
}

static void test_array_and_anyof(void) {
  /* 用例 3：数组 + items + minItems/maxItems */
  const char *sch =
      "{\"type\":\"array\",\"items\":{\"type\":\"integer\"},\"minItems\":2,\"maxItems\":3}";
  sgl_grammar *g = sgl_grammar_from_json_schema(sch, strlen(sch));
  CHECK(g != NULL, "array schema: %s", sgl_grammar_error(NULL));
  if (g) {
    CHECK(ab(g, "[") == 0, "array: '['");
    CHECK(ab(g, "]") != 0, "array: minItems 2 not reached, ']' rejected");
    sgl_grammar_free(g);
  }
  g = sgl_grammar_from_json_schema(sch, strlen(sch));
  if (g) {
    CHECK(ab(g, "[") == 0, "array2: '['");
    CHECK(ab(g, "1") == 0, "array2: '1'");
    CHECK(ab(g, ",") == 0, "array2: ','");
    CHECK(ab(g, "2") == 0, "array2: '2'");
    CHECK(sgl_grammar_can_finish(g) == 0, "array2: [1,2 unfinished");
    CHECK(ab(g, "]") == 0, "array2: ']'");
    CHECK(sgl_grammar_can_finish(g) == 1, "array2: [1,2] can finish");
    sgl_grammar_free(g);
  }
  g = sgl_grammar_from_json_schema(sch, strlen(sch));
  if (g) {
    CHECK(ab(g, "[1,2,3") == 0, "array3: [1,2,3");
    CHECK(ab(g, ",") != 0, "array3: maxItems 3 reached, ',' rejected");
    CHECK(ab(g, "]") == 0, "array3: ']' still ok");
    CHECK(sgl_grammar_can_finish(g) == 1, "array3: [1,2,3] can finish");
    sgl_grammar_free(g);
  }
  g = sgl_grammar_from_json_schema(sch, strlen(sch));
  if (g) {
    CHECK(ab(g, "[1,\"x\"") != 0, "array4: string item must be rejected");
    sgl_grammar_free(g);
  }

  /* 用例 3b：anyOf / oneOf */
  const char *sch2 = "{\"anyOf\":[{\"type\":\"string\"},{\"type\":\"integer\"}]}";
  g = sgl_grammar_from_json_schema(sch2, strlen(sch2));
  CHECK(g != NULL, "anyOf schema: %s", sgl_grammar_error(NULL));
  if (g) {
    CHECK(ab(g, "\"x\"") == 0, "anyOf: string branch");
    CHECK(sgl_grammar_can_finish(g) == 1, "anyOf: string can finish");
    sgl_grammar_reset(g);
    CHECK(ab(g, "42") == 0, "anyOf: integer branch");
    CHECK(sgl_grammar_can_finish(g) == 1, "anyOf: integer can finish");
    sgl_grammar_reset(g);
    CHECK(ab(g, "true") != 0, "anyOf: boolean must be rejected");
    sgl_grammar_reset(g);
    CHECK(ab(g, "\"ab") == 0, "anyOf: unterminated string is a legal prefix");
    CHECK(sgl_grammar_can_finish(g) == 0, "anyOf: unterminated string cannot finish");
    sgl_grammar_free(g);
  }
  const char *sch3 = "{\"oneOf\":[{\"const\":\"red\"},{\"const\":\"green\"}]}";
  g = sgl_grammar_from_json_schema(sch3, strlen(sch3));
  if (g) {
    CHECK(ab(g, "\"red\"") == 0, "oneOf/const: \"red\"");
    CHECK(sgl_grammar_can_finish(g) == 1, "oneOf/const: can finish");
    CHECK(ab(g, "x") != 0, "oneOf/const: trailing byte rejected");
    sgl_grammar_reset(g);
    CHECK(ab(g, "\"blue\"") != 0, "oneOf/const: unknown literal rejected");
    sgl_grammar_free(g);
  }
}

static void test_string_and_number(void) {
  /* 字符串长度（按字符计，转义算 1 个） */
  const char *sch = "{\"type\":\"string\",\"minLength\":2,\"maxLength\":3}";
  sgl_grammar *g = sgl_grammar_from_json_schema(sch, strlen(sch));
  CHECK(g != NULL, "string schema: %s", sgl_grammar_error(NULL));
  if (g) {
    CHECK(ab(g, "\"a") == 0, "str: 1 char is a legal prefix");
    CHECK(sgl_grammar_can_finish(g) == 0, "str: 1 char < minLength");
    CHECK(ab(g, "\"") != 0, "str: cannot close before minLength");
    CHECK(ab(g, "b") == 0, "str: 2 chars");
    CHECK(ab(g, "\"") == 0, "str: closing quote");
    CHECK(sgl_grammar_can_finish(g) == 1, "str: 2 chars can finish");
    sgl_grammar_reset(g);
    CHECK(ab(g, "\"abc\"") == 0, "str: 3 chars");
    CHECK(sgl_grammar_can_finish(g) == 1, "str: 3 chars can finish");
    sgl_grammar_reset(g);
    CHECK(ab(g, "\"abcd\"") != 0, "str: 4 chars > maxLength must be rejected");
    sgl_grammar_reset(g);
    CHECK(ab(g, "\"\\n") == 0, "str: escape counts as one char");
    CHECK(sgl_grammar_can_finish(g) == 0, "str: escape alone < minLength");
    CHECK(ab(g, "z") == 0, "str: second char");
    CHECK(sgl_grammar_can_finish(g) == 0, "str: unterminated cannot finish");
    CHECK(ab(g, "\"") == 0, "str: closing quote after escape");
    CHECK(sgl_grammar_can_finish(g) == 1, "str: escape+char can finish");
    sgl_grammar_free(g);
  }

  /* pattern 用同一套正则匹配器 */
  const char *schp = "{\"type\":\"string\",\"pattern\":\"[0-9]{2,4}\"}";
  g = sgl_grammar_from_json_schema(schp, strlen(schp));
  if (g) {
    CHECK(ab(g, "\"12\"") == 0, "pattern: \"12\"");
    CHECK(sgl_grammar_can_finish(g) == 1, "pattern: \"12\" can finish");
    sgl_grammar_reset(g);
    CHECK(ab(g, "\"1\"") != 0, "pattern: \"1\" too short");
    sgl_grammar_reset(g);
    CHECK(ab(g, "\"12345\"") != 0, "pattern: 5 digits too long");
    sgl_grammar_free(g);
  }

  /* 数字上下界 */
  const char *schn = "{\"type\":\"integer\",\"minimum\":10,\"maximum\":99}";
  g = sgl_grammar_from_json_schema(schn, strlen(schn));
  CHECK(g != NULL, "number schema: %s", sgl_grammar_error(NULL));
  if (g) {
    CHECK(ab(g, "5") == 0, "num: '5' is a legal prefix of 50..59");
    CHECK(sgl_grammar_can_finish(g) == 0, "num: 5 < minimum, cannot finish");
    CHECK(ab(g, "0") == 0, "num: 50");
    CHECK(sgl_grammar_can_finish(g) == 1, "num: 50 in range");
    CHECK(ab(g, "5") != 0, "num: 505 > maximum rejected");
    sgl_grammar_reset(g);
    CHECK(ab(g, "100") != 0, "num: 100 > maximum rejected");
    sgl_grammar_reset(g);
    CHECK(ab(g, "05") != 0, "num: leading zero rejected");
    sgl_grammar_reset(g);
    CHECK(ab(g, "-1") != 0, "num: negative rejected (minimum 10)");
    sgl_grammar_free(g);
  }

  /* 小数：结构性检查 + 整数部分可行性 */
  const char *schf = "{\"type\":\"number\",\"minimum\":0.5,\"maximum\":2.5}";
  g = sgl_grammar_from_json_schema(schf, strlen(schf));
  if (g) {
    CHECK(ab(g, "0.7") == 0, "num2: 0.7");
    CHECK(sgl_grammar_can_finish(g) == 1, "num2: 0.7 can finish");
    sgl_grammar_reset(g);
    CHECK(ab(g, "2.4") == 0, "num2: 2.4");
    CHECK(sgl_grammar_can_finish(g) == 1, "num2: 2.4 can finish");
    sgl_grammar_reset(g);
    CHECK(ab(g, "9") != 0, "num2: 9 out of range rejected");
    sgl_grammar_reset(g);
    CHECK(ab(g, "0.") == 0, "num2: '0.' needs a fraction digit");
    CHECK(sgl_grammar_can_finish(g) == 0, "num2: '0.' cannot finish");
    CHECK(ab(g, "]") != 0, "num2: junk after '0.' rejected");
    sgl_grammar_free(g);
  }
}

static void test_misc(void) {
  /* additionalProperties 缺省 → 退化：任意键 + 任意 JSON 值 */
  const char *sch = "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"string\"}}}";
  sgl_grammar *g = sgl_grammar_from_json_schema(sch, strlen(sch));
  CHECK(g != NULL, "loose object: %s", sgl_grammar_error(NULL));
  if (g) {
    CHECK(ab(g, "{}") == 0, "loose: '{}' accepted (required not enforced)");
    CHECK(sgl_grammar_can_finish(g) == 1, "loose: '{}' can finish");
    sgl_grammar_reset(g);
    CHECK(ab(g, "{\"z\":[1,true,null]}") == 0, "loose: unknown key with arbitrary value");
    CHECK(sgl_grammar_can_finish(g) == 1, "loose: can finish");
    sgl_grammar_free(g);
  }

  /* type 数组 / null / boolean */
  const char *sch2 = "{\"type\":[\"string\",\"null\"]}";
  g = sgl_grammar_from_json_schema(sch2, strlen(sch2));
  if (g) {
    CHECK(ab(g, "null") == 0, "type array: null branch");
    CHECK(sgl_grammar_can_finish(g) == 1, "type array: null can finish");
    sgl_grammar_reset(g);
    CHECK(ab(g, "\"s\"") == 0, "type array: string branch");
    CHECK(sgl_grammar_can_finish(g) == 1, "type array: string can finish");
    sgl_grammar_reset(g);
    CHECK(ab(g, "1") != 0, "type array: number rejected");
    sgl_grammar_free(g);
  }

  /* 空 schema / true → 任意 JSON */
  g = sgl_grammar_from_json_schema("true", 4);
  CHECK(g != NULL, "true schema");
  if (g) {
    CHECK(ab(g, "{\"k\":[1,2.5e3,\"s\"]}") == 0, "any: nested json");
    CHECK(sgl_grammar_can_finish(g) == 1, "any: can finish");
    sgl_grammar_free(g);
  }

  /* 不支持 / 非法 schema → NULL */
  const char *s_allof = "{\"allOf\":[{\"type\":\"string\"}]}";
  CHECK(sgl_grammar_from_json_schema(s_allof, strlen(s_allof)) == NULL, "allOf must return NULL");
  CHECK(strstr(sgl_grammar_error(NULL), "allOf") != NULL, "error text mentions allOf: %s",
        sgl_grammar_error(NULL));
  const char *s_bad = "{oops";
  CHECK(sgl_grammar_from_json_schema(s_bad, strlen(s_bad)) == NULL, "invalid json -> NULL");
  const char *s_req =
      "{\"type\":\"object\",\"properties\":{\"a\":{}},\"required\":[\"zz\"],"
      "\"additionalProperties\":false}";
  CHECK(sgl_grammar_from_json_schema(s_req, strlen(s_req)) == NULL,
        "required key missing from properties -> NULL");
  const char *s_min = "{\"type\":\"string\",\"minimum\":1}";
  g = sgl_grammar_from_json_schema(s_min, strlen(s_min));
  CHECK(g != NULL, "minimum on a string is ignored, grammar still builds");
  sgl_grammar_free(g);
}

/* ---------------- 表驱动用例：每个 schema 验「合法串全接受」+「非法串在最深字节被拒」 ---------------- */
typedef struct {
  const char *schema;
  const char *valid[5]; /* NULL 结尾 */
  const char *bad;      /* 非法串 */
  int bad_at;           /* 第几个字节开始非法（0 起）：前面必须全接受，这一个必须被拒 */
  const char *resume;   /* 拒掉之后仍应能接上的合法尾巴（NULL=不检查；""=只验可结束） */
} tcase;

static void run_schema_cases(const tcase *cs, int n) {
  for (int i = 0; i < n; i++) {
    const tcase *c = &cs[i];
    for (int k = 0; c->valid[k]; k++) {
      sgl_grammar *g = sgl_grammar_from_json_schema(c->schema, strlen(c->schema));
      CHECK(g != NULL, "case %d: schema %s: %s", i, c->schema, sgl_grammar_error(NULL));
      if (!g) continue;
      CHECK(ab(g, c->valid[k]) == 0, "case %d: valid '%s' rejected (schema %s)", i, c->valid[k],
            c->schema);
      CHECK(sgl_grammar_can_finish(g) == 1, "case %d: '%s' should be finishable", i, c->valid[k]);
      sgl_grammar_free(g);
    }
    sgl_grammar *g = sgl_grammar_from_json_schema(c->schema, strlen(c->schema));
    if (!g) {
      CHECK(0, "case %d: schema %s: %s", i, c->schema, sgl_grammar_error(NULL));
      continue;
    }
    int bl = (int)strlen(c->bad);
    if (c->bad_at < 0 || c->bad_at >= bl) {
      CHECK(0, "case %d: bad_at %d out of range for '%s'", i, c->bad_at, c->bad);
      sgl_grammar_free(g);
      continue;
    }
    char pre[128];
    memcpy(pre, c->bad, (size_t)c->bad_at);
    pre[c->bad_at] = 0;
    CHECK(ab(g, pre) == 0, "case %d: prefix '%s' of '%s' must be accepted", i, pre, c->bad);
    int fin = sgl_grammar_can_finish(g);
    CHECK(sgl_grammar_accept_bytes(g, c->bad + c->bad_at, 1) != 0,
          "case %d: byte '%c' (pos %d) of '%s' must be rejected", i, (int)c->bad[c->bad_at],
          c->bad_at, c->bad);
    CHECK(sgl_grammar_can_finish(g) == fin, "case %d: state must not change after rejection", i);
    if (c->resume) {
      CHECK(ab(g, c->resume) == 0, "case %d: resume '%s' must be accepted after rejection", i,
            c->resume);
      CHECK(sgl_grammar_can_finish(g) == 1, "case %d: '%s'+'%s' should be finishable", i, pre,
            c->resume);
    }
    sgl_grammar_free(g);
  }
}

static void test_schema_matrix(void) {
  static const tcase cs[] = {
      /* object + required + additionalProperties:false + integer */
      {"{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"integer\"}},"
       "\"required\":[\"a\"],\"additionalProperties\":false}",
       {"{\"a\":1}", "{\"a\":-22}", NULL}, "{\"b\":1}", 2, "a\":1}"},
      /* array + items + minItems/maxItems */
      {"{\"type\":\"array\",\"items\":{\"type\":\"integer\"},\"minItems\":2,"
       "\"maxItems\":3}",
       {"[1,2]", "[1,2,3]", NULL}, "[1]", 2, ",2]"},
      /* string minLength/maxLength */
      {"{\"type\":\"string\",\"minLength\":2,\"maxLength\":3}",
       {"\"ab\"", "\"abc\"", NULL}, "\"abcd\"", 4, "\""},
      /* string pattern */
      {"{\"type\":\"string\",\"pattern\":\"[a-z]{2,3}\"}",
       {"\"ab\"", "\"abc\"", NULL}, "\"abcd\"", 4, "\""},
      /* number + minimum/maximum */
      {"{\"type\":\"number\",\"minimum\":0,\"maximum\":10}",
       {"0", "10", "9.5", "1e1", NULL}, "11", 1, ""},
      /* integer（不得带小数点） */
      {"{\"type\":\"integer\"}", {"0", "-1", "42", NULL}, "1.5", 1, ""},
      /* boolean */
      {"{\"type\":\"boolean\"}", {"true", "false", NULL}, "tru1", 3, "e"},
      /* null */
      {"{\"type\":\"null\"}", {"null", NULL}, "nulx", 3, "l"},
      /* enum（同首字节的分支：靠 ALT 回溯区分） */
      {"{\"enum\":[\"red\",\"green\"]}", {"\"red\"", "\"green\"", NULL}, "\"blue\"", 1,
       "red\""},
      /* oneOf */
      {"{\"oneOf\":[{\"type\":\"string\"},{\"type\":\"array\",\"items\":{\"type\":"
       "\"boolean\"}}]}",
       {"\"s\"", "[true,false]", NULL}, "[tru]", 4, "e]"},
      /* anyOf + type 数组 */
      {"{\"anyOf\":[{\"type\":\"string\"},{\"type\":\"integer\"}]}",
       {"\"v\"", "7", NULL}, "t", 0, NULL},
      {"{\"type\":[\"boolean\",\"null\"]}", {"null", "false", NULL}, "1", 0, NULL},
      /* 嵌套：对象套数组套对象 */
      {"{\"type\":\"object\",\"properties\":{\"xs\":{\"type\":\"array\",\"items\":{\"type\":"
       "\"object\",\"properties\":{\"k\":{\"type\":\"boolean\"}},\"required\":[\"k\"],"
       "\"additionalProperties\":false}}},\"required\":[\"xs\"],\"additionalProperties\":false}",
       {"{\"xs\":[{\"k\":true}]}", "{\"xs\":[]}", NULL},
       "{\"xs\":[{\"k\":1}]}", 12, "true}]}"},
  };
  run_schema_cases(cs, (int)(sizeof(cs) / sizeof(cs[0])));
}

/* ---------------- 表驱动用例：EBNF / 正则 ---------------- */
typedef struct {
  const char *pat;
  const char *valid[5];
  const char *bad;
  int bad_at;
  const char *resume;
} rcase;

static void run_regex_cases(const rcase *cs, int n) {
  for (int i = 0; i < n; i++) {
    const rcase *c = &cs[i];
    for (int k = 0; c->valid[k]; k++) {
      sgl_grammar *g = sgl_grammar_from_regex(c->pat);
      CHECK(g != NULL, "rcase %d: regex %s: %s", i, c->pat, sgl_grammar_error(NULL));
      if (!g) continue;
      CHECK(ab(g, c->valid[k]) == 0, "rcase %d: valid '%s' rejected by %s", i, c->valid[k], c->pat);
      CHECK(sgl_grammar_can_finish(g) == 1, "rcase %d: '%s' should be finishable", i,
            c->valid[k]);
      sgl_grammar_free(g);
    }
    sgl_grammar *g = sgl_grammar_from_regex(c->pat);
    if (!g) {
      CHECK(0, "rcase %d: regex %s: %s", i, c->pat, sgl_grammar_error(NULL));
      continue;
    }
    char pre[128];
    memcpy(pre, c->bad, (size_t)c->bad_at);
    pre[c->bad_at] = 0;
    CHECK(ab(g, pre) == 0, "rcase %d: prefix '%s' of '%s' must be accepted", i, pre, c->bad);
    int fin = sgl_grammar_can_finish(g);
    CHECK(sgl_grammar_accept_bytes(g, c->bad + c->bad_at, 1) != 0,
          "rcase %d: byte '%c' (pos %d) of '%s' must be rejected", i, (int)c->bad[c->bad_at],
          c->bad_at, c->bad);
    CHECK(sgl_grammar_can_finish(g) == fin, "rcase %d: state must not change after rejection", i);
    if (c->resume) {
      CHECK(ab(g, c->resume) == 0, "rcase %d: resume '%s' must be accepted", i, c->resume);
      CHECK(sgl_grammar_can_finish(g) == 1, "rcase %d: after resume must be finishable", i);
    }
    sgl_grammar_free(g);
  }
}

static void test_regex_matrix(void) {
  static const rcase cs[] = {
      /* 字符类 + '+' */
      {"[a-z]+", {"a", "abc", NULL}, "ab1", 2, ""},
      /* 字面量 + 分组 + 分支 + '*' */
      {"a(b|c)*d", {"ad", "abd", "abcd", NULL}, "abce", 3, "d"},
      /* 分组交替 + '+'（嵌套） */
      {"(ab|cd)+", {"ab", "abcd", NULL}, "abx", 2, ""},
      /* 字符类 {m} + 可选分组 */
      {"x[0-9]{2}(y|z)?", {"x12", "x12y", "x12z", NULL}, "x1a", 2, "2"},
      /* [A-Z] [a-z]* 字面量 */
      {"[A-Z][a-z]*!", {"A!", "Abc!", NULL}, "Abc?", 3, "!"},
      /* 嵌套分组 */
      {"((a|b)(c|d))+e", {"ace", "acbdace", NULL}, "adx", 2, "e"},
      /* 锚点忽略（整串匹配） */
      {"^[0-9]+$", {"1", "123", NULL}, "12a", 2, ""},
      /* 转义与字符类简写 */
      {"\\w-\\d+", {"a-1", "Z-9", NULL}, "-1", 0, NULL},
      /* '.' 加上界固定重复 {m} */
      {".{3}", {"abc", "a b", NULL}, "abcd", 3, ""},
      /* 上界开放的 {m,} */
      {"a{2,}", {"aa", "aaaa", NULL}, "aab", 2, ""},
  };
  run_regex_cases(cs, (int)(sizeof(cs) / sizeof(cs[0])));
}

/* ---------------- 逐字节喂入：增量语义 ---------------- */
static void test_incremental(void) {
  const char *sch =
      "{\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\","
      "\"minLength\":1},\"tags\":{\"type\":\"array\",\"items\":{\"type\":\"string\"},"
      "\"maxItems\":2},\"ok\":{\"type\":\"boolean\"}},"
      "\"required\":[\"name\",\"ok\"],\"additionalProperties\":false}";
  const char *doc = "{\"name\":\"x\",\"tags\":[\"a\",\"b\"],\"ok\":true}";
  int dlen = (int)strlen(doc);

  sgl_grammar *g = sgl_grammar_from_json_schema(sch, strlen(sch));
  CHECK(g != NULL, "incremental schema: %s", sgl_grammar_error(NULL));
  if (!g) return;

  /* 1) 每次只喂一个字节，最后一步才变成可结束 */
  for (int i = 0; i < dlen; i++) {
    CHECK(sgl_grammar_can_finish(g) == 0, "incremental: finishable too early at byte %d", i);
    char c = doc[i];
    CHECK(sgl_grammar_accept_bytes(g, &c, 1) == 0, "incremental: byte %d ('%c') rejected", i, c);
  }
  CHECK(sgl_grammar_can_finish(g) == 1, "incremental: full doc must be finishable");
  /* 结束后再多一个字节必须被拒 */
  CHECK(sgl_grammar_accept_bytes(g, " ", 1) != 0, "incremental: nothing may follow a closed doc");

  /* 2) 中途插一个非法字节：状态不变，剩下的字节仍能接上 */
  int cut = dlen / 2;
  sgl_grammar_reset(g);
  CHECK(sgl_grammar_can_finish(g) == 0, "incremental: reset cannot finish");
  for (int i = 0; i < cut; i++) {
    char c = doc[i];
    CHECK(sgl_grammar_accept_bytes(g, &c, 1) == 0, "incremental2: byte %d rejected", i);
  }
  int fin = sgl_grammar_can_finish(g);
  CHECK(sgl_grammar_accept_bytes(g, "\001", 1) != 0, "incremental2: control byte must be rejected");
  CHECK(sgl_grammar_can_finish(g) == fin, "incremental2: can_finish must survive rejection");
  for (int i = cut; i < dlen; i++) {
    char c = doc[i];
    CHECK(sgl_grammar_accept_bytes(g, &c, 1) == 0, "incremental2: byte %d rejected after recovery",
          i);
  }
  CHECK(sgl_grammar_can_finish(g) == 1, "incremental2: must finish after recovery");

  sgl_grammar_free(g);
}

int main(void) {
  test_object();
  test_regex();
  test_array_and_anyof();
  test_string_and_number();
  test_misc();
  test_schema_matrix();
  test_regex_matrix();
  test_incremental();

  if (fails) {
    printf("test_grammar: %d FAILURES\n", fails);
    return 1;
  }
  printf("test_grammar: all OK\n");
  return 0;
}