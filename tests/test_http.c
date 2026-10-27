/* test_http.c — HTTP 服务端到端测试（进程内起服务 + winsock 客户端）
 * 验证：
 *   1. GET /health
 *   2. POST /generate（greedy，与已验证的参考续写精确比对）
 *   3. POST /v1/completions（OpenAI 格式）
 *   4. 并发两个 /generate（两线程同时发）→ 各自结果仍与串行参考全等
 *      （这正是连续批处理在服务层生效的证明）*/
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <pthread.h>
#include <winsock2.h>
#include <ws2tcpip.h>

#include "sgl/alloc.h"
#include "sgl/http.h"
#include "sgl/json.h"
#include "sgl/scheduler.h"

static int FAILURES = 0;
#define CHECK(cond, ...) do { if (!(cond)) { FAILURES++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static const char *PORT_STR = "18081";

/* 发送原始 HTTP 请求，返回响应 body（malloc），*out_status 写状态码 */
static char *http_raw(const char *req, size_t req_len, int *out_status) {
  WSADATA wd;
  static int wsa = 0;
  if (!wsa) { WSAStartup(MAKEWORD(2, 2), &wd); wsa = 1; }
  SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  if (s == INVALID_SOCKET) return NULL;
  struct sockaddr_in a;
  memset(&a, 0, sizeof(a));
  a.sin_family = AF_INET;
  a.sin_port = htons((unsigned short)atoi(PORT_STR));
  inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
  if (connect(s, (struct sockaddr *)&a, sizeof(a)) != 0) { closesocket(s); return NULL; }
  send(s, req, (int)req_len, 0);
  size_t cap = 65536, len = 0;
  char *buf = (char *)sgl_xmalloc(cap);
  for (;;) {
    if (len + 1 >= cap) { cap *= 2; buf = (char *)sgl_xrealloc(buf, cap); }
    int k = recv(s, buf + len, (int)(cap - len - 1), 0);
    if (k <= 0) break;
    len += (size_t)k;
  }
  closesocket(s);
  buf[len] = 0;
  if (out_status) *out_status = atoi(buf + 9); /* "HTTP/1.1 XXX" */
  char *body = strstr(buf, "\r\n\r\n");
  char *ret = NULL;
  if (body) {
    body += 4;
    size_t n = strlen(body) + 1;
    ret = (char *)sgl_xmalloc(n); /* 与 sgl_xfree 配套（_strdup 是 CRT malloc）*/
    memcpy(ret, body, n);
  }
  sgl_xfree(buf);
  return ret;
}

static char *build_post(const char *path, const char *json_body, size_t *out_len) {
  size_t bl = strlen(json_body);
  char *req = (char *)sgl_xmalloc(bl + 512);
  int n = snprintf(req, bl + 512,
                   "POST %s HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n%s",
                   path, bl, json_body);
  *out_len = (size_t)n;
  return req;
}

/* 已验证的 greedy 参考续写（test_scheduler 与 numpy 参考交叉验证过）*/
static const char *REF_A = " not the same as the meaning of death"; /* "The meaning of life is" 前 8 token */
static const char *REF_B = " a restaurant called The Bistro. I'm";   /* "My name is John and I work at" 10 token */

typedef struct { const char *prompt; int max_new; const char *want; int ok; } gen_job;

static void *gen_thread(void *arg) {
  gen_job *j = (gen_job *)arg;
  char body[512];
  snprintf(body, sizeof(body),
           "{\"text\":\"%s\",\"sampling_params\":{\"max_new_tokens\":%d,\"temperature\":0}}",
           j->prompt, j->max_new);
  size_t rl;
  char *req = build_post("/generate", body, &rl);
  int status = 0;
  char *resp = http_raw(req, rl, &status);
  sgl_xfree(req);
  j->ok = 0;
  if (resp && status == 200) {
    sgl_json *r = sgl_json_parse(resp, strlen(resp));
    if (r) {
      const char *c = sgl_json_get_str(r, "completion", NULL);
      if (c) j->ok = strcmp(c, j->want) == 0;
      sgl_json_free(r);
    }
  }
  sgl_xfree(resp);
  return NULL;
}

int main(int argc, char **argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  const char *dir = argc > 1 ? argv[1] : getenv("SGL_TEST_MODEL");
  if (!dir || !*dir) dir = "models/gpt2";
  char ptmp[1024];
  snprintf(ptmp, sizeof(ptmp), "%s/config.json", dir);
  FILE *f = fopen(ptmp, "rb");
  if (!f) { printf("test_http: SKIP（没找到 %s）\n", dir); return 0; }
  fclose(f);

  sgl_scheduler *sched = sgl_sched_new(dir, 7);
  if (!sched) { printf("test_http: sched init failed\n"); return 1; }
  sgl_http_server *srv = sgl_http_serve(sched, "127.0.0.1", atoi(PORT_STR));
  if (!srv) { printf("test_http: serve failed\n"); return 1; }

  /* 1. /health */
  int status = 0;
  char *resp = http_raw("GET /health HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n",
                        52, &status);
  CHECK(resp && status == 200 && strstr(resp, "\"ok\":true"), "health: %d %s",
        status, resp ? resp : "(null)");
  sgl_xfree(resp);

  /* 2. /generate greedy */
  gen_job a = { "The meaning of life is", 8, REF_A, 0 };
  gen_thread(&a);
  CHECK(a.ok, "/generate A 与参考续写不一致");

  /* 3. /v1/completions（OpenAI）*/
  {
    size_t rl;
    char *req = build_post("/v1/completions",
                           "{\"model\":\"gpt2\",\"prompt\":\"The meaning of life is\",\"max_tokens\":8,\"temperature\":0}",
                           &rl);
    resp = http_raw(req, rl, &status);
    sgl_xfree(req);
    CHECK(resp && status == 200, "v1/completions status=%d", status);
    if (resp) {
      sgl_json *r = sgl_json_parse(resp, strlen(resp));
      CHECK(r, "v1/completions JSON parse");
      if (r) {
        const sgl_json *ch = sgl_json_get(r, "choices");
        const char *txt = ch && ch->count ? sgl_json_get_str(ch->items[0], "text", NULL) : NULL;
        CHECK(txt && strcmp(txt, REF_A) == 0, "v1 text=\"%s\" want=\"%s\"", txt ? txt : "", REF_A);
        int ct = sgl_json_get_int(sgl_json_get(r, "usage"), "completion_tokens", -1);
        CHECK(ct == 8, "usage.completion_tokens=%d", ct);
        sgl_json_free(r);
      }
      sgl_xfree(resp);
    }
  }

  /* 4. 并发两请求（同时发 → 连续批处理，各自结果不变）*/
  gen_job b = { "My name is John and I work at", 10, REF_B, 0 };
  gen_job a2 = { "The meaning of life is", 8, REF_A, 0 };
  pthread_t t1, t2;
  pthread_create(&t1, NULL, gen_thread, &a2);
  pthread_create(&t2, NULL, gen_thread, &b);
  pthread_join(t1, NULL);
  pthread_join(t2, NULL);
  CHECK(a2.ok, "并发请求 A 与参考不一致（批处理不应改变结果）");
  CHECK(b.ok, "并发请求 B 与参考不一致（批处理不应改变结果）");

  /* 5. 流式 chat SSE：delta 拼接 == 非流式同参结果（greedy）*/
  {
    size_t rl;
    char *req = build_post("/v1/chat/completions",
                           "{\"messages\":[{\"role\":\"user\",\"content\":\"Say hi\"}],\"max_tokens\":6,\"temperature\":0,\"stream\":true}", &rl);
    char *sresp = http_raw(req, rl, &status);
    sgl_xfree(req);
    CHECK(sresp && status == 200, "stream status=%d", status);
    int n_data = 0, has_done = 0;
    char joined[512] = "";
    if (sresp) {
      char *line = strtok(sresp, "\n");
      while (line) {
        if (strncmp(line, "data: ", 6) == 0) {
          n_data++;
          const char *payload = line + 6;
          if (strcmp(payload, "[DONE]") == 0) has_done = 1;
          else {
            sgl_json *ev = sgl_json_parse(payload, strlen(payload));
            const sgl_json *ch = sgl_json_get(ev, "choices");
            const sgl_json *delta = ch && ch->count ? sgl_json_get(ch->items[0], "delta") : NULL;
            const char *c = sgl_json_get_str(delta, "content", "");
            if (c) strncat(joined, c, sizeof(joined) - strlen(joined) - 1);
            if (ev) sgl_json_free(ev);
          }
        }
        line = strtok(NULL, "\n");
      }
    }
    CHECK(n_data >= 2 && has_done, "SSE 事件数=%d done=%d", n_data, has_done);
    /* 非流式同参对照 */
    req = build_post("/v1/chat/completions",
                     "{\"messages\":[{\"role\":\"user\",\"content\":\"Say hi\"}],\"max_tokens\":6,\"temperature\":0}", &rl);
    char *nresp = http_raw(req, rl, &status);
    sgl_xfree(req);
    const char *want = "";
    if (nresp && status == 200) {
      sgl_json *r = sgl_json_parse(nresp, strlen(nresp));
      const sgl_json *ch = sgl_json_get(r, "choices");
      const sgl_json *msg = ch && ch->count ? sgl_json_get(ch->items[0], "message") : NULL;
      want = sgl_json_get_str(msg, "content", "");
      CHECK(strcmp(joined, want) == 0, "流式拼接与非流式不一致：%s vs %s", joined, want);
      sgl_json_free(r);
    }
    sgl_xfree(nresp);
    sgl_xfree(sresp);
    printf("[stream] %d 事件，拼接 %zu 字符 ✓\n", n_data, strlen(joined));
  }

  /* 6. stop 截断：stop="the" 应截在 REF_A 中首个 "the"处，finish_reason=stop */
  {
    size_t rl;
    char body[512];
    snprintf(body, sizeof(body),
             "{\"model\":\"gpt2\",\"prompt\":\"The meaning of life is\",\"max_tokens\":8,\"temperature\":0,\"stop\":[\"the\"]}");
    char *req = build_post("/v1/completions", body, &rl);
    char *sresp = http_raw(req, rl, &status);
    sgl_xfree(req);
    CHECK(sresp && status == 200, "stop status=%d", status);
    if (sresp) {
      sgl_json *r = sgl_json_parse(sresp, strlen(sresp));
      const sgl_json *ch = sgl_json_get(r, "choices");
      const char *txt = ch && ch->count ? sgl_json_get_str(ch->items[0], "text", NULL) : NULL;
      const char *fr = ch && ch->count ? sgl_json_get_str(ch->items[0], "finish_reason", "") : "";
      /* REF_A = " not the same..."；截断后应为 " not "（首个 the 前，含尾空格）*/
      CHECK(txt && strcmp(txt, " not ") == 0, "stop 截断结果=[%s] 应=[ not 空格]", txt ? txt : "");
      CHECK(strcmp(fr, "stop") == 0, "finish_reason=%s", fr);
      sgl_json_free(r);
      printf("[stop] 截断 ✓（\"%s\"）\n", txt);
    }
    sgl_xfree(sresp);
  }

  /* 7. n=2：两个 choices（temperature>0 + 不同 seed）*/
  {
    size_t rl;
    char body[512];
    snprintf(body, sizeof(body),
             "{\"model\":\"gpt2\",\"prompt\":\"The meaning of life is\",\"max_tokens\":6,\"temperature\":0.9,\"n\":2}");
    char *req = build_post("/v1/completions", body, &rl);
    char *nresp = http_raw(req, rl, &status);
    sgl_xfree(req);
    CHECK(nresp && status == 200, "n=2 status=%d", status);
    if (nresp) {
      sgl_json *r = sgl_json_parse(nresp, strlen(nresp));
      const sgl_json *ch = sgl_json_get(r, "choices");
      CHECK(ch && ch->count == 2, "choices=%d（应 2）", ch ? ch->count : 0);
      if (ch && ch->count == 2) {
        const char *t0 = sgl_json_get_str(ch->items[0], "text", NULL);
        const char *t1 = sgl_json_get_str(ch->items[1], "text", NULL);
        CHECK(t0 && *t0 && t1 && *t1, "两个 choice 均非空");
        printf("[n=2] choice0=%.20s… choice1=%.20s… ✓\n", t0, t1);
      }
      sgl_json_free(r);
    }
    sgl_xfree(nresp);
  }

  /* 8. 错误路径：404 与非法 JSON */
  resp = http_raw("GET /nope HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n", 50, &status);
  CHECK(status == 404, "404 got %d", status);
  sgl_xfree(resp);
  {
    size_t rl;
    char *req = build_post("/generate", "{bad json", &rl);
    resp = http_raw(req, rl, &status);
    sgl_xfree(req);
    CHECK(status == 400, "400 got %d", status);
    sgl_xfree(resp);
  }

  /* 9. P1 新增端点：/v1/models、/get_server_info、/metrics、/flush_cache、
   *    /update_weights_from_disk、/abort_request、/ready */
  resp = http_raw("GET /v1/models HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n",
                  strlen("GET /v1/models HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n"), &status);
  CHECK(resp && status == 200 && strstr(resp, "\"object\":\"model\""), "/v1/models: %d", status);
  sgl_xfree(resp);

  resp = http_raw("GET /get_server_info HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n",
                  strlen("GET /get_server_info HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n"),
                  &status);
  CHECK(resp && status == 200 && strstr(resp, "\"pool_pages\""), "/get_server_info: %d", status);
  sgl_xfree(resp);

  resp = http_raw("GET /metrics HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n",
                  strlen("GET /metrics HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n"), &status);
  CHECK(resp && status == 200 && strstr(resp, "sgl_decode_tokens_total"), "/metrics: %d", status);
  sgl_xfree(resp);

  resp = http_raw("GET /ready HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n",
                  strlen("GET /ready HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n"), &status);
  CHECK(resp && status == 200, "/ready: %d", status);
  sgl_xfree(resp);

  {
    size_t rl;
    char *req = build_post("/flush_cache", "{}", &rl);
    resp = http_raw(req, rl, &status);
    sgl_xfree(req);
    CHECK(resp && status == 200, "/flush_cache: %d", status);
    sgl_xfree(resp);
  }

  {
    /* 热更新同一目录后仍能正常生成（且结果不变）*/
    size_t rl;
    char body[1200];
    snprintf(body, sizeof(body), "{\"model_path\":\"%s\"}", dir);
    char *req = build_post("/update_weights_from_disk", body, &rl);
    resp = http_raw(req, rl, &status);
    sgl_xfree(req);
    CHECK(resp && status == 200, "/update_weights_from_disk: %d", status);
    sgl_xfree(resp);
    gen_job c = { "The meaning of life is", 8, REF_A, 0 };
    gen_thread(&c);
    CHECK(c.ok, "热更新后 /generate 结果与参考不一致");
  }

  {
    size_t rl;
    char *req = build_post("/abort_request", "{\"rid\":999999}", &rl);
    resp = http_raw(req, rl, &status);
    sgl_xfree(req);
    CHECK(status == 404, "abort_request 未知 rid 应 404，got %d", status);
    sgl_xfree(resp);
  }

  /* 10. P1 约束解码：response_format=json_schema 下输出必须是合法 JSON */
  {
    size_t rl;
    char *req = build_post(
        "/v1/chat/completions",
        "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":24,"
        "\"temperature\":0,\"response_format\":{\"type\":\"json_schema\",\"json_schema\":"
        "{\"name\":\"p\",\"schema\":{\"type\":\"object\",\"properties\":{\"a\":"
        "{\"type\":\"integer\"}},\"required\":[\"a\"],\"additionalProperties\":false}}}}",
        &rl);
    resp = http_raw(req, rl, &status);
    sgl_xfree(req);
    CHECK(resp && status == 200, "grammar chat status=%d", status);
    if (resp) {
      sgl_json *r = sgl_json_parse(resp, strlen(resp));
      const sgl_json *ch = r ? sgl_json_get(r, "choices") : NULL;
      const sgl_json *msg = ch && ch->count ? sgl_json_get(ch->items[0], "message") : NULL;
      const char *txt = msg ? sgl_json_get_str(msg, "content", "") : "";
      sgl_json *cj = sgl_json_parse(txt, strlen(txt));
      CHECK(cj != NULL, "约束输出不是合法 JSON: [%s]", txt);
      if (cj) {
        CHECK(sgl_json_get(cj, "a") != NULL, "约束输出缺少字段 a: [%s]", txt);
        sgl_json_free(cj);
      }
      if (r) sgl_json_free(r);
      sgl_xfree(resp);
    }
  }

  sgl_http_stop(srv);
  sgl_sched_free(sched);
  if (FAILURES) { printf("test_http: %d FAILURES\n", FAILURES); return 1; }
  printf("test_http: all OK\n");
  return 0;
}
