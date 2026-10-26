/* sglc — sglang-c 推理 CLI（对应 sglang 的 python -m sglang.launch_server 单机路径）*/
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sgl/bpe.h"
#include "sgl/clip.h"
#include "sgl/generate.h"
#include "sgl/gpt2.h"
#include "sgl/image.h"
#include "sgl/json.h"
#include "sgl/log.h"
#include "sgl/alloc.h"
#include "sgl/http.h"
#include "sgl/safetensors.h"
#include "sgl/scheduler.h"

typedef struct {
  FILE *out;
  int first;
} cb_ctx;

static int on_token(void *user, int tok, const char *piece) {
  cb_ctx *c = (cb_ctx *)user;
  fputs(piece, c->out);
  fflush(c->out);
  return 0;
}

static double now_sec(void) {
  struct timespec ts;
  timespec_get(&ts, TIME_UTC);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ---- G8 多模态（--vision <clip 目录> --projector <projector.safetensors>）----
 * http.c 收到 OpenAI vision 风格的 image_url 后，把图片字节交给这个回调：
 * 解码 → resize/crop → 归一化 → CLIP 视觉塔 → mm_projector → [n_img][E_lm]。 */
static sgl_clip *g_clip;
static float *g_pj_w, *g_pj_b;
static int g_e_lm;

static float *vision_cb(void *ud, const unsigned char *bytes, size_t n,
                        int *n_img, int *e_lm) {
  (void)ud;
  if (!g_clip || !g_pj_w) return NULL;
  float *out = NULL;
  int nn = 0;
  if (sgl_clip_project_bytes(bytes, n, g_clip, g_pj_w, g_pj_b, g_e_lm, &out, &nn) != 0)
    return NULL;
  *n_img = g_clip->n_patch;
  *e_lm = g_e_lm;
  return out;
}

/* 读模型 config.json 的 hidden_size / n_embd（LM embedding 宽）*/
static int cfg_e_lm(const char *dir) {
  char path[1024];
  snprintf(path, sizeof(path), "%s/config.json", dir);
  FILE *f = NULL;
  fopen_s(&f, path, "rb");
  if (!f) return 0;
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  char *buf = (char *)sgl_xmalloc((size_t)sz + 1);
  if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); sgl_xfree(buf); return 0; }
  buf[sz] = 0;
  fclose(f);
  sgl_json *j = sgl_json_parse(buf, (size_t)sz);
  sgl_xfree(buf);
  if (!j) return 0;
  int e = sgl_json_get_int(j, "hidden_size", sgl_json_get_int(j, "n_embd", 0));
  sgl_json_free(j);
  return e;
}

static int vision_load(const char *clip_dir, const char *proj_path, int e_lm) {
  g_clip = sgl_clip_load(clip_dir);
  if (!g_clip) return -1;
  sgl_st_ctx *st = sgl_st_open(proj_path);
  if (!st) { sgl_clip_free(g_clip); g_clip = NULL; return -1; }
  sgl_tensor w = sgl_st_get(st, "linear_1.weight", 2, (int64_t[]){e_lm, g_clip->E});
  sgl_tensor b = sgl_st_get(st, "linear_1.bias", 1, (int64_t[]){e_lm});
  if (!w.data) {
    SGL_ERROR("vision: %s 里没 linear_1.weight [%d,%d]", proj_path, e_lm, g_clip->E);
    sgl_st_close(st);
    sgl_clip_free(g_clip);
    g_clip = NULL;
    return -1;
  }
  g_pj_w = (float *)sgl_xmalloc(sizeof(float) * (size_t)e_lm * (size_t)g_clip->E);
  memcpy(g_pj_w, w.data, sizeof(float) * (size_t)e_lm * (size_t)g_clip->E);
  if (b.data) {
    g_pj_b = (float *)sgl_xmalloc(sizeof(float) * (size_t)e_lm);
    memcpy(g_pj_b, b.data, sizeof(float) * (size_t)e_lm);
  }
  sgl_st_close(st);
  g_e_lm = e_lm;
  SGL_INFO("vision ready: CLIP E=%d L=%d P=%d img=%d NP=%d quick_gelu=%d, projector [%d,%d]",
           g_clip->E, g_clip->L, g_clip->P, g_clip->img, g_clip->n_patch,
           g_clip->act == SGL_CLIP_ACT_QUICK, e_lm, g_clip->E);
  return 0;
}

static void usage(const char *prog) {
  fprintf(stderr,
          "usage: %s --model <dir> [--prompt <text>] [--max-new N] \
          [--temperature F] [--top-k N] [--top-p F] [--seed N] [--verbose] \
          [--serve [--host H] [--port P]] [--w8] [--kv8] \
          [--lora name=dir[,name2=dir2,...]]\n"
          "  <dir> 需含 config.json / tokenizer.json / model.safetensors\n"
          "  --w8   int8 权重（W8A16，decode ~1.8x）\n"
          "  --kv8  int8 KV cache（内存减半）\n"
          "  --serve 启动 HTTP 服务（/health /generate /v1/completions），Ctrl+C 退出\n"
          "  --lora 多 LoRA 实例池：每个 adapter 预先合并为独立模型副本（独立 KV 池），\n"
          "         请求 JSON 顶层带 \"lora\":\"name\" 路由；不带该字段走 base 实例\n"
          "  例: %s --model gpt2 --prompt \"The meaning of life is\" --max-new 32\n"
          "      %s --model gpt2 --serve --port 30000\n"
          "      %s --model model --serve --lora a=models/adapter-a,b=models/adapter-b\n",
          prog, prog, prog, prog);
}

/* --lora 规格说明（解析结果）*/
typedef struct {
  char name[64];
  char dir[512];
} lora_spec;

/* 为每个 adapter 建一个 base+合并 delta 的实例副本（服务与单发共用）*/
static int sched_add_loras(sgl_scheduler *sc, const char *model_dir,
                           const lora_spec *loras, int n_loras) {
  for (int i = 0; i < n_loras; i++)
    if (sgl_sched_add_lora(sc, loras[i].name, model_dir, loras[i].dir) < 0) return -1;
  return 0;
}

int main(int argc, char **argv) {
  const char *model_dir = NULL, *prompt = "The meaning of life is";
  const char *host = "127.0.0.1";
  const char *vision_dir = NULL, *proj_path = NULL; /* G8 */
  int max_new = 32, port = 30000, serve = 0, w8 = 0, kv8 = 0;
  sgl_sampling_params sp = {0.8f, 50, 0.95f, 0};
  lora_spec loras[16];
  int n_loras = 0;

  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--model") && i + 1 < argc) model_dir = argv[++i];
    else if (!strcmp(argv[i], "--vision") && i + 1 < argc) vision_dir = argv[++i];
    else if (!strcmp(argv[i], "--projector") && i + 1 < argc) proj_path = argv[++i];
    else if (!strcmp(argv[i], "--prompt") && i + 1 < argc) prompt = argv[++i];
    else if (!strcmp(argv[i], "--max-new") && i + 1 < argc) max_new = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--temperature") && i + 1 < argc) sp.temperature = (float)atof(argv[++i]);
    else if (!strcmp(argv[i], "--top-k") && i + 1 < argc) sp.top_k = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--top-p") && i + 1 < argc) sp.top_p = (float)atof(argv[++i]);
    else if (!strcmp(argv[i], "--seed") && i + 1 < argc) sp.seed = (unsigned long long)atoll(argv[++i]);
    else if (!strcmp(argv[i], "--verbose")) sgl_log_set_level(SGL_LOG_DEBUG);
    else if (!strcmp(argv[i], "--serve")) serve = 1;
    else if (!strcmp(argv[i], "--host") && i + 1 < argc) host = argv[++i];
    else if (!strcmp(argv[i], "--port") && i + 1 < argc) port = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--w8")) w8 = 1;
    else if (!strcmp(argv[i], "--kv8")) kv8 = 1;
    else if (!strcmp(argv[i], "--lora") && i + 1 < argc) {
      /* name=dir[,name2=dir2,...]（启动时单线程解析，strtok 可用）*/
      char *spec = argv[++i];
      for (char *tok = strtok(spec, ","); tok; tok = strtok(NULL, ",")) {
        char *eq = strchr(tok, '=');
        if (!eq || eq == tok || !eq[1] || n_loras >= 16) {
          fprintf(stderr, "bad --lora spec '%s' (want name=dir[,name2=dir2,...])\n", tok);
          return 1;
        }
        *eq = 0;
        snprintf(loras[n_loras].name, sizeof(loras[n_loras].name), "%s", tok);
        snprintf(loras[n_loras].dir, sizeof(loras[n_loras].dir), "%s", eq + 1);
        n_loras++;
      }
    }
    else { usage(argv[0]); return 1; }
  }
  if (!model_dir) { usage(argv[0]); return 1; }

  /* 量化开关（llama 后端）：须在模型加载前生效 */
#ifdef _WIN32
#define SGLC_SETENV(k, v) _putenv_s(k, v)
#else
#define SGLC_SETENV(k, v) setenv(k, v, 1)
#endif
  if (w8) SGLC_SETENV("SGL_W8", "1");
  if (kv8) SGLC_SETENV("SGL_KV8", "1");
  /* --lora 时 base 实例必须加载干净权重（合并由 add_lora 按 adapter 逐份做）*/
  if (n_loras > 0) SGLC_SETENV("SGL_LORA", "");

  double t0 = now_sec();

  if (serve) {
#ifdef _MSC_VER
    /* MSVC 构建不含 pthread 版 http.c（MinGW 专用）；服务端用 cuda_http.exe */
    fprintf(stderr, "--serve not available in MSVC builds (use cuda_http.exe)\n");
    return 1;
#else
    /* 服务模式：调度器 + HTTP（对应 sglang launch_server）*/
    sgl_scheduler *sched = sgl_sched_new(model_dir, 0);
    if (!sched) return 2;
    if (n_loras > 0 && sched_add_loras(sched, model_dir, loras, n_loras) != 0) { sgl_sched_free(sched); return 2; }
    SGL_INFO("model loaded in %.2fs", now_sec() - t0);
    sgl_http_server *srv = sgl_http_serve(sched, host, port);
    if (!srv) { sgl_sched_free(sched); return 2; }
    /* G8: 装了视觉塔就注册图片回调（/v1/chat/completions 的 image_url）*/
    if (vision_dir) {
      char pj[1024];
      if (!proj_path) {
        snprintf(pj, sizeof(pj), "%s/projector.safetensors", vision_dir);
        proj_path = pj;
      }
      int e_lm = cfg_e_lm(model_dir);
      if (e_lm <= 0 || vision_load(vision_dir, proj_path, e_lm) != 0) {
        SGL_ERROR("--vision 初始化失败（检查 %s / %s，LM E=%d）", vision_dir, proj_path, e_lm);
        sgl_http_stop(srv); sgl_sched_free(sched); return 2;
      }
      sgl_http_set_vision(srv, vision_cb, NULL);
    }
    fputs("press Enter to stop...\n", stderr);
    getchar(); /* 简单退出方式：Ctrl+C 或回车 */
    sgl_http_stop(srv);
    sgl_sched_free(sched);
    return 0;
#endif
  }

  /* 单发路径：走调度器（全后端，含量化/radix）*/
  {
    sgl_scheduler *sched = sgl_sched_new(model_dir, sp.seed);
    if (!sched) return 2;
    if (n_loras > 0 && sched_add_loras(sched, model_dir, loras, n_loras) != 0) { sgl_sched_free(sched); return 2; }
    SGL_INFO("model loaded in %.2fs", now_sec() - t0);
    int id = sgl_sched_submit(sched, prompt, max_new, &sp);
    if (id < 0) { sgl_sched_free(sched); return 2; }
    int guard = 0;
    while (sgl_sched_nactive(sched) > 0 && guard++ < 100000) sgl_sched_step(sched);
    int rid, n_out, npt, nhit;
    int *out;
    char *text;
    while (sgl_sched_poll(sched, &rid, &text, &out, &n_out, &npt, &nhit)) {
      if (rid == id) {
        fputs(prompt, stdout);
        fputs(text, stdout);
        fputc('\n', stdout);
        sgl_sched_stats st = sgl_sched_get_stats(sched);
        SGL_INFO("generated %d tokens in %.2fs (%.1f tok/s, %.0f ms/tok, cache_hit=%ld)",
                 n_out, now_sec() - t0, n_out / (now_sec() - t0),
                 (now_sec() - t0) * 1000 / (n_out ? n_out : 1), st.cache_hit_tokens);
      }
      sgl_xfree(text);
      sgl_xfree(out);
    }
    sgl_sched_free(sched);
    return 0;
  }
}
