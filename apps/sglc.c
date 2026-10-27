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
#include "sgl/port.h"
#include "sgl/http.h"
#include "sgl/safetensors.h"
#include "sgl/scheduler.h"

#ifdef SGL_EMBED_CUDA_HTTP
#include "sgl/cuda.h" /* sgl_cuda_avail_vram_mb：自动选择后端时的显卡探针 */
/* cuda_http.c 的引擎入口（那里凡 main 部分用 SGL_EMBED_CUDA_HTTP 包住）*/
int sgl_cuda_http_serve(const char *dir, int port, const char *host);
#endif

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
static float *g_pj_w, *g_pj_b, *g_pj_w2, *g_pj_b2;
static int g_e_lm, g_pj_mid;
static sgl_projector g_pj; /* G12: 单层 linear 或两层 MLP（linear_1→GELU→linear_2）*/

static float *vision_cb(void *ud, const unsigned char *bytes, size_t n,
                        int *n_img, int *e_lm) {
  (void)ud;
  if (!g_clip || !g_pj.w1) return NULL;
  float *out = NULL;
  int nn = 0;
  if (sgl_clip_project_bytes_pj(bytes, n, g_clip, &g_pj, &out, &nn) != 0)
    return NULL;
  *n_img = g_clip->n_patch;
  *e_lm = g_pj.out;
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
  /* G12: 从权重形状自动判层数。llava 官方检查点里 mm_projector 是
   * Sequential(Linear, GELU, Linear)，导出成 linear_1 / linear_2 两个 Linear；
   * 也有只导一层的旧权重（t10 的随机模型）。*/
  int E = g_clip->E;
  int mid = 0, has2 = 0;
  /* 先按 e_lm 试，再按「任意 2D」试（mid != e_lm 的小模型也要能认）。*/
  sgl_tensor w2 = sgl_st_get(st, "linear_2.weight", 2, (int64_t[]){e_lm, e_lm});
  if (!w2.data) w2 = sgl_st_get(st, "linear_2.weight", -1, NULL); /* -1 = 不限维数/形状 */
  if (w2.data && w2.ndim == 2) has2 = 1;
  if (has2) mid = (int)w2.shape[1];
  sgl_tensor w = sgl_st_get(st, "linear_1.weight", 2,
                            has2 ? (int64_t[]){mid, E} : (int64_t[]){e_lm, E});
  if (!w.data && has2) w = sgl_st_get(st, "linear_1.weight", -1, NULL);
  if (!w.data) {
    SGL_ERROR("vision: %s 里没 linear_1.weight [%d,%d]", proj_path, e_lm, E);
    sgl_st_close(st);
    sgl_clip_free(g_clip);
    g_clip = NULL;
    return -1;
  }
  int rows1 = (int)w.shape[0];
  g_pj_w = (float *)sgl_xmalloc(sizeof(float) * (size_t)rows1 * (size_t)E);
  memcpy(g_pj_w, w.data, sizeof(float) * (size_t)rows1 * (size_t)E);
  sgl_tensor b = sgl_st_get(st, "linear_1.bias", 1, (int64_t[]){rows1});
  if (b.data) {
    g_pj_b = (float *)sgl_xmalloc(sizeof(float) * (size_t)rows1);
    memcpy(g_pj_b, b.data, sizeof(float) * (size_t)rows1);
  }
  if (has2) {
    g_pj_w2 = (float *)sgl_xmalloc(sizeof(float) * (size_t)e_lm * (size_t)mid);
    memcpy(g_pj_w2, w2.data, sizeof(float) * (size_t)e_lm * (size_t)mid);
    sgl_tensor b2 = sgl_st_get(st, "linear_2.bias", 1, (int64_t[]){e_lm});
    if (b2.data) {
      g_pj_b2 = (float *)sgl_xmalloc(sizeof(float) * (size_t)e_lm);
      memcpy(g_pj_b2, b2.data, sizeof(float) * (size_t)e_lm);
    }
  }
  if (sgl_projector_init(&g_pj, g_pj_w, g_pj_b, rows1, E,
                         has2 ? g_pj_w2 : NULL, has2 ? g_pj_b2 : NULL,
                         has2 ? e_lm : 0, has2 ? mid : 0) != 0) {
    SGL_ERROR("vision: projector 形状不匹配 ([%d,%d] + [%d,%d])", rows1, E, e_lm, mid);
    sgl_st_close(st);
    sgl_clip_free(g_clip);
    g_clip = NULL;
    return -1;
  }
  g_pj_mid = has2 ? mid : 0;
  sgl_st_close(st);
  g_e_lm = g_pj.out;
  SGL_INFO("vision ready: CLIP E=%d L=%d P=%d img=%d NP=%d quick_gelu=%d, "
           "projector %s %d->%s%d",
           g_clip->E, g_clip->L, g_clip->P, g_clip->img, g_clip->n_patch,
           g_clip->act == SGL_CLIP_ACT_QUICK,
           has2 ? "MLP(linear_1-gelu-linear_2)" : "linear", E,
           has2 ? "(mid " : "(", has2 ? mid : g_e_lm);
  return 0;
}

static void usage(const char *prog) {
  fprintf(stderr,
          "usage: %s --model <dir> [--prompt <text>] [--max-new N] \
          [--temperature F] [--top-k N] [--top-p F] [--seed N] [--verbose] \
          [--serve [--host H] [--port P] [--device auto|cuda|mlx|cpu]] [--w8] [--kv8] \
          [--lora name=dir[,name2=dir2,...]]\n"
          "  <dir> 需含 config.json / tokenizer.json / model.safetensors\n"
          "  --w8   int8 权重（W8A16；CPU 走 decode ~1.8x，CUDA 走 SGL_CUDA_W8）\n"
          "  --kv8  int8 KV cache（CPU 引擎；显存减半）\n"
          "  --serve 启动 HTTP 服务（/health /generate /v1/completions /v1/chat/completions）\n"
          "  --device 服务后端：auto（默认，cuda → mlx → cpu）/ cuda / cpu / mlx\n"
          "           MLX 目前无后端（只有 macOS 的数值参考脚本），指定它会报错后回落 cpu\n"
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

/* 探 CUDA 前先自检一次 cuBLAS 能不能加载。
 *
 * 为什么单独这一步：Windows 的 sglc 把 cublas 编成了 delay-load，而 cudart 是静态库，
 * 所以“有驱动”≠“能算”——只装驱动、没装 CUDA toolkit 的机器上 sgl_cuda_avail_vram_mb()
 * 照样能拿到显存数（cudart 自己的 nvcuda 直调），但第一次 GEMM 去加载 cublas64_*.dll
 * 时失败，崩的是 worker 线程，主线程的 SEH 包不住。这里先用 LoadLibrary 探一下，
 * 加载不了就直接走 CPU，完全不碰 CUDA。
 * 名字由 CMake 注入（SGL_CUBLAS_DLL），因为 DLL 名带 toolkit 版本号（cublas64_13.dll）。*/
#ifndef SGL_CUBLAS_DLL
#define SGL_CUBLAS_DLL ""
#endif

#if defined(_WIN32) && defined(SGL_EMBED_CUDA_HTTP)
#include <windows.h> /* HMODULE / LoadLibraryA / EXCEPTION_EXECUTE_HANDLER（SEH 宏在 winnt.h 里）*/
static int sglc_cublas_loadable(void) {
  if (!SGL_CUBLAS_DLL[0]) return 1; /* 没注入具体名字：交给下面的 SEH 探测兜底 */
  static int cached = -1;
  if (cached >= 0) return cached;
  HMODULE h = LoadLibraryA(SGL_CUBLAS_DLL);
  cached = h ? 1 : 0;
  /* 成功就不卸载：delay-load 随后会在已加载的模块上解析，少一次加载开销 */
  return cached;
}
#else
static int sglc_cublas_loadable(void) { return 1; }
#endif

/* 探一下能不能用 CUDA：返回 0 = 调到了 API（*mb_out 为可用显存，负数表示无卡），
 * 非 0 = 连调用都没走到。
 *
 * 为什么不能直接调 sgl_cuda_avail_vram_mb()：Windows 的 sglc 把 cublas 编成了
 * delay-load（见 CMakeLists 里 /DELAYLOAD 那段），第一次碰 CUDA API 才去加载
 * cublas64_*.dll；机器上没有 NVIDIA 驱动或没装 cuBLAS 时，加载失败会被 delay-load
 * 的默认处理变成一次结构化异常（VcppException(ERROR_MOD_NOT_FOUND)）。不兜住的话
 * 进程就带着 0xC0000135 死掉，“没卡就用 CPU”这条路径根本走不到。
 * __try/__except 在纯 C 函数里可用（不能和 C++ 对象展开混用，这里没有）。*/
#if defined(_MSC_VER) && defined(SGL_EMBED_CUDA_HTTP)
static int sglc_probe_cuda_vram(long *mb_out) {
  __try {
    *mb_out = sgl_cuda_avail_vram_mb();
    return 0;
  } __except (EXCEPTION_EXECUTE_HANDLER) {
    *mb_out = -1;
    return -1;
  }
}
#elif defined(SGL_EMBED_CUDA_HTTP)
static int sglc_probe_cuda_vram(long *mb_out) {
  *mb_out = sgl_cuda_avail_vram_mb();
  return 0;
}
#endif

int main(int argc, char **argv) {
  const char *model_dir = NULL, *prompt = "The meaning of life is";
  const char *host = "127.0.0.1";
  const char *device = "auto"; /* --device：auto / cuda / mlx / cpu */
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
    else if (!strcmp(argv[i], "--device") && i + 1 < argc) device = argv[++i];
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
  if (strcmp(device, "auto") && strcmp(device, "cuda") && strcmp(device, "cpu") &&
      strcmp(device, "mlx")) {
    fprintf(stderr, "unknown --device '%s' (auto|cuda|mlx|cpu)\n", device);
    return 1;
  }

  /* 量化开关（llama 后端）：须在模型加载前生效 */
#ifdef _WIN32
#define SGLC_SETENV(k, v) _putenv_s(k, v)
#else
#define SGLC_SETENV(k, v) setenv(k, v, 1)
#endif
  /* --lora 时 base 实例必须加载干净权重（合并由 add_lora 按 adapter 逐份做）*/
  if (n_loras > 0) SGLC_SETENV("SGL_LORA", "");
  /* 量化开关：serve 分支内部按选中的引擎再设（CUDA 用 SGL_CUDA_W8，不能开 SGL_W8，
   * 否则 CPU 侧权重已被 int8 化，GPU 上传拿到的不是 fp32 源）*/
  if (w8 && !serve) SGLC_SETENV("SGL_W8", "1");
  if (kv8 && !serve) SGLC_SETENV("SGL_KV8", "1");

  double t0 = now_sec();

  if (serve) {
    /* ---- 后端选择（--device）----
     * auto 的次序：cuda → mlx → cpu。
     *  - cuda：编了 SGL_CUDA、且运行时能拿到一块可用显卡（avail_vram > 0）才用它；
     *  - mlx：本仓库没有 MLX 后端（只有 macOS 的 tools/ref_llava_mlx.py 数值参考），
     *         这一档只打一行日志就跳过；
     *  - cpu：回落路径，调度器 + http.c。
     * 两个引擎的端点集一致（/health /generate /v1/completions /v1/chat/completions），
     * 客户端不需要因为换了后端而改代码。*/
    int dev_auto = !strcmp(device, "auto");
    int dev_cuda = dev_auto || !strcmp(device, "cuda");
    if (!strcmp(device, "mlx"))
      fprintf(stderr, "[device] mlx：本仓库未提供 MLX 后端（只有 macOS 数值参考脚本），回落 cpu\n");
    else if (dev_auto)
      SGL_INFO("device: 选择次序 cuda → mlx（未编入）→ cpu");
#ifdef SGL_EMBED_CUDA_HTTP
    if (dev_cuda && !sglc_cublas_loadable()) {
      if (!dev_auto) {
        fprintf(stderr, "[device] --device cuda：%s 加载不了（没装 CUDA toolkit 或显卡驱动）\n",
                SGL_CUBLAS_DLL[0] ? SGL_CUBLAS_DLL : "cuBLAS");
        return 2;
      }
      SGL_INFO("device: %s 加载不了（没装 CUDA toolkit 或显卡驱动）→ cpu",
               SGL_CUBLAS_DLL[0] ? SGL_CUBLAS_DLL : "cuBLAS");
      dev_cuda = 0;
    }
    if (dev_cuda) {
      long free_mb = -1;
      int probed = sglc_probe_cuda_vram(&free_mb); /* 无显卡/无驱动/加载不了 cuBLAS → 负数 */
      if (probed == 0 && free_mb > 0) {
        if (n_loras > 0) {
          /* GPU 引擎（cuda_http）只有单实例；多 LoRA 路由在 CPU 调度器里 */
          fprintf(stderr, "[device] --lora 多实例路由只有 CPU 引擎支持；改用 cpu 引擎\n");
          if (!dev_auto) return 1;
        } else {
          if (w8) SGLC_SETENV("SGL_CUDA_W8", "1"); /* GPU 侧 int8 权重 */
          SGL_INFO("device: cuda（空闲显存 %ld MB）—— GPU 引擎起服务，端口 %d", free_mb, port);
          return sgl_cuda_http_serve(model_dir, port, host);
        }
      } else {
        if (!dev_auto) {
          if (probed != 0)
            fprintf(stderr,
                    "[device] --device cuda：cuBLAS/CUDA 运行时加载不了（无驱动或没装 DLL）\n");
          else
            fprintf(stderr, "[device] --device cuda：没有可用的 CUDA 设备/驱动（avail_vram=%ld MB）\n",
                    free_mb);
          return 2;
        }
        if (probed != 0)
          SGL_INFO("device: cuBLAS/CUDA 运行时加载不了（无驱动或没装 DLL）→ cpu");
        else
          SGL_INFO("device: cuda 不可用（avail_vram=%ld MB）→ cpu", free_mb);
      }
    }
#else
    if (!dev_auto && !strcmp(device, "cuda")) {
      fprintf(stderr, "[device] --device cuda：这个构建没编 CUDA 后端（cmake -DSGL_CUDA=ON）；只用 cpu\n");
      return 1;
    }
    if (dev_auto) SGL_INFO("device: 本构建未编入 CUDA 后端 → cpu");
#endif
    if (w8) SGLC_SETENV("SGL_W8", "1");
    if (kv8) SGLC_SETENV("SGL_KV8", "1");
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
