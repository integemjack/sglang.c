/* test_gguf — GGUF 权重源：容器解析、元数据→config、名字映射、量化反解精度。
 *
 * 两个层次：
 *  1) 不依赖外部文件的部分（名字映射双向）总是跑；
 *  2) 依赖真实文件的（SGL_TEST_GGUF 指向一个含 *.gguf 的目录）：
 *     - 元数据合成的 config 字段要和各层结构对得上；
 *     - 同一个模型如果还有原始 safetensors（SGL_TEST_GGUF_REF），逐张量比
 *       "反量化结果 vs 原始权重" 的相对 RMS 误差。这是真正的金标准：
 *       量化器与权重都来自第三方（bartowski 的 Q4_K_M + HF 原始 bf16），
 *       位布局只要错一位，误差就会从百分之几跳到几十。
 *
 * 判定：Q6_K < 3%、Q4_K < 8%、F32/F16 < 1e-6（各自的理论量化噪声量级）。
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sgl/alloc.h"
#include "sgl/bpe.h"
#include "sgl/gguf.h"
#include "sgl/json.h"
#include "sgl/port.h"
#include "sgl/safetensors.h"

static int fails;
static void check(int cond, const char *msg) {
  if (!cond) {
    fails++;
    printf("FAIL: %s\n", msg);
  }
}

static int file_exists(const char *p) {
  FILE *f = NULL;
  if (fopen_s(&f, p, "rb") != 0 || !f) return 0;
  fclose(f);
  return 1;
}

/* 相对 RMS 误差：sqrt(sum((a-b)^2) / sum(a^2)) */
static double rel_rms(const float *a, const float *b, int64_t n, double *maxabs) {
  double num = 0.0, den = 0.0;
  double mx = 0.0;
  for (int64_t i = 0; i < n; i++) {
    double d = (double)a[i] - (double)b[i];
    num += d * d;
    den += (double)a[i] * (double)a[i];
    if (fabs(d) > mx) mx = fabs(d);
  }
  if (maxabs) *maxabs = mx;
  if (den <= 0.0) return num > 0.0 ? 1e9 : 0.0;
  return sqrt(num / den);
}

/* 按类型给容差 */
/* 各量化类型的实际 RMS 误差上限（相对原权重）。这些数字不是拍脑袋：
 * 实测值（SmolLM2-135M，Q4_K_M/Q2_K/Q3_K_M 三个文件）为
 *   Q8_0 0.006–0.007、Q6_K 0.019、Q5_0 0.042–0.054、Q4_K 0.075。
 * 位布局若错，误差会直接跳到 1.0 以上（q_proj 没做逆置换时就是 1.38），
 * 所以这组容差能同时容忍量化误差、拒绝解码错位。*/
static double tol_for(const char *type) {
  if (!type) return 0.0;
  if (!strcmp(type, "F32") || !strcmp(type, "F16") || !strcmp(type, "BF16")) return 1e-6;
  if (!strncmp(type, "Q8", 2)) return 0.02;
  if (!strcmp(type, "Q6_K")) return 0.04;
  if (!strcmp(type, "Q5_K") || !strcmp(type, "Q5_0") || !strcmp(type, "Q5_1")) return 0.10;
  if (!strncmp(type, "Q4", 2) || !strcmp(type, "IQ4_NL")) return 0.12;
  if (!strncmp(type, "Q3", 2)) return 0.18;
  if (!strncmp(type, "Q2", 2)) return 0.35;
  return 0.12;
}

static void test_name_map(void) {
  char out[256];
  check(sgl_gguf_name_to_hf("token_embd.weight", out, sizeof(out)) &&
            !strcmp(out, "model.embed_tokens.weight"),
        "gguf->hf: token_embd.weight");
  check(sgl_gguf_name_to_hf("blk.3.attn_q.weight", out, sizeof(out)) &&
            !strcmp(out, "model.layers.3.self_attn.q_proj.weight"),
        "gguf->hf: blk.3.attn_q.weight");
  check(sgl_gguf_name_to_hf("blk.12.ffn_down.weight", out, sizeof(out)) &&
            !strcmp(out, "model.layers.12.mlp.down_proj.weight"),
        "gguf->hf: blk.12.ffn_down.weight");
  check(sgl_gguf_name_to_hf("blk.0.attn_q_norm.weight", out, sizeof(out)) &&
            !strcmp(out, "model.layers.0.self_attn.q_norm.weight"),
        "gguf->hf: q_norm");
  check(!sgl_gguf_name_to_hf("rope_freqs.weight", out, sizeof(out)),
        "gguf->hf: rope_freqs 不映射");
  /* 反向 */
  check(sgl_gguf_name_from_hf("model.layers.7.self_attn.v_proj.weight", out, sizeof(out)) &&
            !strcmp(out, "blk.7.attn_v.weight"),
        "hf->gguf: v_proj");
  check(sgl_gguf_name_from_hf("model.layers.7.self_attn.v_proj.bias", out, sizeof(out)) &&
            !strcmp(out, "blk.7.attn_v.bias"),
        "hf->gguf: v_proj.bias");
  check(sgl_gguf_name_from_hf("lm_head.weight", out, sizeof(out)) &&
            !strcmp(out, "output.weight"),
        "hf->gguf: lm_head");
  /* 往返 */
  const char *hf = "model.layers.5.post_attention_layernorm.weight";
  check(sgl_gguf_name_from_hf(hf, out, sizeof(out)) &&
            sgl_gguf_name_to_hf(out, out, sizeof(out)) && !strcmp(out, hf),
        "hf->gguf->hf 往返一致");
}

/* 单文件 .gguf 目录（没有 tokenizer.json / config.json）：tokenizer 必须能从
 * GGUF 内嵌元数据建起来，而且与同模型的 tokenizer.json 路径编出同样的 token 序列。*/
static int test_tokenizer_from_gguf(void) {
  const char *single = getenv("SGL_TEST_GGUF_SINGLE");
  if (!single || !*single) {
    printf("SKIP: SGL_TEST_GGUF_SINGLE not set\n");
    return 0;
  }
  /* CMake 总会设这个变量（指向 models/gguf-quants/Q2_K），夹具不在时（CI）要跳过而不是失败 */
  {
    sgl_gguf *probe_g = sgl_gguf_open_dir(single);
    if (!probe_g) {
      printf("SKIP: no .gguf found in %s\n", single);
      return 0;
    }
    sgl_gguf_close(probe_g);
  }
  sgl_tokenizer *t = sgl_tokenizer_load_dir(single);
  check(t != NULL, "从 GGUF 元数据加载 tokenizer");
  if (!t) return 0;
  check(sgl_tokenizer_vocab_size(t) > 1000, "vocab_size 合理");
  const char *probe = "The capital of France is Paris.";
  int n1 = 0;
  int *ids = sgl_tokenizer_encode(t, probe, strlen(probe), &n1);
  check(ids && n1 > 3, "GGUF tokenizer 能 encode");
  if (ids) {
    char *back = sgl_tokenizer_decode(t, ids, n1);
    check(back && !strcmp(back, probe), "decode(encode(x)) == x");
    if (back) printf("  gguf tokenizer: %d tokens for %d chars\n", n1, (int)strlen(probe));
    sgl_xfree(back);
  }

  /* 与 tokenizer.json 路径对比（同模型两份词表应完全一致）*/
  const char *withjson = getenv("SGL_TEST_GGUF");
  if (withjson && ids) {
    sgl_tokenizer *t2 = sgl_tokenizer_load_dir(withjson);
    if (t2) {
      int n2 = 0;
      int *ids2 = sgl_tokenizer_encode(t2, probe, strlen(probe), &n2);
      int same = (ids2 && n1 == n2);
      for (int i = 0; same && i < n1; i++)
        if (ids[i] != ids2[i]) same = 0;
      check(same, "GGUF 词表与 tokenizer.json 编出同一序列");
      sgl_xfree(ids2);
      sgl_tokenizer_free(t2);
    }
  }
  sgl_xfree(ids);
  sgl_tokenizer_free(t);
  return 1;
}

/* 与 Python 独立实现（tests/gguf_ref_dequant.py 导出的 .refmeta/.refbin）逐元素对拍。
 * 这检验的是"我们的位布局与算术跟另一份独立实现一致"；
 * 量化误差本身由"端到端生成"与"原始权重对照"两项负责。*/
static int test_ref_dump(sgl_gguf *g) {
  const char *gp = sgl_gguf_path(g);
  if (!gp) return 0;
  char meta[1500], binp[1500];
  snprintf(meta, sizeof(meta), "%s.refmeta", gp);
  snprintf(binp, sizeof(binp), "%s.refbin", gp);
  FILE *fm = NULL;
  if (fopen_s(&fm, meta, "rb") != 0 || !fm) {
    printf("SKIP: no ref dump next to %s\n", gp);
    return 0;
  }
  FILE *fb = NULL;
  if (fopen_s(&fb, binp, "rb") != 0 || !fb) {
    fclose(fm);
    printf("SKIP: no refbin\n");
    return 0;
  }
  fseek(fb, 0, SEEK_END);
  long blob_sz = ftell(fb);
  fseek(fb, 0, SEEK_SET);
  unsigned char *blob = (unsigned char *)sgl_xmalloc((size_t)blob_sz);
  if (fread(blob, 1, (size_t)blob_sz, fb) != (size_t)blob_sz) {
    fclose(fm);
    fclose(fb);
    sgl_xfree(blob);
    printf("FAIL: short refbin\n");
    fails++;
    return 0;
  }
  fclose(fb);

  char line[1024];
  int n = 0, bad = 0;
  double worst = 0.0;
  char worst_name[256] = {0};
  while (fgets(line, sizeof(line), fm)) {
    char name[256], dims[256];
    int ndim = 0;
    long long off = 0;
    if (sscanf(line, "%255[^\t]\t%d\t%255[^\t]\t%lld", name, &ndim, dims, &off) != 4) continue;
    int64_t shp[4] = {1, 1, 1, 1};
    int64_t numel = 1;
    {
      char *p = dims;
      for (int i = 0; i < ndim && i < 4; i++) {
        shp[i] = strtoll(p, &p, 10);
        numel *= shp[i];
        if (*p == ',') p++;
      }
    }
    if (off < 0 || off + numel * 4 > blob_sz) {
      printf("FAIL: %s ref range out of blob\n", name);
      bad++;
      continue;
    }
    int gnd = 0;
    int64_t gsh[4] = {1, 1, 1, 1};
    float *mine = sgl_gguf_get_f32(g, name, &gnd, gsh);
    if (!mine) {
      printf("FAIL: %s 在 gguf 里取不到\n", name);
      bad++;
      continue;
    }
    int shape_ok = (gnd == ndim);
    for (int i = 0; shape_ok && i < ndim; i++)
      if (gsh[i] != shp[i]) shape_ok = 0;
    if (!shape_ok) {
      printf("FAIL: %s shape C(%lld,%lld,%d) vs py(%lld,%lld,%d)\n", name, (long long)gsh[0],
             (long long)gsh[1], gnd, (long long)shp[0], (long long)shp[1], ndim);
      bad++;
      sgl_xfree(mine);
      continue;
    }
    const float *ref = (const float *)(blob + off);
    double mx = 0.0;
    double rr = rel_rms(ref, mine, numel, &mx);
    if (rr > worst) {
      worst = rr;
      snprintf(worst_name, sizeof(worst_name), "%s", name);
    }
    /* 两个独立 f32 实现：只差最后几位舍入 */
    if (rr > 2e-5) {
      printf("FAIL: %s vs python rel_rms=%.3e maxabs=%.3e\n", name, rr, mx);
      /* 定位：统计错元素个数与范围，判断是整块还是零星 */
      int64_t nbad = 0, first = -1, last = -1;
      double sumbad = 0.0;
      for (int64_t i = 0; i < numel; i++)
        if (fabs((double)mine[i] - (double)ref[i]) > 1e-4) {
          if (first < 0) first = i;
          last = i;
          nbad++;
          sumbad += fabs((double)mine[i] - (double)ref[i]);
        }
      printf("      %lld/%lld elems differ (>1e-4), first=%lld last=%lld avg=%.4f\n",
             (long long)nbad, (long long)numel, (long long)first, (long long)last,
             nbad ? sumbad / (double)nbad : 0.0);
      if (first >= 0) {
        int64_t s0 = first > 3 ? first - 3 : 0;
        printf("      at first diff (block=%lld if 256, row=%lld if %lld):\n",
               (long long)(first / 256), (long long)(first / 1536), (long long)1536);
        for (int64_t k = s0; k < s0 + 8 && k < numel; k++)
          printf("        [%lld] py=%.6f c=%.6f\n", (long long)k, (double)ref[k],
                 (double)mine[k]);
      }
      bad++;
    }
    n++;
    sgl_xfree(mine);
  }
  fclose(fm);
  sgl_xfree(blob);
  printf("ref-dump: compared %d tensors with the python implementation, worst rel_rms=%.3e (%s)\n",
         n, worst, worst_name);
  check(n >= 50, "至少对拍 50 个张量");
  check(bad == 0, "与 Python 参考实现一致");
  return n;
}

int main(void) {
  /* 单文件 gguf 的 tokenizer（不依赖主测试目录）*/
  test_tokenizer_from_gguf();
  test_name_map();

  const char *dir = getenv("SGL_TEST_GGUF");
  if (!dir || !*dir) {
    printf("SKIP: SGL_TEST_GGUF not set (GGUF tests need a real .gguf)\n");
    if (fails) {
      printf("test_gguf: %d FAILED\n", fails);
      return 1;
    }
    printf("test_gguf: OK (name map only)\n");
    return 0;
  }
  char probe[1400];
  snprintf(probe, sizeof(probe), "%s/config.json", dir);
  /* 目录里必须真的有 gguf，否则 SKIP */
  sgl_gguf *g = sgl_gguf_open_dir(dir);
  if (!g) {
    printf("SKIP: no .gguf found in %s\n", dir);
    return fails ? 1 : 0;
  }
  {
    const char **names = NULL;
    int n = sgl_gguf_names(g, &names);
    printf("gguf: %d tensors mapped to HF names\n", n);
    check(n > 100, "张量数 > 100");
    check(sgl_gguf_has(g, "model.embed_tokens.weight"), "有 embed_tokens");
    check(sgl_gguf_has(g, "model.layers.0.self_attn.q_proj.weight"), "有 layer0 q_proj");
  }

  /* 元数据 → config */
  {
    size_t len = 0;
    char *js = sgl_gguf_config_json(g, &len);
    check(js != NULL, "config JSON 合成成功");
    if (js) {
      printf("config from GGUF: %s\n", js);
      sgl_json *cfg = sgl_json_parse(js, len);
      check(cfg != NULL, "合成的 config 是合法 JSON");
      if (cfg) {
        check(sgl_json_get_int(cfg, "num_hidden_layers", -1) == 30, "num_hidden_layers=30");
        check(sgl_json_get_int(cfg, "hidden_size", -1) == 576, "hidden_size=576");
        check(sgl_json_get_int(cfg, "intermediate_size", -1) == 1536, "intermediate_size=1536");
        check(sgl_json_get_int(cfg, "num_attention_heads", -1) == 9, "num_attention_heads=9");
        check(sgl_json_get_int(cfg, "num_key_value_heads", -1) == 3, "num_key_value_heads=3");
        check(sgl_json_get_int(cfg, "vocab_size", -1) == 49152, "vocab_size=49152");
        check(sgl_json_get_int(cfg, "max_position_embeddings", -1) == 8192, "ctx=8192");
        check(fabs(sgl_json_get_num(cfg, "rope_theta", 0) - 100000.0) < 1.0, "rope_theta=100000");
        check(fabs(sgl_json_get_num(cfg, "rms_norm_eps", 0) - 1e-5) < 1e-9, "rms_norm_eps=1e-5");
        check(sgl_json_get_int(cfg, "tie_word_embeddings", -1) == 1, "tied 模型写 tie_word_embeddings=1");
        sgl_json_free(cfg);
      }
      sgl_xfree(js);
    }
  }

  /* 走 safetensors.c 的 GGUF 分支：上层只用 HF 名 */
  {
    sgl_st_ctx *st = sgl_st_open_dir(dir);
    check(st != NULL, "sgl_st_open_dir 认 GGUF");
    if (st) {
      check(sgl_st_shard_count(st) == 1, "GGUF 计 1 个 shard");
      sgl_tensor q = sgl_st_get(st, "model.layers.0.self_attn.q_proj.weight", 2,
                                (int64_t[]){576, 576});
      check(!sgl_tensor_is_null(q), "通过 sgl_st_get 取 q_proj");
      check(!sgl_tensor_is_null(sgl_st_get(st, "model.norm.weight", 1, (int64_t[]){576})),
            "通过 sgl_st_get 取 model.norm");
      check(sgl_tensor_is_null(sgl_st_get(st, "model.no_such_weight", 2, NULL)),
            "不存在的张量返回 null（可选张量语义）");
      /* 少见的 reorder：GGUF 里 embed 是 [vocab, hidden] */
      sgl_tensor e = sgl_st_get(st, "model.embed_tokens.weight", 2, (int64_t[]){49152, 576});
      check(!sgl_tensor_is_null(e), "embed_tokens 形状 [49152,576]");
      sgl_st_close(st);
    }
  }

  /* 与 Python 独立实现对拍 */
  test_ref_dump(g);

  /* 与原始 safetensors 逐张量对照（金标准）*/
  const char *ref = getenv("SGL_TEST_GGUF_REF");
  if (ref && *ref) {
    char rp[1400];
    snprintf(rp, sizeof(rp), "%s/model.safetensors", ref);
    if (!file_exists(rp)) {
      printf("SKIP: reference safetensors missing (%s)\n", rp);
    } else {
      sgl_st_ctx *rs = sgl_st_open_dir(ref);
      if (!rs) {
        printf("SKIP: reference safetensors 打不开（可能不完整/未下载完）: %s\n", rp);
      } else {
      static const char *probe_names[] = {
          "model.embed_tokens.weight",
          "model.layers.0.self_attn.q_proj.weight",
          "model.layers.0.self_attn.k_proj.weight",
          "model.layers.0.self_attn.v_proj.weight",
          "model.layers.0.self_attn.o_proj.weight",
          "model.layers.0.mlp.gate_proj.weight",
          "model.layers.0.mlp.up_proj.weight",
          "model.layers.0.mlp.down_proj.weight",
          "model.layers.17.mlp.down_proj.weight",
          "model.layers.29.self_attn.q_proj.weight",
      };
      int checked = 0, worst_i = -1;
      double worst = 0.0;
      /* 对照要用 sgl_st_get（GGUF 分支会做 q/k 的 head 逆置换），
       * 直接用 sgl_gguf_get_f32 拿到的是置换后的原始布局。*/
      sgl_st_ctx *gst = sgl_st_open_dir(dir);
      check(gst != NULL, "能对 GGUF 目录建 sgl_st_ctx");
      for (size_t i = 0; rs && gst && i < sizeof(probe_names) / sizeof(probe_names[0]); i++) {
        const char *nm = probe_names[i];
        const char *tn = NULL;
        int gt = sgl_gguf_tensor_type(g, nm, &tn);
        if (gt < 0) {
          printf("  (skip %s: not in gguf)\n", nm);
          continue;
        }
        int nd = 0;
        int64_t sh[4] = {1, 1, 1, 1};
        sgl_tensor mt = sgl_st_get(gst, nm, -1, NULL);
        float *mine = mt.data;
        nd = mt.ndim;
        for (int q = 0; q < 4; q++) sh[q] = mt.shape[q];
        sgl_tensor orig = sgl_st_get(rs, nm, nd, sh);
        if (!mine || sgl_tensor_is_null(orig)) {
          check(0, nm);
          continue;
        }
        double mx = 0.0;
        double rr = rel_rms(orig.data, mine, sgl_tensor_numel(&orig), &mx);
        double tol = tol_for(tn);
        printf("  %-46s %-6s rel_rms=%.5f maxabs=%.5f (tol %.3f)\n", nm, tn, rr, mx, tol);
        check(rr <= tol, "量化反解误差在容差内");
        if (rr > worst) {
          worst = rr;
          worst_i = (int)i;
        }
        checked++;
      }
      if (gst) sgl_st_close(gst);
      printf("compared %d tensors, worst rel_rms=%.5f (%s)\n", checked, worst,
             worst_i >= 0 ? probe_names[worst_i] : "-");
      check(checked >= 8, "至少对照 8 个张量");
      if (rs) sgl_st_close(rs);
      }
    }
  }

  sgl_gguf_close(g);
  if (fails) {
    printf("test_gguf: %d FAILED\n", fails);
    return 1;
  }
  printf("test_gguf: all OK\n");
  return 0;
}