/* test_qweight — AWQ / GPTQ 的解包与反量化
 *
 * 三层验证：
 *   1) 位布局单元测试：直接断言"哪个自然列/行落在哪个 4bit 槽位"，
 *      期望值由官方打包代码推出（AWQ order_map=[0,2,4,6,1,3,5,7]，
 *      GPTQ 沿 K 每 8 行一个 int32）。
 *   2) 真实检查点切片：models/qweight-real/{awq,gptq} 是从 Qwen 官方发布的
 *      AWQ / GPTQ-Int4 检查点里按 HTTP Range 抠出来的几个张量（脚本
 *      tests/fetch_qweight_slices.py），参考值由 **AutoAWQ 官方 dequantize_gemm
 *      原样代码**（AWQ）和 GPTQModel 的语义（GPTQ）算出。这条证明"我们认的
 *      落盘布局与真实文件一致"。
 *   3) 本地夹具：models/mistral-rand-{awq,gptq,gptq-act} 由
 *      tests/gen_awq_gptq.py 生成，与其未量化原始模型比 rel_rms（4bit 的
 *      理论误差量级）；act-order 夹具的 g_idx 是非连续分组，专门压 g_idx 路径。
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sgl/alloc.h"
#include "sgl/json.h"
#include "sgl/port.h"
#include "sgl/qweight.h"
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

/* ---- marlin 布局定值用例（tests/marlin_layout.py --emit-unit 生成）----
 * 同一份数据由 Python 按三个独立参考打成 marlin（权重 / scales / 零点），
 * 期望值是按**自然列序**算的 (q - z) * s。这里逐元素比，证明 C 的
 * marlin 解包（含 64 列置换、零点二次 interleave、tile 内 word/nibble 序）
 * 与参考完全一致。*/
static void test_marlin_unit(const char *path) {
  FILE *f = NULL;
  if (fopen_s(&f, path, "rb") != 0 || !f) {
    printf("SKIP: marlin 定值用例读不到 %s\n", path);
    return;
  }
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  rewind(f);
  char *txt = (char *)sgl_xmalloc((size_t)sz + 1);
  size_t rd = fread(txt, 1, (size_t)sz, f);
  txt[rd] = 0;
  fclose(f);
  sgl_json *root = sgl_json_parse(txt, rd);
  sgl_xfree(txt);
  if (!root) {
    check(0, "marlin 定值用例 JSON 解析失败");
    return;
  }
  const sgl_json *cases = sgl_json_get(root, "cases");
  int ncase = cases && cases->type == SGL_JSON_ARRAY ? cases->count : 0;
  for (int ci = 0; ci < ncase; ci++) {
    const sgl_json *c = cases->items[ci];
    const int64_t K = sgl_json_get_int(c, "K", 0);
    const int64_t N = sgl_json_get_int(c, "N", 0);
    const int has_z = sgl_json_get_int(c, "has_z", 0);
    const sgl_json *jw = sgl_json_get(c, "qweight");
    const sgl_json *js = sgl_json_get(c, "scales");
    const sgl_json *jz = sgl_json_get(c, "qzeros");
    const sgl_json *je = sgl_json_get(c, "expected");
    if (!jw || !js || !je || K <= 0 || N <= 0 || je->count != (int)(K * N)) {
      check(0, "marlin 定值用例字段不全");
      continue;
    }
    int32_t *qwv = (int32_t *)sgl_xmalloc(sizeof(int32_t) * (size_t)jw->count);
    /* JSON 里可能是无符号 32 位数，先经 uint32 转再落 int32，避开越界转换的 UB */
    for (int i = 0; i < jw->count; i++)
      qwv[i] = (int32_t)(uint32_t)(unsigned long long)jw->items[i]->num;
    float *scv = (float *)sgl_xmalloc(sizeof(float) * (size_t)js->count);
    for (int i = 0; i < js->count; i++) scv[i] = (float)js->items[i]->num;
    int32_t *qzv = NULL;
    if (has_z && jz && jz->count > 0) {
      qzv = (int32_t *)sgl_xmalloc(sizeof(int32_t) * (size_t)jz->count);
      for (int i = 0; i < jz->count; i++)
        qzv[i] = (int32_t)(uint32_t)(unsigned long long)jz->items[i]->num;
    }
    sgl_qcfg q;
    memset(&q, 0, sizeof(q));
    q.method = SGL_Q_AWQ;
    q.bits = (int)sgl_json_get_int(c, "bits", 4);
    q.group_size = sgl_json_get_int(c, "G", 8);
    q.layout = SGL_QLAYOUT_MARLIN;
    float *out = (float *)sgl_xmalloc(sizeof(float) * (size_t)(N * K));
    int rc = sgl_qweight_dequant_marlin(&q, qwv, sgl_json_get_int(c, "qw_rows", 0),
                                       sgl_json_get_int(c, "qw_cols", 0), qzv,
                                       sgl_json_get_int(c, "qz_rows", 0),
                                       sgl_json_get_int(c, "qz_cols", 0), scv,
                                       sgl_json_get_int(c, "sc_rows", 0), N, N, K, out);
    double maxd = 0;
    if (rc == 0) {
      for (int i = 0; i < (int)(N * K); i++) {
        double d = fabs((double)out[i] - je->items[i]->num);
        if (d > maxd) maxd = d;
      }
    }
    char msg[256];
    snprintf(msg, sizeof(msg), "marlin 定值用例[%s] K=%lld N=%lld has_z=%d（rc=%d, maxdiff=%.3g）",
             sgl_json_get_str(c, "name", "?"), (long long)K, (long long)N, has_z, rc, maxd);
    check(rc == 0 && maxd == 0.0, msg);
    printf("%s %s maxdiff=%.3g\n", rc == 0 && maxd == 0.0 ? "PASS" : "FAIL",
           sgl_json_get_str(c, "name", "?"), maxd);
    sgl_xfree(out);
    sgl_xfree(qwv);
    sgl_xfree(scv);
    sgl_xfree(qzv);
  }
  sgl_json_free(root);
}

/* ---------------- 1) 位布局单元测试 ---------------- */

static void test_layout(void) {
  /* AWQ 的 8 元重排：自然列 n 的槽位是 rev[n&7] = [0,4,1,5,2,6,3,7]。
   * 于是"自然列 1 的码"应该出现在 int32 的第 4 个 4bit 槽位（bit 16 起）。*/
  {
    sgl_qcfg cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.method = SGL_Q_AWQ;
    cfg.bits = 4;
    cfg.group_size = 2;
    cfg.zero_point = 0;
    int32_t qw[4] = {0, 0, 0, 0}; /* [K=4][N/8=1] */
    float scales[4] = {1.0f, 1.0f, 1.0f, 1.0f}; /* [K/G=2][N=2] */
    /* k=0：自然列 n=0 放 1，n=1 放 2 */
    qw[0] = (int32_t)((1u << (4 * 0)) | (2u << (4 * 4)));
    float out[8];
    int rc = sgl_qweight_dequant(&cfg, qw, 4, 1, NULL, 0, 0, scales, 2, 2, NULL, 0, 2, 4, out);
    check(rc == 0, "awq 位布局：dequant 返回成功");
    /* out 是 [N, K]：out[n*K + k] */
    check(fabsf(out[0 * 4 + 0] - 1.0f) < 1e-6f, "awq: 自然列0 → 槽位0");
    check(fabsf(out[1 * 4 + 0] - 2.0f) < 1e-6f, "awq: 自然列1 → 槽位4（不是槽位1）");
    check(fabsf(out[0 * 4 + 1]) < 1e-6f, "awq: k=1 该行为 0");
  }
  /* AWQ gemv（转置布局）：qweight [N/pack, K] */
  {
    sgl_qcfg cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.method = SGL_Q_AWQ;
    cfg.bits = 4;
    cfg.group_size = 2;
    cfg.transposed = 1;
    int32_t qw[4] = {0, 0, 0, 0}; /* [N/8=1][K=4] */
    float scales[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    qw[0] = (int32_t)(3u << (4 * 4)); /* 自然列 1 = 3，槽位 4 */
    float out[8];
    int rc = sgl_qweight_dequant(&cfg, qw, 1, 4, NULL, 0, 0, scales, 2, 2, NULL, 0, 2, 4, out);
    check(rc == 0, "awq/gemv 位布局：dequant 返回成功");
    check(fabsf(out[1 * 4 + 0] - 3.0f) < 1e-6f, "awq/gemv: 转置布局读取正确");
  }
  /* GPTQ：qweight 沿 K 打包，[K/pack][N]；k=3 应在第 3 个 4bit 槽位 */
  {
    sgl_qcfg cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.method = SGL_Q_GPTQ;
    cfg.bits = 4;
    cfg.group_size = 4;
    cfg.zero_point = 1;
    int32_t qw[2] = {0, 0}; /* [K/8=1][N=2]：一个 int32 装一列的 8 行 */
    qw[0] = (int32_t)(5u << 12); /* 列0 的 k=3 = 5 */
    qw[1] = (int32_t)(7u << 28); /* 列1 的 k=7 = 7 */
    /* 两组都存 zero=8：这样期望值就是 (q-8)*scale，同时验证槽位和减 zero */
    int32_t qz[2] = {(int32_t)((8u << 0) | (8u << 4)), (int32_t)((8u << 0) | (8u << 4))};
    float scales[4] = {0.5f, 0.5f, 0.5f, 0.5f};        /* [K/G=2][N=2] */
    float out[16];
    int rc = sgl_qweight_dequant(&cfg, qw, 1, 2, qz, 2, 1, scales, 2, 2, NULL, 0, 2, 8, out);
    check(rc == 0, "gptq 位布局：dequant 返回成功");
    check(fabsf(out[0 * 8 + 3] - (5 - 8) * 0.5f) < 1e-6f, "gptq: k=3 → 槽位3，减 zero 8");
    check(fabsf(out[1 * 8 + 7] - (7 - 8) * 0.5f) < 1e-6f, "gptq: 列1 的 k=7 读对");
    check(fabsf(out[0 * 8 + 0] - (0 - 8) * 0.5f) < 1e-6f, "gptq: k=0 槽位空 → 只剩 -zero*scale");
  }
  /* g_idx 非连续分组：k 的 group 由 g_idx 给 */
  {
    sgl_qcfg cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.method = SGL_Q_GPTQ;
    cfg.bits = 4;
    cfg.group_size = 2;
    cfg.zero_point = 1;
    cfg.desc_act = 1;
    int32_t qw[1] = {(int32_t)((2u << 0) | (2u << 4))}; /* [1][N=1]，k=0 与 k=1 都是 2 */
    int32_t qz[2] = {0, 0};
    int32_t g_idx[8] = {1, 0, 1, 0, 1, 0, 1, 0}; /* k=0 属 group1 */
    float scales[2] = {1.0f, 4.0f};              /* group0=1, group1=4 */
    float out[8];
    int rc = sgl_qweight_dequant(&cfg, qw, 1, 1, qz, 2, 1, scales, 2, 1, g_idx, 8, 1, 8, out);
    check(rc == 0, "gptq+g_idx：dequant 返回成功");
    check(fabsf(out[0 * 8 + 0] - 2 * 4.0f) < 1e-6f, "g_idx: k=0 → group1（scale 4）");
    check(fabsf(out[0 * 8 + 1] - 2 * 1.0f) < 1e-6f, "g_idx: k=1 → group0（scale 1）");
  }
  /* 出错路径：AWQ 非 4bit / 形状不符 必须报错而不是算出垃圾 */
  {
    sgl_qcfg cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.method = SGL_Q_AWQ;
    cfg.bits = 8;
    cfg.group_size = 2;
    int32_t qw[4] = {0, 0, 0, 0};
    float scales[4] = {1, 1, 1, 1};
    float out[8];
    check(sgl_qweight_dequant(&cfg, qw, 4, 1, NULL, 0, 0, scales, 2, 2, NULL, 0, 2, 4, out) != 0,
          "awq 8bit 被拒绝");
    cfg.bits = 4;
    check(sgl_qweight_dequant(&cfg, qw, 4, 1, NULL, 0, 0, scales, 2, 3, NULL, 0, 2, 4, out) != 0,
          "scales 形状不符被拒绝");
  }
  printf("layout unit tests done\n");
}

/* ---------------- 2) 真实检查点切片 ---------------- */

static int test_real_dir(const char *dir) {
  char mp[1500], bp[1500];
  snprintf(mp, sizeof(mp), "%s/refmeta", dir);
  snprintf(bp, sizeof(bp), "%s/refbin", dir);
  FILE *fm = NULL;
  if (fopen_s(&fm, mp, "rb") != 0 || !fm) {
    printf("SKIP: %s/refmeta 缺失\n", dir);
    return 0;
  }
  FILE *fb = NULL;
  if (fopen_s(&fb, bp, "rb") != 0 || !fb) {
    fclose(fm);
    printf("SKIP: %s/refbin 缺失\n", dir);
    return 0;
  }
  fseek(fb, 0, SEEK_END);
  long sz = ftell(fb);
  fseek(fb, 0, SEEK_SET);
  unsigned char *blob = (unsigned char *)sgl_xmalloc((size_t)sz);
  if (fread(blob, 1, (size_t)sz, fb) != (size_t)sz) {
    printf("FAIL: %s refbin 读不全\n", dir);
    fclose(fm);
    fclose(fb);
    sgl_xfree(blob);
    fails++;
    return 0;
  }
  fclose(fb);

  sgl_st_ctx *st = sgl_st_open_dir(dir);
  if (!st) {
    printf("FAIL: %s 打不开\n", dir);
    fclose(fm);
    sgl_xfree(blob);
    fails++;
    return 0;
  }
  char line[1024];
  int n = 0, bad = 0;
  double worst = 0.0;
  char worst_name[256] = {0};
  while (fgets(line, sizeof(line), fm)) {
    char name[256];
    int ndim = 0;
    long long d0 = 0, d1 = 0, off = 0;
    if (sscanf(line, "%255[^\t]\t%d\t%lld,%lld\t%lld", name, &ndim, &d0, &d1, &off) != 5) continue;
    int64_t numel = d0 * d1;
    if (off < 0 || off + numel * 4 > sz) {
      printf("FAIL: %s 参考范围越界\n", name);
      bad++;
      continue;
    }
    sgl_tensor t = sgl_st_get(st, name, 2, (int64_t[]){d0, d1});
    if (sgl_tensor_is_null(t)) {
      printf("FAIL: %s 取不到\n", name);
      bad++;
      continue;
    }
    const float *ref = (const float *)(blob + off);
    double num = 0.0, den = 0.0, mx = 0.0;
    for (int64_t i = 0; i < numel; i++) {
      double d = (double)t.data[i] - (double)ref[i];
      num += d * d;
      den += (double)ref[i] * (double)ref[i];
      if (fabs(d) > mx) mx = fabs(d);
    }
    double rr = den > 0 ? sqrt(num / den) : (num > 0 ? 1e9 : 0.0);
    printf("  %-46s rel_rms=%.3e maxabs=%.3e\n", name, rr, mx);
    if (rr > worst) {
      worst = rr;
      snprintf(worst_name, sizeof(worst_name), "%s", name);
    }
    /* 官方实现用 float32 乘加，这里也是 f32：只应差几个 ulp */
    if (rr > 1e-5) {
      printf("FAIL: %s 与官方参考实现不一致（rel_rms=%.3e）\n", name, rr);
      bad++;
    }
    n++;
  }
  fclose(fm);
  sgl_xfree(blob);
  sgl_st_close(st);
  printf("%s: 比了 %d 个真实检查点张量，worst rel_rms=%.3e (%s)\n", dir, n, worst, worst_name);
  check(n >= 2, "真实切片至少比 2 个张量");
  return n;
}

/* ---------------- 3) 本地夹具 vs 未量化原始权重 ---------------- */

/* 4bit 量化本身有误差，这里只要求"在量化误差量级内"：
 * 布局错了会直接到 0.5~1.0 的量级（比如 zeros 少减一次、或列顺序错了）。*/
static int test_fixture_vs_original(const char *fix, const char *orig, const char *label) {
  char probe[1500];
  snprintf(probe, sizeof(probe), "%s/model.safetensors", fix);
  if (!file_exists(probe)) {
    printf("SKIP: %s 不存在\n", probe);
    return 0;
  }
  sgl_st_ctx *a = sgl_st_open_dir(fix);
  sgl_st_ctx *b = sgl_st_open_dir(orig);
  if (!a || !b) {
    printf("FAIL: %s 或 %s 打不开\n", fix, orig);
    fails++;
    if (a) sgl_st_close(a);
    if (b) sgl_st_close(b);
    return 0;
  }
  static const char *names[] = {
      "model.layers.0.self_attn.q_proj.weight",
      "model.layers.0.self_attn.k_proj.weight",
      "model.layers.0.self_attn.o_proj.weight",
      "model.layers.0.mlp.gate_proj.weight",
      "model.layers.0.mlp.down_proj.weight",
      "model.layers.1.mlp.down_proj.weight",
  };
  double worst = 0.0;
  int n = 0;
  for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
    sgl_tensor ta = sgl_st_get(a, names[i], -1, NULL);
    sgl_tensor tb = sgl_st_get(b, names[i], -1, NULL);
    if (sgl_tensor_is_null(ta) || sgl_tensor_is_null(tb)) {
      printf("FAIL: %s %s 取不到\n", label, names[i]);
      fails++;
      continue;
    }
    int64_t numel = sgl_tensor_numel(&tb);
    if (sgl_tensor_numel(&ta) != numel) {
      printf("FAIL: %s %s 元素数不同\n", label, names[i]);
      fails++;
      continue;
    }
    double num = 0.0, den = 0.0;
    for (int64_t k = 0; k < numel; k++) {
      double d = (double)ta.data[k] - (double)tb.data[k];
      num += d * d;
      den += (double)tb.data[k] * (double)tb.data[k];
    }
    double rr = sqrt(num / (den > 0 ? den : 1.0));
    if (rr > worst) worst = rr;
    if (rr > 0.20) {
      printf("FAIL: %s %s rel_rms=%.4f 超出 4bit 量化误差量级\n", label, names[i], rr);
      fails++;
    }
    n++;
  }
  printf("%s: %d 个张量 vs 未量化原始权重，worst rel_rms=%.4f（4bit group-8 的正常量级）\n", label,
         n, worst);
  check(n >= 4, "夹具至少比 4 个张量");
  sgl_st_close(a);
  sgl_st_close(b);
  return n;
}

/* ---------------- 4) config 解析 ---------------- */

static void test_cfg_parse(void) {
  sgl_qcfg q;
  const char *awq = "{\"quantization_config\":{\"bits\":4,\"group_size\":128,"
                    "\"quant_method\":\"awq\",\"version\":\"gemm\",\"zero_point\":true}}";
  check(sgl_qcfg_from_json(awq, strlen(awq), &q) == 1 && q.method == SGL_Q_AWQ &&
            q.bits == 4 && q.group_size == 128 && q.zero_point == 1 && !q.transposed,
        "解析 AWQ config（gemm）");
  const char *gemv = "{\"quantization_config\":{\"bits\":4,\"group_size\":128,"
                     "\"quant_method\":\"awq\",\"version\":\"gemv\",\"zero_point\":false}}";
  check(sgl_qcfg_from_json(gemv, strlen(gemv), &q) == 1 && q.transposed && !q.zero_point,
        "解析 AWQ config（gemv + 无 zero_point）");
  const char *gptq = "{\"quantization_config\":{\"bits\":4,\"group_size\":128,"
                     "\"quant_method\":\"gptq\",\"desc_act\":true,\"sym\":true,"
                     "\"format\":\"gptq\"}}";
  check(sgl_qcfg_from_json(gptq, strlen(gptq), &q) == 1 && q.method == SGL_Q_GPTQ && q.desc_act,
        "解析 GPTQ config（desc_act）");
  const char *v2 = "{\"quantization_config\":{\"bits\":4,\"group_size\":128,"
                   "\"quant_method\":\"gptq\",\"format\":\"gptq_v2\"}}";
  check(sgl_qcfg_from_json(v2, strlen(v2), &q) == 1 && q.method == SGL_Q_GPTQ && q.v2,
        "解析 gptq_v2（置 v2 标志）");
  const char *bad = "{\"quantization_config\":{\"bits\":4,\"group_size\":128,"
                    "\"quant_method\":\"gptq\",\"format\":\"gptq_marlin_v9\"}}";
  check(sgl_qcfg_from_json(bad, strlen(bad), &q) == 0,
        "不认识的 GPTQ format 被明确拒绝（而不是静默解错）");
  /* marlin / gptq_marlin / awq version=marlin：认下来，置 layout=marlin */
  const char *fm = "{\"quantization_config\":{\"bits\":4,\"group_size\":128,"
                   "\"quant_method\":\"gptq\",\"format\":\"gptq_marlin\"}}";
  check(sgl_qcfg_from_json(fm, strlen(fm), &q) == 1 && q.method == SGL_Q_GPTQ &&
            q.layout == SGL_QLAYOUT_MARLIN,
        "gptq format=gptq_marlin → layout=marlin");
  const char *am = "{\"quantization_config\":{\"bits\":4,\"group_size\":128,"
                   "\"quant_method\":\"awq\",\"version\":\"marlin\",\"zero_point\":false}}";
  check(sgl_qcfg_from_json(am, strlen(am), &q) == 1 && q.method == SGL_Q_AWQ &&
            q.layout == SGL_QLAYOUT_MARLIN,
        "awq version=marlin → layout=marlin");
  const char *plain = "{\"hidden_size\":64}";
  check(sgl_qcfg_from_json(plain, strlen(plain), &q) == 0, "无量化信息的 config 返回 0");
}

int main(void) {
  test_layout();
  test_cfg_parse();
  const char *mu = getenv("SGL_TEST_MARLIN_UNIT");
  if (mu && file_exists(mu)) {
    test_marlin_unit(mu);
  } else {
    printf("SKIP: SGL_TEST_MARLIN_UNIT 未设置（marlin 定值用例，python tests/marlin_layout.py "
           "--emit-unit tests/_marlin_unit.json）\n");
  }

  const char *real = getenv("SGL_TEST_QWEIGHT_REAL");
  if (real && *real) {
    char d[1500];
    snprintf(d, sizeof(d), "%s/awq", real);
    test_real_dir(d);
    snprintf(d, sizeof(d), "%s/gptq", real);
    test_real_dir(d);
  } else {
    printf("SKIP: SGL_TEST_QWEIGHT_REAL 未设置（真实检查点切片）\n");
  }

  const char *fx = getenv("SGL_TEST_QW_FIXTURES");
  const char *og = getenv("SGL_TEST_QW_ORIGINAL");
  if (fx && og) {
    char d[1500];
    snprintf(d, sizeof(d), "%s/mistral-rand-awq", fx);
    test_fixture_vs_original(d, og, "AWQ 夹具");
    snprintf(d, sizeof(d), "%s/mistral-rand-gptq", fx);
    test_fixture_vs_original(d, og, "GPTQ 夹具");
    snprintf(d, sizeof(d), "%s/mistral-rand-gptq-act", fx);
    test_fixture_vs_original(d, og, "GPTQ act-order 夹具");
  } else {
    printf("SKIP: SGL_TEST_QW_FIXTURES/ORIGINAL 未设置（本地夹具）\n");
  }

  if (fails) {
    printf("test_qweight: %d FAILED\n", fails);
    return 1;
  }
  printf("test_qweight: all OK\n");
  return 0;
}