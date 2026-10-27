/* sgl qweight 实现：AWQ / GPTQ 的位解包与反量化
 *
 * 位布局来源（都按官方落盘代码核对，不是凭记忆）：
 *   AWQ  casper-hansen/AutoAWQ  awq/modules/linear/gemm.py::from_linear（打包）
 *        awq/utils/packing_utils.py::unpack_awq / reverse_awq_order / dequantize_gemm
 *   GPTQ ModelCloud/GPTQModel    nn_modules/qlinear/__init__.py::_dequantize_from_codes
 *        （dequant 不做行置换）+ 落盘形状由真实 checkpoint 头部核对
 */
#include "sgl/qweight.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sgl/json.h"
#include "sgl/log.h"

int sgl_q_pack(int bits) { return bits > 0 ? 32 / bits : 0; }

/* AWQ 的 8 元重排（AWQ_REVERSE_ORDER）：自然列 n 的 4bit 槽位是 rev[n & 7] */
static const int AWQ_REV[8] = {0, 4, 1, 5, 2, 6, 3, 7};

static int get_code(int32_t word, int slot, int bits, unsigned mask) {
  return (int)(((unsigned)word >> (bits * slot)) & mask);
}

int sgl_qweight_dequant(const sgl_qcfg *cfg, const int32_t *qweight, int64_t qw_rows,
                        int64_t qw_cols, const int32_t *qzeros, int64_t qz_rows,
                        int64_t qz_cols, const float *scales, int64_t sc_rows, int64_t sc_cols,
                        const int32_t *g_idx, int64_t g_n, int64_t N, int64_t K, float *out) {
  if (!cfg || !qweight || !scales || !out || N <= 0 || K <= 0) return -1;
  const int bits = cfg->bits ? cfg->bits : 4;
  if (bits != 4 && bits != 8) {
    SGL_ERROR("quant: %d bit 不支持（只支持 4/8）", bits);
    return -1;
  }
  if (cfg->method != SGL_Q_AWQ && cfg->method != SGL_Q_GPTQ) {
    SGL_ERROR("quant: 未知的量化方法 %d", cfg->method);
    return -1;
  }
  if (cfg->method == SGL_Q_AWQ && bits != 4) {
    /* 官方 from_linear 对非 4bit 直接 NotImplementedError */
    SGL_ERROR("quant: AWQ 只支持 4 bit（config 里是 %d）", bits);
    return -1;
  }
  const int pack = sgl_q_pack(bits);
  const unsigned mask = (1u << bits) - 1u;
  const int64_t G = cfg->group_size > 0 ? cfg->group_size : K;
  const int has_z = qzeros != NULL;

  if (sc_cols != N) {
    SGL_ERROR("quant: scales 形状 [%lld,%lld]，期望 [%lld,%lld]", (long long)sc_rows,
              (long long)sc_cols, (long long)((K + G - 1) / G), (long long)N);
    return -1;
  }
  if (has_z && qz_cols * pack < N) {
    SGL_ERROR("quant: qzeros 打包宽度 %lld*%d < N=%lld", (long long)qz_cols, pack,
              (long long)N);
    return -1;
  }

  if (cfg->method == SGL_Q_AWQ) {
    /* ---- AWQ：沿输出维打包，含 8 元重排 ---- */
    if (cfg->transposed) {
      if (qw_rows * pack < N || qw_cols != K) {
        SGL_ERROR("quant(awq/gemv): qweight [%lld,%lld] 与 N=%lld K=%lld 不符",
                  (long long)qw_rows, (long long)qw_cols, (long long)N, (long long)K);
        return -1;
      }
    } else if (qw_rows != K || qw_cols * pack < N) {
      SGL_ERROR("quant(awq/gemm): qweight [%lld,%lld] 与 N=%lld K=%lld 不符",
                (long long)qw_rows, (long long)qw_cols, (long long)N, (long long)K);
      return -1;
    }
    for (int64_t k = 0; k < K; k++) {
      const int64_t g = k / G;
      if (g >= sc_rows) {
        SGL_ERROR("quant(awq): 行 %lld 的 group %lld 超出 scales 行数 %lld", (long long)k,
                  (long long)g, (long long)sc_rows);
        return -1;
      }
      const float *srow = scales + g * N;
      const int32_t *zrow = NULL;
      if (has_z) {
        if (g >= qz_rows) {
          SGL_ERROR("quant(awq): group %lld 超出 qzeros 行数 %lld", (long long)g,
                    (long long)qz_rows);
          return -1;
        }
        zrow = qzeros + g * qz_cols;
      }
      for (int64_t n = 0; n < N; n++) {
        const int slot = AWQ_REV[n & 7];
        const int32_t qw = cfg->transposed ? qweight[(n >> 3) * qw_cols + k]
                                          : qweight[k * qw_cols + (n >> 3)];
        int q = get_code(qw, slot, bits, mask);
        if (zrow) {
          int z = get_code(zrow[n >> 3], slot, bits, mask);
          q -= z;
        }
        out[n * K + k] = (float)q * srow[n];
      }
    }
    return 0;
  }

  /* ---- GPTQ：沿输入维打包；group 由 g_idx（act-order）或 k/G 决定 ---- */
  if (qw_cols != N || qw_rows * pack < K) {
    SGL_ERROR("quant(gptq): qweight [%lld,%lld] 与 N=%lld K=%lld 不符", (long long)qw_rows,
              (long long)qw_cols, (long long)N, (long long)K);
    return -1;
  }
  if (g_idx && g_n < K) {
    SGL_ERROR("quant(gptq): g_idx 长度 %lld < K=%lld", (long long)g_n, (long long)K);
    return -1;
  }
  for (int64_t k = 0; k < K; k++) {
    int64_t g = g_idx ? (int64_t)g_idx[k] : k / G;
    if (g < 0 || g >= sc_rows) {
      SGL_ERROR("quant(gptq): 行 %lld 的 group %lld 超出 scales 行数 %lld", (long long)k,
                (long long)g, (long long)sc_rows);
      return -1;
    }
    const float *srow = scales + g * N;
    const int32_t *zrow = has_z ? qzeros + g * qz_cols : NULL;
    if (has_z && g >= qz_rows) {
      SGL_ERROR("quant(gptq): group %lld 超出 qzeros 行数 %lld", (long long)g,
                (long long)qz_rows);
      return -1;
    }
    const int slot = (int)(k % pack);
    const int32_t *qwrow = qweight + (k / pack) * qw_cols;
    for (int64_t n = 0; n < N; n++) {
      int q = get_code(qwrow[n], slot, bits, mask);
      if (zrow) q -= get_code(zrow[n >> 3], (int)(n % pack), bits, mask);
      out[n * K + k] = (float)q * srow[n];
    }
  }
  return 0;
}

/* ---------- config.json 的 quantization_config ---------- */

/* 在一个对象里按名字取子对象 */
static const sgl_json *obj_get(const sgl_json *root, const char *key) {
  return root ? sgl_json_get(root, key) : NULL;
}

int sgl_qcfg_from_json(const char *json, size_t len, sgl_qcfg *cfg) {
  if (!cfg) return 0;
  memset(cfg, 0, sizeof(*cfg));
  cfg->method = SGL_Q_NONE;
  if (!json || len == 0) return 0;
  sgl_json *root = sgl_json_parse(json, len);
  if (!root) return 0;
  const sgl_json *qc = obj_get(root, "quantization_config");
  if (!qc || qc->type != SGL_JSON_OBJECT) {
    sgl_json_free(root);
    return 0;
  }
  const char *method = sgl_json_get_str(qc, "quant_method", "");
  cfg->bits = sgl_json_get_int(qc, "bits", 4);
  cfg->group_size = sgl_json_get_int(qc, "group_size", 128);
  if (!strcmp(method, "awq")) {
    cfg->method = SGL_Q_AWQ;
    cfg->zero_point = sgl_json_get_bool(qc, "zero_point", 1);
    cfg->bits = sgl_json_get_int(qc, "bits", 4);
    const char *ver = sgl_json_get_str(qc, "version", "gemm");
    cfg->transposed = !strcmp(ver, "gemv");
    if (!strcmp(ver, "marlin") || !strcmp(ver, "gemv_fast"))
      SGL_INFO("quant: AWQ version=%s 的落盘布局不通用，按 gemm/gemv 形状自动判定", ver);
  } else if (!strcmp(method, "gptq")) {
    cfg->method = SGL_Q_GPTQ;
    cfg->zero_point = 1;
    cfg->desc_act = sgl_json_get_bool(qc, "desc_act", 0);
    const char *fmt = sgl_json_get_str(qc, "format", "gptq");
    if (fmt && fmt[0] && strcmp(fmt, "gptq")) {
      /* gptq_v2 的 zeros 带 +1 偏移，本实现按 v1 语义（HF 主流）*/
      SGL_ERROR("quant: GPTQ format=%s 不支持（只支持 HF 主流的 v1 落盘）", fmt);
      cfg->method = SGL_Q_NONE;
      sgl_json_free(root);
      return 0;
    }
  } else {
    SGL_INFO("quant: quantization_config.quant_method=%s 不由本实现处理", method);
    sgl_json_free(root);
    return 0;
  }
  sgl_json_free(root);
  SGL_INFO("quant: %s bits=%d group=%d zero_point=%d desc_act=%d%s",
           cfg->method == SGL_Q_AWQ ? "AWQ" : "GPTQ", cfg->bits, cfg->group_size,
           cfg->zero_point, cfg->desc_act, cfg->transposed ? " (gemv layout)" : "");
  return 1;
}