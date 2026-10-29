/* sgl qweight 实现：AWQ / GPTQ 的位解包与反量化
 *
 * 位布局来源（都按官方落盘代码核对，不是凭记忆）：
 *   AWQ  casper-hansen/AutoAWQ  awq/modules/linear/gemm.py::from_linear（打包）
 *        awq/utils/packing_utils.py::unpack_awq / reverse_awq_order / dequantize_gemm
 *   GPTQ ModelCloud/GPTQModel    nn_modules/qlinear/__init__.py::_dequantize_from_codes
 *        （dequant 不做行置换）+ 落盘形状由真实 checkpoint 头部核对
 */
#include "sgl/qweight.h"

#include <math.h>
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
      if (zrow) q -= get_code(zrow[n >> 3], (int)(n % pack), bits, mask) + (cfg->v2 ? 1 : 0);
      out[n * K + k] = (float)q * srow[n];
    }
  }
  return 0;
}

/* ---------- marlin 布局（AWQ / GPTQ 的 marlin ABI，layout=1） ----------
 *
 * 这是 vLLM 的 awq_marlin / gptq_marlin 在 process_weights_after_loading 里把标准
 * 落盘权重重排出来的内存形态，kernel 直接吃它。布局不是“谁随便定的”：
 *   - Marlin 作者的官方 Python 打包（IST-DASLab/marlin `Layer.pack`）
 *   - vLLM 的 `csrc/quantization/marlin/awq_marlin_repack.cu`（CUDA kernel）
 * 这两份独立实现产出同一份 B；`tests/marlin_layout.py` 把两边和下面的闭式索引都对过了
 * （4 种形状逐字节相同）。所以这里的下标不是猜的。
 *
 * 一个 16(k) x 64(n) 的 tile：4bit 时 = 128 个 int32、每 word 8 个 nibble；
 * 8bit 时 = 256 个 int32、每 word 4 个 byte（每 tile 的 k×n 数不变，都是 1024）。
 *   word p 的第 j 个 nibble/byte 对应的自然坐标：
 *     4bit: k_off = 2*((p>>2)&3) + (j&1 ? 8 : 0) + (j>=4 ? 1 : 0)
 *           n_off = 16*(p&3) + (p>>4) + (j&2 ? 8 : 0)                    j=0..7
 *     8bit: k_off = 2*((p>>3)&3) + [0,8,1,9][j]
 *           n_off = 16*((p>>1)&3) + (p>>5) + (p&1 ? 8 : 0)               j=0..3
 * 8bit 那组：每个线程写**两个** word（out[th*8 + warp*2] 与 +1，第二个对应 n+8），
 * 所以 word 下标里的 p&1 就是“前 4 列 / 后 4 列”、(p>>1)&3 是 warp、(p>>3)&3 是 tc_row/2。
 * 两份互相独立的 vLLM kernel（`awq_marlin_repack.cu` 的 8bit 分支与 `gptq_marlin_repack.cu`
 * 的 8bit 分支，输入的自然布局和 shared memory 组织都不同）产出同一个 B，
 * 我们的闭式索引与它们逐字节相同（`tests/marlin_layout.py`）；Marlin 作者的官方 Python
 * 只做 4bit，所以 8bit 没有第三份参考。
 * 码是无符号的（marlin 打包时 +2^(bits-1)）；没有 zero point 时值 = (q - 2^(bits-1)) * scale。*/
#define MARLIN_TILE_N 64

/* 自然列 n 在 marlin 行里的位置：
 *   sidx = 该列在 marlin 置换后的 scales/qzeros 行里的下标（不做 8 元 interleave）
 *   zcol/znib = 该列在 marlin 打包后的 qzeros（[K/G, N/pack]）里的 word 列与 nibble
 * 正向链（vLLM marlin_utils.py）：s' = s.reshape(-1,64)[:, scale_perm]
 *   → 零点再过一遂 interleave（4bit [0,2,4,6,1,3,5,7]、8bit [0,2,1,3]）→ pack_cols。*/
static void marlin_col_map(int64_t n, int pack, int64_t *sidx, int64_t *zcol, int *znib) {
  const int64_t blk = (n / MARLIN_TILE_N) * MARLIN_TILE_N;
  const int64_t p_local = 8 * (n & 7) + ((n % MARLIN_TILE_N) >> 3);
  const int64_t f = blk + p_local;
  if (sidx) *sidx = f;
  int pos, gsz;
  if (pack == 8) { /* 4bit：interleave 8 元，逆 = (j>>1) + 4*(j&1) */
    const int j = (int)(f & 7);
    pos = (j >> 1) + 4 * (j & 1);
    gsz = 8;
  } else { /* 8bit：interleave 4 元 [0,2,1,3]，自逆 */
    const int j = (int)(f & 3);
    pos = ((j & 1) << 1) | (j >> 1);
    gsz = 4;
  }
  const int64_t f2 = gsz * (f / gsz) + pos;
  if (zcol) *zcol = f2 / pack;
  if (znib) *znib = (int)(f2 % pack);
}

int sgl_qweight_dequant_marlin(const sgl_qcfg *cfg, const int32_t *qweight, int64_t qw_rows,
                               int64_t qw_cols, const int32_t *qzeros, int64_t qz_rows,
                               int64_t qz_cols, const float *scales, int64_t sc_rows,
                               int64_t sc_cols, int64_t N, int64_t K, float *out) {
  if (!cfg || !qweight || !scales || !out || N <= 0 || K <= 0) return -1;
  const int bits = cfg->bits ? cfg->bits : 4;
  if (bits != 4 && bits != 8) {
    SGL_ERROR("quant(marlin): 只做 4/8bit（config 里是 %d）", bits);
    return -1;
  }
  const int pack = 32 / bits;            /* 4bit:8，8bit:4 */
  const int per_word = pack;             /* 每 word 的码数 */
  const unsigned mask = (1u << bits) - 1;
  const int qoff = 1 << (bits - 1);      /* 无零点时的无符号码偏移 */
  if (K % 16 != 0 || N % MARLIN_TILE_N != 0) {
    SGL_ERROR("quant(marlin): K=%lld 要能被 16 整除、N=%lld 要能被 64 整除", (long long)K,
              (long long)N);
    return -1;
  }
  if (qw_rows != K / 16 || qw_cols * pack < N * 16) {
    SGL_ERROR("quant(marlin): qweight [%lld,%lld] 与 N=%lld K=%lld（期望 [%lld,%lld]）不符",
              (long long)qw_rows, (long long)qw_cols, (long long)N, (long long)K,
              (long long)(K / 16), (long long)(N * 16 / pack));
    return -1;
  }
  if (sc_cols != N) {
    SGL_ERROR("quant(marlin): scales 宽度 %lld，期望 N=%lld", (long long)sc_cols, (long long)N);
    return -1;
  }
  const int has_z = qzeros != NULL;
  if (has_z && (qz_cols * pack < N || qz_rows < sc_rows)) {
    SGL_ERROR("quant(marlin): qzeros [%lld,%lld] 与 N=%lld / %lld 组不符", (long long)qz_rows,
              (long long)qz_cols, (long long)N, (long long)sc_rows);
    return -1;
  }
  const int64_t G = cfg->group_size > 0 ? cfg->group_size : K;
  const int64_t k_tiles = K / 16;
  const int64_t n_tiles = N / MARLIN_TILE_N;
  const int64_t tile_words = MARLIN_TILE_N * 16 / pack; /* 4bit:128，8bit:256 */
  static const int OFF8[4] = {0, 8, 1, 9};              /* 8bit 的 tc_offsets 顺序 */
  for (int64_t kt = 0; kt < k_tiles; kt++) {
    for (int64_t nt = 0; nt < n_tiles; nt++) {
      const int32_t *tile = qweight + kt * qw_cols + nt * tile_words;
      for (int p = 0; p < (int)tile_words; p++) {
        const unsigned word = (unsigned)tile[p];
        for (int j = 0; j < per_word; j++) {
          int k_off, n_off;
          if (bits == 4) {
            k_off = 2 * ((p >> 2) & 3) + ((j & 1) ? 8 : 0) + (j >= 4 ? 1 : 0);
            n_off = 16 * (p & 3) + (p >> 4) + ((j & 2) ? 8 : 0);
          } else {
            k_off = 2 * ((p >> 3) & 3) + OFF8[j];
            n_off = 16 * ((p >> 1) & 3) + (p >> 5) + ((p & 1) ? 8 : 0);
          }
          const int64_t k = kt * 16 + k_off;
          const int64_t n = nt * MARLIN_TILE_N + n_off;
          const int64_t g = k / G;
          if (g >= sc_rows) {
            SGL_ERROR("quant(marlin): 行 %lld 的 group %lld 超出 scales 行数 %lld", (long long)k,
                      (long long)g, (long long)sc_rows);
            return -1;
          }
          int64_t sidx = 0, zcol = 0;
          int znib = 0;
          marlin_col_map(n, pack, &sidx, has_z ? &zcol : NULL, has_z ? &znib : NULL);
          int q = (int)((word >> (bits * j)) & mask);
          if (has_z) {
            if (g >= qz_rows) {
              SGL_ERROR("quant(marlin): group %lld 超出 qzeros 行数 %lld", (long long)g,
                        (long long)qz_rows);
              return -1;
            }
            q -= (int)(((unsigned)qzeros[g * qz_cols + zcol] >> (bits * znib)) & mask);
          } else {
            q -= qoff;
          }
          out[n * K + k] = (float)q * scales[g * sc_cols + sidx];
        }
      }
    }
  }
  return 0;
}

/* ---------- compressed-tensors pack-quantized（W4A16 / W8A16） ----------
 * 与 AWQ/GPTQ 不同：没有零点张量，码是无符号的，减 1<<(bits-1) 得有符号值；
 * 打包是“密集位流”，且 4bit 时按字节小端：每字节低半字节 = 偶下标、高半字节 = 奇下标。
 * 依据：compressed-tensors
 *   src/compressed_tensors/compressors/pack_quantized/helpers.py::unpack_from_int32。*/
int sgl_qweight_dequant_ct(const sgl_qcfg *cfg, const int32_t *wpacked, int64_t rows,
                           int64_t cols, const float *scales, int64_t sc_rows, int64_t sc_cols,
                           int64_t N, int64_t K, float *out) {
  if (!cfg || !wpacked || !scales || !out || N <= 0 || K <= 0) return -1;
  const int bits = cfg->bits ? cfg->bits : 4;
  const int64_t G = cfg->group_size > 0 ? cfg->group_size : K;
  const int off = cfg->offset ? cfg->offset : (1 << (bits - 1));
  if (sc_cols != (K + G - 1) / G) {
    SGL_ERROR("quant(ct): scales [%lld,%lld]，期望 [%lld,%lld]", (long long)sc_rows,
              (long long)sc_cols, (long long)N, (long long)((K + G - 1) / G));
    return -1;
  }
  if (rows != N) {
    SGL_ERROR("quant(ct): weight_packed 行数 %lld != N=%lld", (long long)rows, (long long)N);
    return -1;
  }
  const int per_word = bits == 4 ? 8 : 4; /* 每 int32 装的元素数 */
  if (cols * per_word < K) {
    SGL_ERROR("quant(ct): weight_packed [%lld,%lld] 装不下 K=%lld", (long long)rows,
              (long long)cols, (long long)K);
    return -1;
  }
  for (int64_t n = 0; n < N; n++) {
    const int32_t *wrow = wpacked + n * cols;
    const float *srow = scales + n * sc_cols;
    if (bits == 4) {
      const uint8_t *bytes = (const uint8_t *)wrow;
      for (int64_t k = 0; k < K; k++) {
        int b = (int)(k >> 1);
        int code = (k & 1) ? (bytes[b] >> 4) : (bytes[b] & 0xF);
        out[n * K + k] = (float)(code - off) * srow[k / G];
      }
    } else {
      for (int64_t k = 0; k < K; k++) {
        int code = (int)(((uint32_t)wrow[k >> 2] >> (8 * (k & 3))) & 0xFFu);
        out[n * K + k] = (float)(code - off) * srow[k / G];
      }
    }
  }
  return 0;
}

/* e2m1 码书（OCP MXFP4）：0, .5, 1, 1.5, 2, 3, 4, 6，符号位最高 */
static float e2m1_val(int code) {
  static const float mag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  float v = mag[code & 7];
  return (code & 8) ? -v : v;
}

/* ---------- gpt-oss 的 mxfp4 ----------
 * weight 是 uint8 流，每字节两个 e2m1（低半字节在前）；
 * scales 是 e8m0：值 = 2^(x-127)（x=255 时为 NaN，本实现当 0 处理），每 32 个元素一个。*/
int sgl_qweight_dequant_mxfp4(const int32_t *cfg_unused, const uint8_t *wpacked, int64_t rows,
                              int64_t cols, const uint8_t *scales, int64_t sc_rows,
                              int64_t sc_cols, int64_t N, int64_t K, float *out) {
  (void)cfg_unused;
  if (!wpacked || !scales || !out || N <= 0 || K <= 0) return -1;
  if (rows != N) {
    SGL_ERROR("quant(mxfp4): weight 行数 %lld != N=%lld", (long long)rows, (long long)N);
    return -1;
  }
  if (cols * 2 < K) {
    SGL_ERROR("quant(mxfp4): weight [%lld,%lld] 装不下 K=%lld", (long long)rows, (long long)cols,
              (long long)K);
    return -1;
  }
  const int64_t nb = (K + 31) / 32;
  if (sc_cols != nb) {
    SGL_ERROR("quant(mxfp4): scales [%lld,%lld]，期望 [%lld,%lld]", (long long)sc_rows,
              (long long)sc_cols, (long long)N, (long long)nb);
    return -1;
  }
  for (int64_t n = 0; n < N; n++) {
    const uint8_t *wrow = wpacked + n * cols;
    const uint8_t *srow = scales + n * nb;
    for (int64_t k = 0; k < K; k++) {
      int b = (int)(k >> 1);
      int code = (k & 1) ? (wrow[b] >> 4) : (wrow[b] & 0xF);
      unsigned ex = srow[k >> 5];
      float sc = ex == 255 ? 0.f : ldexpf(1.0f, (int)ex - 127);
      out[n * K + k] = e2m1_val(code) * sc;
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
    /* version=marlin：有的导出工具会这么写。真实生态里**没有**把 AWQ 权重存成 marlin 布局的
     * 落盘格式（marlin 是 vLLM 加载期的内存布局，见 docs/GAP_vs_upstream.md A5），所以这里
     * 只当“优先按 marlin 读”的提示；_load 时形状对不上会回退到 gemm/gemv 判定。*/
    if (!strcmp(ver, "marlin")) {
      cfg->layout = SGL_QLAYOUT_MARLIN;
      SGL_INFO("quant: AWQ version=marlin → 先按 marlin 布局读，形状对不上再回退 gemm/gemv");
    } else if (!strcmp(ver, "gemv_fast")) {
      SGL_INFO("quant: AWQ version=%s 的落盘布局不通用，按 gemm/gemv 形状自动判定", ver);
    }
  } else if (!strcmp(method, "gptq")) {
    cfg->method = SGL_Q_GPTQ;
    cfg->zero_point = 1;
    cfg->desc_act = sgl_json_get_bool(qc, "desc_act", 0);
    const char *fmt = sgl_json_get_str(qc, "format", "gptq");
    /* format:
     *   "gptq"（缺省，同 v1）：qzeros 里存的就是零点本身
     *   "gptq_v2"：qzeros 存的是零点 - 1，解量化前要 +1
     *              （vLLM 的 gptq v2→v1 转换就是 zeros + 1，见
     *               srt/layers/quantization/gptq.py）*/
    if (fmt && !strcmp(fmt, "gptq_v2")) {
      cfg->v2 = 1;
    } else if (fmt && (!strcmp(fmt, "marlin") || !strcmp(fmt, "gptq_marlin"))) {
      /* 同 AWQ：marlin 不是落盘格式，当“优先按 marlin 读”的提示 */
      cfg->layout = SGL_QLAYOUT_MARLIN;
      SGL_INFO("quant: GPTQ format=%s → 先按 marlin 布局读，形状对不上再回退", fmt);
    } else if (fmt && fmt[0] && strcmp(fmt, "gptq")) {
      SGL_ERROR("quant: GPTQ format=%s 不支持（只认 gptq / gptq_v2 / marlin）", fmt);
      cfg->method = SGL_Q_NONE;
      sgl_json_free(root);
      return 0;
    }
  } else if (!strcmp(method, "compressed-tensors")) {
    /* llm-compressor / compressed-tensors 的 W4A16（format=pack-quantized）*/
    const char *fmt = sgl_json_get_str(qc, "format", "");
    const sgl_json *groups = sgl_json_get(qc, "config_groups");
    const sgl_json *g0 = NULL;
    if (groups && groups->type == SGL_JSON_OBJECT && groups->count > 0) g0 = groups->items[0];
    const sgl_json *wq = g0 ? sgl_json_get(g0, "weights") : NULL;
    int nb = 4, gsz = 128, sym = 1;
    if (wq && wq->type == SGL_JSON_OBJECT) {
      nb = sgl_json_get_int(wq, "num_bits", 4);
      gsz = sgl_json_get_int(wq, "group_size", 128);
      sym = sgl_json_get_bool(wq, "symmetric", 1);
      if (gsz <= 0) gsz = 128;
      if (!sym) {
        SGL_INFO("quant: compressed-tensors 非对称的 W%dA16 还没支持（只做 symmetric）", nb);
        sgl_json_free(root);
        return 0;
      }
    }
    if (fmt && strcmp(fmt, "pack-quantized")) {
      SGL_INFO("quant: compressed-tensors format=%s 不处理（只支持 pack-quantized）", fmt);
      sgl_json_free(root);
      return 0;
    }
    if (nb != 4 && nb != 8) {
      SGL_INFO("quant: compressed-tensors num_bits=%d 不支持", nb);
      sgl_json_free(root);
      return 0;
    }
    cfg->method = SGL_Q_CT;
    cfg->bits = nb;
    cfg->group_size = gsz;
    cfg->packed_dim = sgl_json_get_int(qc, "packed_dim", 1);
    cfg->offset = 1 << (nb - 1);
    cfg->zero_point = 0;
  } else if (!strcmp(method, "mxfp4")) {
    /* gpt-oss：e2m1 权重 + e8m0 块缩放（block=32）*/
    cfg->method = SGL_Q_MXFP4;
    cfg->bits = 4;
    cfg->group_size = 32;
    cfg->zero_point = 0;
  } else {
    SGL_INFO("quant: quantization_config.quant_method=%s 不由本实现处理", method);
    sgl_json_free(root);
    return 0;
  }
  sgl_json_free(root);
  SGL_INFO("quant: %s bits=%d group=%d zero_point=%d desc_act=%d%s%s%s",
           cfg->method == SGL_Q_AWQ ? "AWQ"
           : cfg->method == SGL_Q_GPTQ ? "GPTQ"
           : cfg->method == SGL_Q_CT ? "compressed-tensors"
                                       : "MXFP4",
           cfg->bits, cfg->group_size,
           cfg->zero_point, cfg->desc_act, cfg->transposed ? " (gemv layout)" : "",
           cfg->v2 ? " (gptq_v2)" : "", cfg->layout ? " (marlin layout)" : "");
  return 1;
}