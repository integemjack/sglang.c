/* sgl mlx_backend.cpp — MLX（Apple Silicon / Metal）后端 T1：llama 系前向
 *
 * 目标：与 src/llama.c 的 CPU 前向**逐算子同语义**，只把矩阵乘/归一化/注意力
 * 换成 MLX 的 Metal 实现。正确性由 tests/mlx_test.c 拿 CPU 当金标准对拍。
 *
 * T1 支持：llama / qwen2 / qwen3 / mistral / gemma 的 dense full-attention，
 *   fp16（默认）或 fp32 权重、GQA、q/k/v bias、qk-norm、partial rotary、
 *   每层独立 rope 逆频率表、YaRN attn_factor、sliding window、KV 段池、
 *   批量 decode、前缀缓存搬运。
 * T1 不支持（构造时直接拒绝，服务层据此回落 CPU）：
 *   hybrid/linear-attention（GDN，T3）、MLA（T3）、MoE（T2）、gpt2（T2）、
 *   q_gate（G13a）、MTP/投机（T3）。多模态 emb 入口已留（T2 接）。
 *
 * 实测钉死的 MLX 语义（tools/mlx_smoke/probe2.cpp、probe4.cpp）：
 *   1. mx::array(float*, Shape, Dtype) 是**拷贝**并转换 dtype，不是视图；
 *   2. mx::fast::rope 的位置轴是 **-2**，且它的 freqs 参数与本仓库 layer_invf
 *      语义不等价 → 这里用手写 rope（host 用 double 算 cos/sin），顺带把
 *      attn_factor / partial rotary / 每层不同 invf 一并覆盖；
 *   3. mx::slice_update 产生**新数组**（原数组不变）→ KV 池必须按层拆开，
 *      否则每层写入都会复制整个池（L 倍放大）；
 *   4. mx::array **没有默认构造函数** → 容器只能 reserve+push_back，
 *      可选权重用 std::optional<mx::array>；
 *   5. mx::matmul 在 4D 上按前两维批处理；mx::softmax(a, {-1}, true) 按最后一轴。
 */
#include "sgl/mlx.h"

#include "sgl/llama.h"

#include <mlx/fast.h>
#include <mlx/io.h>
#include <mlx/mlx.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

namespace mx = mlx::core;
using OArr = std::optional<mx::array>;

/* MLX 的 affine 量化参数（mlx-community 的 4-bit 检查点就是这个）。
 * 权重存 `<base>.weight`(U32，每 uint32 打包 32/bits 个值) + `<base>.scales`(F16)
 * + `<base>.biases`(F16)，后两者都是 [N, K/group]。
 * MXFP4：没有 biases，scales 是 U8（e8m0 指数），group=32。
 * bits/group/mode 从张量形状推出（QW 里带），不再硬编码：
 *   affine：K = scales列数 × 64，bits = 32 × weight列数 / K（4bit→8列，8bit→4列）
 *   mxfp4 ：scales dtype=U8 → group=32 bits=4 mode=mxfp4
 * （常量已并入 QW，此处仅存档说明；load_q3 的 infer() 现场推。）*/

/* 一个前向用的 2D 权重：稠密，或是量化三件套。
 * sc 为空 = 稠密（走普通 matmul）；非空 = 量化（走 quantized_matmul，
 * group/bits/mode 每份自带 —— 8-bit 检查点的 bits=8，mxfp4 的 mode 不同）。
 * bi 为空 + sc 是 U8 = mxfp4（无零点，scale 本身是 e8m0 指数）。*/
struct QW {
  mx::array w;
  OArr sc, bi;
  int group = 64;
  int bits = 4;
  int mxfp = 0; /* 1 = mxfp4（quantized_matmul 的 mode 参数传 "mxfp4"）*/
  /* 默认构造：mx::array 没有默认构造（MLX 的设计），所以拿 0 元素数组占位。
   * 只在“先声明、后交给 load2d 填”的地方用得到，不参与前向。*/
  QW() : w(mx::zeros(mx::Shape{0})), sc(), bi() {}
  QW(mx::array w_, OArr sc_ = OArr(), OArr bi_ = OArr(), int group_ = 64, int bits_ = 4,
     int mxfp_ = 0)
      : w(std::move(w_)), sc(std::move(sc_)), bi(std::move(bi_)), group(group_),
        bits(bits_), mxfp(mxfp_) {}
  bool quant() const { return sc.has_value(); }
  const char *mode() const { return mxfp ? "mxfp4" : "affine"; }
};

#define MLX_ERR(...)                     \
  do {                                   \
    fprintf(stderr, "[mlx] ");           \
    fprintf(stderr, __VA_ARGS__);        \
    fputc('\n', stderr);                 \
  } while (0)
#define MLX_LOG(...)                     \
  do {                                   \
    fprintf(stderr, "[mlx.dbg] ");       \
    fprintf(stderr, __VA_ARGS__);        \
    fputc('\n', stderr);                 \
  } while (0)

/* ============================ 实例 ============================ */

struct sgl_mlx_model {
  /* 几何（与 sgl_llama 同名同义）*/
  int L = 0, H = 0, KVH = 0, E = 0, I = 0, D = 0;
  int qdim = 0, kvdim = 0, V = 0, ctx = 0, rep = 1;
  float eps = 1e-5f;
  double pos_scale = 1.0, attn_factor = 1.0; /* CPU 侧也是 double，不降精度 */
  float gemma_scale = 0.f, emb_softcap = 0.f;
  int tie = 0, norm_p1 = 0, sliding_window = 0;
  mx::Dtype wdt = mx::float16; /* 权重/激活 dtype */
  mx::Dtype kdt = mx::float16; /* KV dtype */

  OArr wte, lm_head, final_norm;
  /* 量化 embed 表：非空时 wte 不解量化（7B 省 1GB 常驻），查表走 emb_lookup。*/
  QW wte_q;
  /* lm_head 的量化形式（量化检查点且 tie 时才有）。非空时优先用它算 logits，
   * 不必每步扫 272MB 的稠密 [E,V] 副本。*/
  QW lm_head_q;
  /* 每层权重：proj 用的 2D 权重按 wt_on() 存 [in, out]（默认）或 [out, in]。
   * q/k/v 与 gate/up 在加载时就拼成一份（qkv_w / gu_w），前向一次 matmul 再切：
   * 每层少 3 次 kernel、少 3 次权重读取，且 M1 上 kernel launch 正是 decode 的大头。
   * 量化检查点上每份是三件套沿 N 拼（weight/scales/biases 都拼，长度一致）。*/
  std::vector<mx::array> inp_norm, post_norm;
  std::vector<QW> qkv_w, o_w, gu_w, down_w;
  std::vector<OArr> qkv_b, q_norm, k_norm;
  /* rope：每层旋转维数 rot[l]（<=D）与逆频率表 invf[l][rot/2]（host 侧，double 算角度）*/
  std::vector<int> rot;
  std::vector<std::vector<double>> invf; /* layer_invf / rot_invf 在 CPU 侧就是 double */
  /* >0 = 这一层可以走 mx::fast::rope（标准 RoPE，只存 base）；0 = 回退手写。*/
  std::vector<double> rope_base;
  std::vector<int> layer_window;
  /* KV 池：按层拆开（见文件头第 3 条），每层 [ctx, KVH, D] */
  std::vector<mx::array> kc, vc;
};

/* ============================ 权重上传 ============================ */

/* 权重布局开关（SGL_MLX_WT，默认 1）：
 *   1 = 上传时就物化转置，存成 [in, out]，前向直接 matmul(x, w)；
 *   0 = 存 CPU 原布局 [out, in]，前向 matmul(x, transpose(w))。
 *
 * 为什么默认 1：MLX 的 transpose 是 view，matmul 碰到「转置且非连续」的 B 时
 * 要么先物化一份连续拷贝、要么用慢路径。lm_head 是 [V,E]=[151936,896] fp16
 * = 272 MB，这个开销是**每步**都要付的。实测差距见 docs/NOTES_mlx_t2.md：
 * 换布局前后 decode（带 lm_head）从 75 ms/步降到 ~17 ms/步。
 * 留 0 是为了能 A/B 复现（也方便对照 MLX 以后版本的行为变化）。*/
static int wt_on(void) {
  static int v = -1;
  if (v < 0) {
    const char *e = getenv("SGL_MLX_WT");
    v = (e && *e) ? atoi(e) : 1;
  }
  return v;
}

static mx::array up2(const sgl_tensor *t, mx::Dtype dt) {
  return mx::astype(mx::array(t->data, mx::Shape{(int)t->shape[0], (int)t->shape[1]},
                              mx::float32),
                    dt);
}
/* 2D 权重：按 wt_on() 决定是否上传时转置成 [in, out]（转置后立刻 eval，
 * 把拷贝做实，以后每次前向都省掉）。*/
static mx::array up2w(const sgl_tensor *t, mx::Dtype dt) {
  mx::array a = up2(t, dt);
  if (!wt_on()) return a;
  a = mx::contiguous(mx::transpose(a));
  a.eval();
  return a;
}
static mx::array up1(const sgl_tensor *t, mx::Dtype dt) {
  return mx::astype(mx::array(t->data, mx::Shape{(int)t->shape[0]}, mx::float32), dt);
}
/* norm_p1（Qwen3-Next 的 RMSNorm 用 (1+w)）：上传时就把 1 加进去，
 * 之后与普通 RMSNorm 走同一条 mx::fast::rms_norm。*/
static mx::array up1_norm(const sgl_tensor *t, mx::Dtype dt, int p1) {
  mx::array a = up1(t, dt);
  return p1 ? mx::add(a, mx::array(1.0f)) : a;
}
static OArr opt1(const sgl_tensor *t, mx::Dtype dt) {
  return (t && t->data) ? OArr(up1(t, dt)) : OArr();
}
static OArr opt1_norm(const sgl_tensor *t, mx::Dtype dt, int p1) {
  return (t && t->data) ? OArr(up1_norm(t, dt, p1)) : OArr();
}
/* 2D 权重（可选）：按 wt_on() 转置，给 proj 用 */
static OArr opt2w(const sgl_tensor *t, mx::Dtype dt) {
  return (t && t->data) ? OArr(up2w(t, dt)) : OArr();
}

struct QW;
struct sgl_mlx_model;

/* ============ 直接从 safetensors 读权重（量化检查点走这里）============
 *
 * 为什么不能复用 sgl_tensor：mlx-community 的 4-bit 检查点把稠密矩阵存成
 * `<base>.weight`(U32 打包) + `.scales`/`.biases`(F16)，sgl_st_get 只认浮点，
 * 会直接报 “dtype=U32 不支持”。而 4-bit 必须保住（解成 fp32 就丢了 3.4x 内存
 * 与全部的带宽优势），所以走设备侧直接建量化数组。
 *
 * U32/F16 的裸字节怎么变成 array —— 这里踩了一串坑，最后选了最笨但最稳的路：
 * **交给 MLX 自己的 load_safetensors 读**，不自己碰字节。试过且失败的（均已实测）：
 *   a) `mx::array(ptr, shape, dtype)`：独立 probe 里正常，但链进 sglc/mlx_q4_test
 *      时在 `array::init` 的 std::copy 里 SIGBUS（uint32 与 float32 两个分支都崩）；
 *   b) 先 `mx::zeros(sh,dt)` 再 `memcpy(a.data<T>(), ...)`：不崩，但 zeros 带
 *      primitive，MLX 的图优化把 `zeros → take` 常量折叠成 0 —— 实测 logits 全 0
 *      （top10 全 0.0000）；
 *   c) `allocator::malloc` + memcpy + `array(Buffer,...)`：SIGSEGV。
 * load_safetensors 返回的 map 里是 MLX 自己建的 array（无 primitive、dtype 正确），
 * 拿去直接用就行。支持单分片（model.safetensors）与多分片
 * （model.safetensors.index.json 的 weight_map → 逐个加载合并， mmap 不占内存）。*/
static std::unordered_map<std::string, mx::array> *g_mxw = nullptr;

/* 调试开关（定义在下面，mxw_open 先用）*/
static int dbg(void);

static void mxw_open(const char *dir) {
  if (g_mxw) return;
  char path[1024];
  std::vector<std::string> shards;
  snprintf(path, sizeof(path), "%s/model.safetensors", dir);
  if (FILE *f0 = fopen(path, "rb")) {
    fclose(f0);
    shards.push_back(path);
  } else {
    /* 多分片：读 index.json，把 weight_map 里引用到的分片名去重后逐个加载。
     * 格式固定（"tensor": "model-00001-of-00002.safetensors"），手写提取比拉
     * 一个 JSON 解析器进这个文件划算。*/
    char ipath[1024];
    snprintf(ipath, sizeof(ipath), "%s/model.safetensors.index.json", dir);
    FILE *f = fopen(ipath, "rb");
    if (!f) return; /* 没有 safetensors 可读：不启用，走 sgl_st */
    std::string js;
    char buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) js.append(buf, n);
    fclose(f);
    for (size_t i = 0; i < js.size();) {
      size_t a = js.find('"', i);
      if (a == std::string::npos) break;
      size_t b = js.find('"', a + 1);
      if (b == std::string::npos) break;
      std::string tok = js.substr(a + 1, b - a - 1);
      i = b + 1;
      if (tok.size() > 11 && tok.compare(tok.size() - 11, 11, ".safetensors") == 0 &&
          std::find(shards.begin(), shards.end(), tok) == shards.end())
        shards.push_back(dir + ("/" + tok));
    }
    if (shards.empty()) {
      MLX_ERR("index.json 里没提取到分片名");
      return;
    }
    if (dbg()) MLX_LOG("多分片：%zu 个", shards.size());
  }
  g_mxw = new std::unordered_map<std::string, mx::array>();
  for (auto &s : shards) {
    auto ld = mx::load_safetensors(s.c_str());
    for (auto &kv : ld.first) g_mxw->insert_or_assign(kv.first, kv.second);
    if (dbg()) MLX_LOG("分片 %s: %zu 张量", s.c_str(), ld.first.size());
  }
}

static void mxw_close(void) {
  delete g_mxw;
  g_mxw = nullptr;
}

/* 模型目录：sgl_llama 不存它，所以由服务层/sgl_mlx_from_dir 告知。
 * 有了它 from_llama 也能给量化检查点启 mxw（config-only 场景）。*/
static std::string g_dir;

void sgl_mlx_set_model_dir(const char *dir) {
  g_dir = dir ? dir : "";
  if (!g_dir.empty()) mxw_open(g_dir.c_str());
}

static mx::array *mxw_get(const std::string &name) {
  if (!g_mxw) return nullptr;
  auto it = g_mxw->find(name);
  return it == g_mxw->end() ? nullptr : &it->second;
}

/* 从原始字节建一个 array。**总是拷贝**：mmap 里张量的起始偏移不保证按 dtype 对齐，
 * 把未对齐指针交给 MLX 是 UB（它内部可能用向量 load）。单张量峰值拷贝，可接受。*/
template <typename T>
static mx::array arr_from(const void *p, const mx::Shape &sh, mx::Dtype dt) {
  /* 三个坑：
   * 1) 源指针（mmap 里的张量）不保证按 dtype 对齐 —— 不能按 T* 解引用，必须 memcpy；
   * 2) 传给 MLX 的缓冲要 64B 对齐（vector 只保证 16B）；
   * 3) **uint32 的构造分支在 sglc 里会 SIGBUS**（frame 落在 array::init，
   *    std::copy 到 data<uint32_t>()），而同一份代码在独立 probe5 里正常。
   *    还没查到根因（sglc 比 probe 多链了 CPU 库、多跑了前面的 CPU 加载）。
   *    绕法：先按 float32 构造（这条路径在所有 fp16 模型上从未出过问题），
   *    再 view 成目标 dtype —— 字节完全相同，只是重解释。
   *    float16 不绕（它走的是 float16 分支，fp16 权重一直在用）。*/
  const size_t n = sh.size();
  const size_t bytes = n * sizeof(T);
  void *buf = nullptr;
  if (posix_memalign(&buf, 64, (bytes + 63u) & ~(size_t)63u) != 0 || !buf) {
    MLX_ERR("arr_from: 分配 %zu 字节失败", bytes);
    return mx::zeros(sh, dt);
  }
  memcpy(buf, p, bytes);
  /* 走到这一步兜了好几个坑（都实测过）：
   * 1) `mx::array(ptr, shape, dtype)` → 在 sglc 里 SIGBUS（standalone probe 里正常）；
   * 2) `mx::zeros(sh,dt).eval()` 后 memcpy 进去 → 不崩，但数据被丢了：
   *    zeros 带 primitive，MLX 的图优化把 `zeros → take` 常量折叠成 0，
   *    实测 logits 全 0（top10 全 0.0000）。
   * 所以拿 allocator::Buffer 自己填，交给 array 接管（primitive 为空 = 纯数据数组，
   * 不会被折叠）：Buffer 由 array 持有，deleter 用默认的 allocator::free。*/
  mx::allocator::Buffer mb = mx::allocator::malloc(bytes);
  if (!mb.ptr()) {
    MLX_ERR("arr_from: allocator::malloc(%zu) 返回空", bytes);
    free(buf);
    return mx::zeros(sh, dt);
  }
  memcpy(mb.ptr(), buf, bytes);
  free(buf);
  return mx::array(mb, sh, dt);
}

/* 调试开关：SGL_MLX_DBG=1 打印权重加载细节（定位加载期崩溃用）。*/
static int dbg(void) {
  static int v = -1;
  if (v < 0) {
    const char *e = getenv("SGL_MLX_DBG");
    v = (e && *e) ? atoi(e) : 0;
  }
  return v;
}
/* SGL_MLX_TRACE=1：逐层打 x.sum（与 Python 对照定位数值分叉，强制同步）
 * SGL_MLX_TRACE=2：再打层内每个算子的 maxabs（定位 NaN 产生点）*/
static int trace_on(void) {
  static int v = -1;
  if (v < 0) {
    const char *e = getenv("SGL_MLX_TRACE");
    v = (e && *e) ? atoi(e) : 0;
  }
  return v;
}
static int trace2_on(void) { return trace_on() >= 2; }
/* SGL_MLX_EVAL 位掩码：控制哪里强制同步。背景：7B-4bit 上纯惰性图会在最后一层
 * 产生 NaN，而同样代码逐算子 eval 全程与官方逐位一致 —— 是图执行调度问题，
 * 不是数学问题。二分找出最小必要位置（1=KV写 2=分数 4=层尾）。
 * 强制同步会破坏跨层融合（见 NOTES_mlx_t2.md），用最小集合。*/
static int eval_mask(void) {
  static int v = -1;
  if (v < 0) {
    const char *e = getenv("SGL_MLX_EVAL");
    v = (e && *e) ? atoi(e) : 0;
  }
  return v;
}
/* 深度 trace：打 maxabs（含 inf/nan 标记）。只在 TRACE=2 时用。*/
static void tmax(const mx::array &a, const char *tag) {
  if (!trace2_on()) return;
  auto f = mx::astype(a, mx::float32);
  auto m = mx::max(mx::abs(f));
  m.eval();
  float v = m.item<float>();
  fprintf(stderr, "  [op] %-12s maxabs=%.6g%s%s\n", tag, v,
          std::isinf(v) ? " (INF!)" : "", std::isnan(v) ? " (NAN!)" : "");
}

/* 读 <base>.weight/scales(/biases) 三元组。成功 1；不是 MLX 量化权重返回 0。
 * out: w=[N, K*bits/32] uint32；affine 的 sc/bi=[N, K/64] f16；mxfp4 只有 sc（U8）。
 * group/bits/mode 从形状推出：K = scales列数 × group；bits = 32×weight列数 / K。*/
static int load_q3(sgl_st_ctx *st, const char *base, QW *out, int *N, int *K) {
  char nm[320];
  const char *dt = nullptr;
  int nd = 0;
  int64_t dims[4] = {0, 0, 0, 0};
  const void *p = nullptr;
  size_t nb = 0;

  /* 形状 → 量化参数。返回 false = 对不上（写明原因）。*/
  auto infer = [&](const char *b, int n, int kp, int scc, bool has_bi, int *group,
                   int *bits, int *mxfp) -> int {
    if (has_bi) {
      /* weight [N, K*bits/32] + scales [N, K/group] → bits = 32*kp/(scc*group)。
       * （之前漏除 group，8-bit 上推出 512 直接报错 —— 对拍抓出来的）*/
      *group = 64;
      *mxfp = 0;
      *bits = scc ? (32 * kp) / (scc * *group) : 0;
      if (*bits != 4 && *bits != 8) {
        MLX_ERR("%s: 只支持 4/8-bit affine（推出 bits=%d，w列=%d scales列=%d）", b, *bits,
                kp, scc);
        return 0;
      }
      return 1;
    }
    /* 无 biases：mxfp4（scales 是 U8 的 e8m0 指数，group=32）*/
    *group = 32;
    *bits = 4;
    *mxfp = 1;
    (void)n;
    return 1;
  };

  /* 首选：MLX 自己读出来的 array（无 primitive，不会被常量折叠）*/
  if (g_mxw) {
    snprintf(nm, sizeof(nm), "%s.weight", base);
    mx::array *w = mxw_get(nm);
    snprintf(nm, sizeof(nm), "%s.scales", base);
    mx::array *sc = mxw_get(nm);
    snprintf(nm, sizeof(nm), "%s.biases", base);
    mx::array *bi = mxw_get(nm);
    if (w && sc) {
      if (w->dtype() != mx::uint32 || w->ndim() != 2 || sc->ndim() != 2) return 0;
      const int n = w->shape(0), kp = w->shape(1);
      int group = 64, bits = 4, mxfp = 0;
      if (!infer(base, n, kp, sc->shape(1), bi != nullptr, &group, &bits, &mxfp)) return 0;
      if (mxfp && sc->dtype() != mx::uint8) {
        MLX_ERR("%s: 无 biases 但 scales 不是 U8（不认的格式）", base);
        return 0;
      }
      const int k = sc->shape(1) * group;
      if (sc->shape(0) != n || k % group != 0) {
        MLX_ERR("%s: 量化参数形状对不上（w=%dx%d sc=%dx%d K=%d group=%d）", base, n, kp,
                sc->shape(0), sc->shape(1), k, group);
        return 0;
      }
      *out = QW(*w, OArr(*sc), bi ? OArr(*bi) : OArr(), group, bits, mxfp);
      if (N) *N = n;
      if (K) *K = k;
      return 1;
    }
  }

  snprintf(nm, sizeof(nm), "%s.weight", base);
  if (!sgl_st_get_raw(st, nm, &dt, &nd, dims, &p, &nb)) return 0;
  if (!dt || strcmp(dt, "U32") != 0 || nd != 2) return 0; /* 不是量化权重 */
  const int n = (int)dims[0], kp = (int)dims[1];

  /* scales 先读出来定 group/bits/mode（affine=F16+有biases；mxfp4=U8无biases）*/
  snprintf(nm, sizeof(nm), "%s.scales", base);
  if (!sgl_st_get_raw(st, nm, &dt, &nd, dims, &p, &nb)) return 0;
  if (!dt || nd != 2 || (int)dims[0] != n) return 0;
  const int scales_is_u8 = strcmp(dt, "U8") == 0;
  int group = 64, bits = 4, mxfp = 0;
  if (!infer(base, n, kp, (int)dims[1], !scales_is_u8, &group, &bits, &mxfp)) return 0;
  const int k = (int)dims[1] * group;
  if (k % group != 0) {
    MLX_ERR("%s: scales 形状 [%lld,%lld] 与 K=%d/group=%d 对不上", base,
            (long long)dims[0], (long long)dims[1], k, group);
    return 0;
  }

  /* 防御：mmap 里引用的字节数必须至少 cover [n,kp] 的 uint32 —— 不然
   * vector 构造后 MLX 会读越界（在 ARM 上表现为 SIGBUS）。*/
  if (nb < (size_t)n * (size_t)kp * 4u) {
    MLX_ERR("%s.weight: 数据只有 %zu 字节，按 [%d,%d] uint32 需要 %zu", base, nb, n, kp,
            (size_t)n * (size_t)kp * 4u);
    return 0;
  }
  if (dbg()) MLX_LOG("q3 %s: w=[%d,%d] U32 (%zu B) k=%d bits=%d mxfp=%d", base, n, kp,
                     nb, k, bits, mxfp);
  mx::array w = arr_from<uint32_t>(p, mx::Shape{n, kp}, mx::uint32);
  if (dbg()) MLX_LOG("q3 %s: w ok, sum=%.4g", base,
                     mx::sum(mx::astype(w, mx::float32)).item<float>());

  /* scales：上面已经 get_raw 过，指针还有效（mmap 未释放），直接建 array */
  mx::array sc = scales_is_u8
                     ? arr_from<uint8_t>(p, mx::Shape{n, (int)dims[1]}, mx::uint8)
                     : arr_from<uint16_t>(p, mx::Shape{n, (int)dims[1]}, mx::float16);

  if (scales_is_u8) { /* mxfp4：没有 biases */
    *out = QW(w, OArr(sc), OArr(), group, bits, mxfp);
    if (N) *N = n;
    if (K) *K = k;
    return 1;
  }

  snprintf(nm, sizeof(nm), "%s.biases", base);
  if (!sgl_st_get_raw(st, nm, &dt, &nd, dims, &p, &nb)) return 0;
  if (!dt || strcmp(dt, "F16") != 0 || nd != 2) return 0;
  mx::array bi = arr_from<uint16_t>(p, mx::Shape{(int)dims[0], (int)dims[1]}, mx::float16);

  *out = QW(w, OArr(sc), OArr(bi), group, bits, mxfp);
  if (N) *N = n;
  if (K) *K = k;
  return 1;
}

/* 读一个 2D 权重：优先量化三件套，否则 <base>.weight 的 F16/F32。
 * 返回 1 成功（out 有效），0 = 目录里没这个权重。*/
static int load2d(sgl_st_ctx *st, const char *base, mx::Dtype dt, QW *out) {
  int N = 0, K = 0;
  if (load_q3(st, base, out, &N, &K)) return 1;
  char nm[320];
  snprintf(nm, sizeof(nm), "%s.weight", base);
  if (g_mxw) { /* 稠密权重也优先用 MLX 读的，config-only 时 st 里的浮点路径也行 */
    mx::array *w = mxw_get(nm);
    if (w && w->ndim() == 2) {
      mx::array a = mx::astype(*w, dt);
      if (wt_on()) {
        a = mx::contiguous(mx::transpose(a));
        a.eval();
      }
      *out = QW(a);
      return 1;
    }
  }
  sgl_tensor t = sgl_st_get(st, nm, 2, NULL);
  if (!t.data) return 0;
  mx::array a = up2w(&t, dt); /* 已经按 wt_on() 转置成 [in,out] */
  sgl_st_drop(st, &t);
  *out = QW(a);
  return 1;
}

/* 沿 N（axis=0）拼两份量化或两份稠密权重。拼之前要求 quant 状态一致。
 * 量化参数（group/bits/mxfp）也要带上 —— 之前丢了，拼出来的 8-bit 权重
 * 被当成 4-bit 去算（实测报 quantized_matmul 形状不兼容）。*/
static QW cat2(const QW &a, const QW &b) {
  if (a.quant() != b.quant()) {
    MLX_ERR("拼接的两份权重量化状态不一致（一个量化一个稠密）");
    return a;
  }
  if (a.quant() && (a.group != b.group || a.bits != b.bits || a.mxfp != b.mxfp)) {
    MLX_ERR("拼接的两份权重量化参数不一致（%d/%d/%d vs %d/%d/%d）", a.group, a.bits,
            a.mxfp, b.group, b.bits, b.mxfp);
    return a;
  }
  if (!a.quant())
    return QW(mx::concatenate(std::vector<mx::array>{a.w, b.w}, wt_on() ? 1 : 0));
  /* 量化：三件套都沿 axis=0（输出维）拼 */
  return QW(mx::concatenate(std::vector<mx::array>{a.w, b.w}, 0),
            OArr(mx::concatenate(std::vector<mx::array>{*a.sc, *b.sc}, 0)),
            a.bi ? OArr(mx::concatenate(std::vector<mx::array>{*a.bi, *b.bi}, 0)) : OArr(),
            a.group, a.bits, a.mxfp);
}
static QW cat3(const QW &a, const QW &b, const QW &c) { return cat2(cat2(a, b), c); }

/* 读一个 1D 权重（norm / bias），不存在返回空。
 * plus1 = norm_p1：Qwen3-Next 的 RMSNorm 权重用 (1+w)，上传时加进去。*/
static OArr st1(sgl_st_ctx *st, const char *name, mx::Dtype dt, int plus1) {
  /* config-only 模型（4-bit 检查点）的 L->* 全是 NULL，所以 1D 权重也得先查
   * MLX 读出来的表；否则 norm/bias 全空，前向直接崩。*/
  if (g_mxw) {
    mx::array *w = mxw_get(name);
    if (w && w->ndim() == 1) {
      mx::array a = mx::astype(*w, dt);
      return OArr(plus1 ? mx::add(a, mx::array(1.0f)) : a);
    }
  }
  sgl_tensor t = sgl_st_get(st, name, 1, NULL);
  if (!t.data) return OArr();
  mx::array a = up1_norm(&t, dt, plus1);
  sgl_st_drop(st, &t);
  return OArr(a);
}

/* 读一个 1D 权重，必须存在（否则给错误上下文）。*/
static mx::array st1_req(sgl_st_ctx *st, const char *name, mx::Dtype dt, int plus1) {
  OArr a = st1(st, name, dt, plus1);
  if (!a) {
    MLX_ERR("缺少必需权重 %s", name);
    return mx::zeros(mx::Shape{0}, dt); /* 调用方靠 shape 判失败 */
  }
  return *a;
}

/* embed：量化时【不】再反量化成常驻稠密表（0.5B 272MB / 7B 1GB），
 * 而是把三件套留在 wte_q，查表时按行 gather + dequantize（见 emb_lookup）。
 * 稠密时照旧。返回 1 = wte 有效；返回 2 = wte_q 有效；0 = 没有。*/
static int load_emb(sgl_st_ctx *st, const char *base, mx::Dtype dt, sgl_mlx_model *g) {
  QW q;
  if (load_q3(st, base, &q, NULL, NULL)) {
    g->wte_q = q;
    return 2;
  }
  char nm[320];
  snprintf(nm, sizeof(nm), "%s.weight", base);
  sgl_tensor t = sgl_st_get(st, nm, 2, NULL);
  if (!t.data) return 0;
  mx::array a = up2(&t, dt); /* 不转置：embedding 要 [V,E] 给 take */
  sgl_st_drop(st, &t);
  g->wte = OArr(a);
  return 1;
}

/* 只读量化三件套（不解量化），给 lm_head 用：4-bit 检查点上 [E,V] 的 fp16 副本
 * 是 272MB，每步 decode 都要全量读一遍；留量化形式只要 87MB，快 3 倍。*/
static int load_q3_only(sgl_st_ctx *st, const char *base, QW *out) {
  return load_q3(st, base, out, NULL, NULL);
}

/* ============================ 构造 / 析构 ============================ */

static int env_int(const char *k, int dflt) {
  const char *v = getenv(k);
  return (v && *v) ? atoi(v) : dflt;
}

sgl_mlx_model *sgl_mlx_from_llama(const void *llama_model) {
  const sgl_llama *m = (const sgl_llama *)llama_model;
  if (!m) return NULL;
  /* 量化检查点（4-bit）此时 CPU 那边是 config-only，权重全 NULL：
   * 只有先让 MLX 把 safetensors 读进 mxw，下面的 load_* 才拿得到东西。*/
  if (!g_mxw && !g_dir.empty()) mxw_open(g_dir.c_str());

  /* 范围检查：不支持的形状直接拒绝（服务层回落 CPU），绝不静默算错。
   * m->layers 只在完整加载时存在；config-only（JSON 后端 / 量化检查点）时用
   * m 上的几何字段判（那些字段在 config 解析阶段就算好了）。*/
  if (m->n_experts > 0) {
    MLX_ERR("MoE 模型（n_experts=%d）还没做（T2）", m->n_experts);
    return NULL;
  }
  if (m->q_gate) {
    MLX_ERR("q_gate（G13a 门控注意力）还没做");
    return NULL;
  }
  if (m->layers) {
    for (int l = 0; l < m->n_layer; l++) {
      const sgl_llama_layer *L = &m->layers[l];
      if (L->lin_idx >= 0) {
        MLX_ERR("hybrid/linear-attention 层（layer %d）还没做（T3 用 mx::fast::metal_kernel）", l);
        return NULL;
      }
      if (L->is_moe || L->router_w.data) {
        MLX_ERR("layer %d 是 MoE 层，还没做（T2）", l);
        return NULL;
      }
    }
  } else if (m->layer_kind) {
    for (int l = 0; l < m->n_layer; l++)
      if (m->layer_kind[l] != 0) {
        MLX_ERR("hybrid/linear-attention 层（layer %d）还没做（T3）", l);
        return NULL;
      }
  }
  if (m->head_dim <= 0 || m->n_kv_head <= 0 || m->n_head % m->n_kv_head != 0) {
    MLX_ERR("几何不支持：head_dim=%d n_head=%d n_kv_head=%d", m->head_dim, m->n_head,
            m->n_kv_head);
    return NULL;
  }

  sgl_mlx_model *g = new sgl_mlx_model();
  g->L = m->n_layer;
  g->H = m->n_head;
  g->KVH = m->n_kv_head;
  g->E = m->n_embd;
  g->I = m->n_inter;
  g->D = m->head_dim;
  g->qdim = m->q_dim;
  g->kvdim = m->kv_dim;
  g->V = m->vocab;
  g->eps = m->rms_eps;
  g->pos_scale = m->rope_pos_scale;
  g->attn_factor = m->rope_attn_factor;
  g->gemma_scale = m->gemma_scale;
  g->emb_softcap = m->emb_softcap;
  g->tie = m->tie_embeddings;
  g->norm_p1 = m->norm_p1;
  g->sliding_window = m->sliding_window;
  g->rep = g->H / g->KVH;

  /* dtype：SGL_MLX_FP16=0 走 fp32（对拍/排障用）；KV 单独一个开关 */
  int fp16 = env_int("SGL_MLX_FP16", 1);
  g->wdt = fp16 ? mx::float16 : mx::float32;
  g->kdt = env_int("SGL_MLX_KV16", fp16) ? mx::float16 : mx::float32;

  /* ctx：默认取模型的 n_ctx，可用 SGL_MLX_CTX 压小（16GB 机器上跑大模型要用）*/
  int want = env_int("SGL_MLX_CTX", m->n_ctx);
  if (want <= 0) want = m->n_ctx;
  if (want > m->n_ctx) want = m->n_ctx;
  g->ctx = want;

  /* 总入口：所有权重（包括稠密 fp16）都直读 safetensors。
   * 为何不再走 m->layers 里的 sgl_tensor：config-only 模式下那些字段全是 NULL，
   * 而 4-bit 检查点的 U32 权重 sgl_st_get 根本拿不到；两条路写成一套就不会
   * “fp16 能跑、4bit 不行”这种分叉 bug。测试已回归：fp16 结果与改前逐项一致。*/
  load_emb(m->st, "model.embed_tokens", g->wdt, g);
  g->final_norm = st1(m->st, "model.norm.weight", g->wdt, g->norm_p1);
  if ((!g->wte && !g->wte_q.quant()) || !g->final_norm) {
    MLX_ERR("缺 embed_tokens 或 model.norm.weight");
    delete g;
    return NULL;
  }
  /* lm_head 走 proj，所以要转置；tie 时 wte 本身是 [V,E]（take 要用），
   * 单独物化一份 [E,V] 副本 —— 一次性代价，换取每步 decode 不再转置。
   *
   * 量化检查点：tie 时直接用 embed 的量化三件套当 head（不物化稠密副本）：
   * 0.5B-4bit 的 [V,E] fp16 副本 272MB，量化三件套只要 87MB；
   * 实测这是量化 decode 的头号开销来源（272MB 逐 token 扫一遍）。*/
  {
    QW lh_q;
    int lh_q_ok = load_q3_only(m->st, "lm_head", &lh_q);
    if (g->tie && g->wte_q.quant()) {
      g->lm_head_q = g->wte_q; /* tie：head = embed 同一份量化表，无需另存 */
    } else if (g->tie && g->wte && wt_on()) {
      g->lm_head = OArr(mx::contiguous(mx::transpose(*g->wte)));
    } else if (g->tie && g->wte) {
      g->lm_head = g->wte;
    } else if (lh_q_ok) {
      g->lm_head_q = lh_q; /* 不 tie 但 lm_head 量化：proj 直接吃三件套 */
    } else {
      sgl_tensor t = sgl_st_get(m->st, "lm_head.weight", 2, NULL);
      if (t.data) {
        mx::array a = up2w(&t, g->wdt);
        sgl_st_drop(m->st, &t);
        g->lm_head = OArr(a);
      }
    }
  }
  if (!g->lm_head && !g->lm_head_q.quant()) {
    MLX_ERR("没有 lm_head 且 tie=0");
    delete g;
    return NULL;
  }

  /* rope 表：partial rotary 时用 rot_invf（每层 rot_dim/2），否则 layer_invf（每层 D/2）*/
  int rot = m->rot_dim;
  if (rot <= 0 || rot > g->D) rot = g->D;
  if (rot % 2) rot &= ~1; /* 奇数维不成对，退到偶数 */
  const double *src_invf =
      (m->rot_dim > 0 && m->rot_dim < g->D) ? m->rot_invf : m->layer_invf;
  int half = rot / 2;
  if (!src_invf || half <= 0) {
    MLX_ERR("rope 逆频率表缺失（rot=%d half=%d）", rot, half);
    delete g;
    return NULL;
  }

  int L = g->L;
  g->inp_norm.reserve(L); g->qkv_w.reserve(L);
  g->o_w.reserve(L); g->post_norm.reserve(L); g->gu_w.reserve(L);
  g->down_w.reserve(L); g->kc.reserve(L); g->vc.reserve(L);
  g->qkv_b.resize(L);
  g->q_norm.resize(L); g->k_norm.resize(L);
  g->rot.assign(L, rot);
  g->invf.resize(L);
  g->layer_window.assign(L, 0);

  for (int l = 0; l < L; l++) {
    /* 层名前缀。权重一律从 m->st 读（见上面“总入口”的说明）。*/
    char p[192];
    snprintf(p, sizeof(p), "model.layers.%d", l);
    char nm[320];

    snprintf(nm, sizeof(nm), "%s.input_layernorm.weight", p);
    g->inp_norm.push_back(st1_req(m->st, nm, g->wdt, g->norm_p1));

    /* q/k/v 拼成一份。量化时三件套都沿 axis=0（输出维）拼，稠密时按 wt_on()。*/
    {
      QW qa, ka, va;
      snprintf(nm, sizeof(nm), "%s.self_attn.q_proj", p);
      if (!load2d(m->st, nm, g->wdt, &qa)) {
        MLX_ERR("缺 %s.weight", nm);
        delete g;
        return NULL;
      }
      snprintf(nm, sizeof(nm), "%s.self_attn.k_proj", p);
      if (!load2d(m->st, nm, g->wdt, &ka)) {
        MLX_ERR("缺 %s.weight", nm);
        delete g;
        return NULL;
      }
      snprintf(nm, sizeof(nm), "%s.self_attn.v_proj", p);
      if (!load2d(m->st, nm, g->wdt, &va)) {
        MLX_ERR("缺 %s.weight", nm);
        delete g;
        return NULL;
      }
      g->qkv_w.push_back(cat3(qa, ka, va));
    }
    snprintf(nm, sizeof(nm), "%s.self_attn.o_proj", p);
    {
      QW o;
      if (!load2d(m->st, nm, g->wdt, &o)) {
        MLX_ERR("缺 %s.weight（o_proj）", nm);
        delete g;
        return NULL;
      }
      g->o_w.push_back(o);
    }

    snprintf(nm, sizeof(nm), "%s.post_attention_layernorm.weight", p);
    g->post_norm.push_back(st1_req(m->st, nm, g->wdt, g->norm_p1));
    {
      QW ga, ua;
      snprintf(nm, sizeof(nm), "%s.mlp.gate_proj", p);
      if (!load2d(m->st, nm, g->wdt, &ga)) {
        MLX_ERR("缺 %s.weight", nm);
        delete g;
        return NULL;
      }
      snprintf(nm, sizeof(nm), "%s.mlp.up_proj", p);
      if (!load2d(m->st, nm, g->wdt, &ua)) {
        MLX_ERR("缺 %s.weight", nm);
        delete g;
        return NULL;
      }
      g->gu_w.push_back(cat2(ga, ua));
    }
    snprintf(nm, sizeof(nm), "%s.mlp.down_proj", p);
    {
      QW d;
      if (!load2d(m->st, nm, g->wdt, &d)) {
        MLX_ERR("缺 %s.weight（down_proj）", nm);
        delete g;
        return NULL;
      }
      g->down_w.push_back(d);
    }

    /* 投影 bias（Qwen2 系）：三个里任何一个存在就拼一份（不存在的补 0）。
     * bias 从不量化（F16 小张量），所以走普通 sgl_st_get。*/
    {
      char bq_n[320], bk_n[320], bv_n[320];
      snprintf(bq_n, sizeof(bq_n), "%s.self_attn.q_proj.bias", p);
      snprintf(bk_n, sizeof(bk_n), "%s.self_attn.k_proj.bias", p);
      snprintf(bv_n, sizeof(bv_n), "%s.self_attn.v_proj.bias", p);
      OArr bq = st1(m->st, bq_n, g->wdt, 0), bk = st1(m->st, bk_n, g->wdt, 0),
           bv = st1(m->st, bv_n, g->wdt, 0);
      if (bq || bk || bv) {
        const int K = g->kvdim;
        auto pad = [&](const OArr &a, int n) {
          if (a) return *a;
          return mx::zeros(mx::Shape{n}, g->wdt);
        };
        g->qkv_b[l] = OArr(mx::concatenate(
            std::vector<mx::array>{pad(bq, g->qdim), pad(bk, K), pad(bv, K)}, 0));
      }
    }
    /* qk-norm（Qwen3 系）的 eps 在 CPU 侧写死 1e-6（见 llama.c），这里保持一致 */
    {
      char qn[320], kn[320];
      snprintf(qn, sizeof(qn), "%s.self_attn.q_norm.weight", p);
      snprintf(kn, sizeof(kn), "%s.self_attn.k_norm.weight", p);
      g->q_norm[l] = st1(m->st, qn, g->wdt, g->norm_p1);
      g->k_norm[l] = st1(m->st, kn, g->wdt, g->norm_p1);
    }
    g->invf[l].assign(src_invf + (size_t)l * half, src_invf + (size_t)(l + 1) * half);
    if (m->layer_window) g->layer_window[l] = m->layer_window[l];
    g->kc.push_back(mx::zeros(mx::Shape{g->ctx, g->KVH, g->D}, g->kdt));
    g->vc.push_back(mx::zeros(mx::Shape{g->ctx, g->KVH, g->D}, g->kdt));
  }
  g->rope_base.assign(L, 0.0);
  /* 判断能不能走 mx::fast::rope：全维旋转 + 没折过 pos_scale/attn_factor +
   * invf 就是标准表 invf[i] = base^(-2i/D)。满足时反推 base。
   * （invf[0] 恒为 1，所以用 invf[1] 反推：invf[1] = base^(-2/D) → base = invf[1]^(-D/2)。）*/
  if (rot == g->D && g->pos_scale == 1.0 && g->attn_factor == 1.0 && half >= 2) {
    for (int l = 0; l < L; l++) {
      const std::vector<double> &iv = g->invf[l];
      if ((int)iv.size() != half || !(iv[1] > 0.0) || iv[1] >= 1.0) continue;
      const double base = std::pow(iv[1], -(double)g->D / 2.0);
      int ok = 1;
      for (int i = 0; i < half; i++) { /* 逐项对一下，别把「像但不等于」的当标准*
                                       *（gemma-2 的 dynamic NTK 每层不同，就靠这挡住）*/
        const double want = std::pow(base, -(double)(2 * i) / (double)g->D);
        if (std::fabs(iv[i] - want) > std::fabs(want) * 1e-12) {
          ok = 0;
          break;
        }
      }
      if (ok) g->rope_base[l] = base;
    }
  }
  {
    int nf = 0;
    for (int l = 0; l < L; l++)
      if (g->rope_base[l] > 0) nf++;
    if (nf > 0) printf("[mlx] rope: %d/%d 层走 fast::rope（base=%.6g）\n", nf, L, g->rope_base[0]);
  }
  for (int l = 0; l < L; l++) {
    g->kc[l].eval();
    g->vc[l].eval();
  }

  printf("[mlx] loaded: L=%d H=%d KVH=%d E=%d I=%d D=%d V=%d ctx=%d rot=%d %s/%s%s\n", g->L,
         g->H, g->KVH, g->E, g->I, g->D, g->V, g->ctx, rot,
         g->wdt == mx::float16 ? "fp16" : "fp32", g->kdt == mx::float16 ? "kv16" : "kv32",
         g->norm_p1 ? " norm+1" : "");
  fflush(stdout);
  return g;
}

void sgl_mlx_free(sgl_mlx_model *m) { delete m; }

/* 直接从模型目录建 MLX 实例（mlx-community 的 4-bit 检查点走这里）。
 *
 * 与 sgl_mlx_from_llama 的关系：几何与 config 仍由 sgl_llama_load_full(dir, 1)
 * 解析（那里面有 rope scaling / per-layer 窗口 / norm_p1 / tie 等数百行细节，
 * 不重复实现），只是不加载权重。权重全由 build 阶段从 m->st 直读，所以 fp16 与
 * 4-bit 共用一套加载代码（见上面的“总入口”说明）。 */
sgl_mlx_model *sgl_mlx_from_dir(const char *dir) {
  mxw_open(dir); /* 有 model.safetensors 就让 MLX 自己读（4-bit 必须）*/
  sgl_llama *m = sgl_llama_load_full(dir, /*config_only=*/1);
  if (!m) {
    mxw_close();
    return NULL;
  }
  sgl_mlx_model *g = sgl_mlx_from_llama(m);
  sgl_llama_free(m);
  mxw_close(); /* array 已持有各自的数据（拷贝/引用计数），可以放掉 map */
  return g;
}

int sgl_mlx_ctx(const sgl_mlx_model *m) { return m ? m->ctx : 0; }
int sgl_mlx_vocab(const sgl_mlx_model *m) { return m ? m->V : 0; }

const char *sgl_mlx_device_name(void) {
  static char buf[128];
  if (!buf[0]) {
#ifdef __APPLE__
    size_t n = sizeof(buf) - 1;
    if (sysctlbyname("machdep.cpu.brand_string", buf, &n, NULL, 0) != 0) snprintf(buf, sizeof(buf), "Apple Silicon");
#else
    snprintf(buf, sizeof(buf), "mlx");
#endif
  }
  return buf;
}

/* 统一内存预算：Metal 的推荐工作集约为物理内存的 75%（本机 M1/16GB 实测
 * maxRecommendedWorkingSetSize=11.84GB ≈ 0.74×16GB）。这里不引 Metal 头，
 * 直接用 sysctl 的 hw.memsize 乘 3/4 —— 与实测值差 <2%。*/
long sgl_mlx_avail_mem_mb(void) {
#ifdef __APPLE__
  int64_t bytes = 0;
  size_t n = sizeof(bytes);
  if (sysctlbyname("hw.memsize", &bytes, &n, NULL, 0) != 0 || bytes <= 0) return -1;
  return (long)(bytes * 3 / 4 / 1048576);
#else
  return -1;
#endif
}

/* ============================ rope ============================
 * 两条路：
 *   快：mx::fast::rope（1 个 Metal kernel）。条件 —— 全维旋转、所有层逆频率相同
 *       （= 标准 RoPE，base 能从 invf[1] 反推）、attn_factor==1、pos_scale==1。
 *       这是绝大多数模型的路径（Qwen2/2.5、Llama、Mistral…）。
 *   慢：手写（~10 个 op）。probe2 实测 fast::rope 的 freqs 参数与本仓库
 *       layer_invf 语义不等价（err≈1.07），而且它做不了 attn_factor 与
 *       partial rotary —— 这些情况才回退。角度在 host 用 double 算，
 *       与 CPU 的 rope_layer 逐字同式（probe2 3e 实测一致）。*
 *
 * 为什么快路重要：手写版每层 q/k 各 ~10 个 op，24 层就是 ~480 个 kernel。
 * M1 上 kernel launch 是 decode 的主要成本（见 docs/NOTES_mlx_t2.md）。*/

/* cos/sin 表：n_pos 个绝对位置 → [n_pos, half] */
static void rope_cs(const sgl_mlx_model *g, int l, const int *pos, int n_pos,
                    std::vector<float> &cs, std::vector<float> &sn) {
  const int half = g->rot[l] / 2;
  const double *invf = g->invf[l].data();
  const double sc = g->pos_scale, af = g->attn_factor;
  cs.assign((size_t)n_pos * half, 0.f);
  sn.assign((size_t)n_pos * half, 0.f);
  for (int s = 0; s < n_pos; s++) {
    const double p = (double)pos[s] * sc; /* CPU: double pos = (pos0+t) * rope_pos_scale */
    for (int i = 0; i < half; i++) {
      const double ang = p * (double)invf[i];
      cs[(size_t)s * half + i] = (float)(std::cos(ang) * af);
      sn[(size_t)s * half + i] = (float)(std::sin(ang) * af);
    }
  }
}

/* x: [B, Hh, S, D]，只旋转每头前 rot 维；cs/sn 的形状 cs_shape 要能广播到 [B,Hh,S,half]。
 * 中间用 fp32 算（cos/sin 是 fp32），最后转回 x 的 dtype。*/
static mx::array rope_apply(const mx::array &x, const mx::Shape &xs, int D, int rot,
                            const std::vector<float> &cs, const std::vector<float> &sn,
                            const mx::Shape &cs_shape) {
  const int half = rot / 2;
  auto C = mx::array(cs.data(), cs_shape, mx::float32);
  auto S = mx::array(sn.data(), cs_shape, mx::float32);
  mx::Shape st{0, 0, 0, 0};
  mx::Shape sp1 = xs;
  sp1[3] = half;
  mx::Shape st2 = st;
  st2[3] = half;
  mx::Shape sp2 = xs;
  sp2[3] = rot;
  auto x1 = mx::slice(x, st, sp1);
  auto x2 = mx::slice(x, st2, sp2);
  auto o1 = mx::astype(mx::subtract(mx::multiply(x1, C), mx::multiply(x2, S)), x.dtype());
  auto o2 = mx::astype(mx::add(mx::multiply(x1, S), mx::multiply(x2, C)), x.dtype());
  if (rot >= D) return mx::concatenate(std::vector<mx::array>{o1, o2}, -1);
  mx::Shape st3 = st;
  st3[3] = rot; /* partial rotary：rot 之后的维原样带过 */
  auto rest = mx::slice(x, st3, xs);
  return mx::concatenate(std::vector<mx::array>{o1, o2, rest}, -1);
}

/* 快路：能不能整层交给 mx::fast::rope。pos_scale/attn_factor 已折进 cs/sn，
 * 所以这里只要求「没折过」；invf 相同性在建模型时算好存进 g->rope_base[l]
 * （<=0 = 不可用）。
 * offset 必须是**整数**、且只能是标量或长度 1 的向量（两条都是实测报错逼出来的）：
 *   "[rope] offset must be an integer but got type float32."
 *   "[rope] offset must be a scalar or vector with 1 elements but has shape (8)."*/
static mx::array rope_fast(const sgl_mlx_model *g, int l, const mx::array &x4, int D,
                           int pos) {
  static thread_local std::vector<int> one(1, 0);
  one[0] = pos;
  auto offset = mx::array(one.data(), mx::Shape{1}, mx::int32);
  return mx::fast::rope(x4, D, /*traditional=*/false, g->rope_base[l], 1.0f, offset);
}

/* ============================ 因果 mask ============================
 * 加性 mask（fp32，0 / -1e30）。列 p 是**段内相对位置**（KV 切片已从段基址起）。
 * 按 window 值缓存：同一次前向里各层窗口通常只有 1~2 种取值（mistral 全同、
 * gemma-2 局部/全局交替），不必每层重建、重传。*/
struct MaskCache {
  std::vector<int> wins;
  std::vector<mx::array> arrs;
  const mx::array *get(int win, int n, int pos0, int Sk) {
    for (size_t i = 0; i < wins.size(); i++)
      if (wins[i] == win) return &arrs[i];
    std::vector<float> mk((size_t)n * Sk, 0.f);
    const float NEG = -1e30f;
    for (int i = 0; i < n; i++) {
      int upto = pos0 + i + 1;
      int from = (win > 0 && upto > win) ? upto - win : 0;
      float *r = mk.data() + (size_t)i * Sk;
      for (int p = 0; p < Sk; p++) r[p] = (p >= from && p < upto) ? 0.f : NEG;
    }
    wins.push_back(win);
    arrs.push_back(mx::array(mk.data(), mx::Shape{1, 1, n, Sk}, mx::float32));
    return &arrs.back();
  }
};

/* [n,in] @ [in,out]（wt_on）或 [n,in] @ [out,in]^T（!wt_on）→ [n,out]
 * 量化权重：weight 已是 [out, in/8] 的 U32 打包，quantized_matmul 自己解包并转置。*/
static mx::array proj(const mx::array &x, const QW &W) {
  if (W.quant()) {
    /* mxfp4 kernel（MLX 0.29.3）对 f16/bf16 输入给出垃圾值（f16 输入 2e23 vs
     * f32 输入 1659，实测差 20 个量级）—— 强制 f32 输入，算完转回去。
     * affine（4/8-bit）无此问题，实测 f16 输入与 mlx_lm 逐位一致。*/
    if (W.mxfp)
      return mx::astype(mx::quantized_matmul(mx::astype(x, mx::float32), W.w, *W.sc, W.bi,
                                             /*transpose=*/true, W.group, W.bits, W.mode()),
                        x.dtype());
    return mx::quantized_matmul(x, W.w, *W.sc, W.bi, /*transpose=*/true, W.group, W.bits,
                                W.mode());
  }
  return wt_on() ? mx::matmul(x, W.w) : mx::matmul(x, mx::transpose(W.w));
}

/* logits = x @ lm_head^T。量化 head 优先（4-bit 下只读约 87MB，而稠密副本是
 * 272MB，逐 token 扫一遍差异巨大）。传 g 而不是用全局：服务层是多线程调 forward。*/
static mx::array head(sgl_mlx_model *g, const mx::array &x) {
  if (g->lm_head_q.quant()) return proj(x, g->lm_head_q);
  return proj(x, *g->lm_head);
}

/* embed 查表（量化表：按行 gather 三件套再 dequantize —— 整表反量化 0.5B 要
 * 272MB / 7B 要 1GB 常驻，而每步只取 B 行；量化 take 的代价是 B×E×bits/8 字节）。
 * 不用 gather_qmm：x=[n,1] 的伪 matmul 路径 kernel 覆盖差（待实测，见 NOTES_mlx_q4）。
 * 两个调用方（forward_impl / decode_batch）共用。*/
static mx::array emb_lookup(const sgl_mlx_model *g, const mx::array &idx) {
  if (!g->wte_q.quant()) return mx::take(*g->wte, idx, 0);
  const QW &q = g->wte_q;
  auto wg = mx::take(q.w, idx, 0);
  auto sg = mx::take(*q.sc, idx, 0);
  /* biases 也必须按行取：全量 [V, scc] 传给 dequantize 会因形状与 wg 不一致
   * 被拒（8-bit 上实测 Abort；4-bit 是 MLX 恰好没查 —— 别赌）。*/
  OArr bg = q.bi ? OArr(mx::take(*q.bi, idx, 0)) : OArr();
  auto d = mx::dequantize(wg, sg, bg, q.group, q.bits, q.mode());
  /* mxfp4 的 dequantize 输出 bf16（affine 是 f16）→ 统一回 wdt，后面残差/归一
   * 不至于一路 bf16 混算。*/
  return d.dtype() == g->wdt ? d : mx::astype(d, g->wdt);
}

/* ============================ 前向（prefill / 多行 / emb）============================ */

static int forward_impl(sgl_mlx_model *g, const float *emb, const int *ids, int n, int pos0,
                        int kb, float *logits, int out_rows) {
  if (!g || n <= 0) return -1;
  if (!logits && out_rows > 0) out_rows = 0;
  if (pos0 < 0 || kb < 0 || kb + pos0 + n > g->ctx) {
    MLX_ERR("forward 越界：kb=%d pos0=%d n=%d ctx=%d", kb, pos0, n, g->ctx);
    return -1;
  }
  const int E = g->E, H = g->H, KVH = g->KVH, D = g->D;

  /* 1) 词嵌入（或预融合的视觉 embedding）+ gemma 系缩放
   * 注：mx::array 没有默认构造函数，所以用三目一次初始化。*/
  mx::array x = emb ? mx::astype(mx::array(emb, mx::Shape{n, E}, mx::float32), g->wdt)
                    : emb_lookup(g, mx::array(ids, mx::Shape{n}, mx::int32));
  if (g->emb_softcap > 0) /* gemma-3：先 softcap 再 sqrt(E)，顺序与 CPU 一致 */
    x = mx::multiply(mx::array(g->emb_softcap),
                     mx::tanh(mx::divide(x, mx::array(g->emb_softcap))));
  if (g->gemma_scale > 0) x = mx::multiply(x, mx::array(g->gemma_scale));
  if (trace_on()) {
    auto t = mx::sum(mx::astype(x, mx::float32));
    t.eval();
    fprintf(stderr, "[mlx.trace] emb x.sum=%.8g\n", t.item<float>());
  }

  /* rope 位置表（本次调用内所有层共用位置，逆频率表逐层不同）*/
  std::vector<int> pos((size_t)n);
  for (int i = 0; i < n; i++) pos[(size_t)i] = pos0 + i;
  std::vector<float> cs, sn;

  const int Sk = pos0 + n; /* 段内已有 + 本次写入的总行数 */
  MaskCache mc;
  const float inv_sqrtD = 1.0f / std::sqrt((float)D);

  for (int l = 0; l < g->L; l++) {
    auto h = mx::fast::rms_norm(x, g->inp_norm[l], g->eps); /* [n,E] */

    /* 2) q/k/v 投影：一份 [E, qdim+2kvdim] 权重一次 matmul，再切三段 */
    auto qkv = proj(h, g->qkv_w[l]); /* [n, qdim+2kvdim] */
    tmax(qkv, "qkv");
    if (g->qkv_b[l]) qkv = mx::add(qkv, *g->qkv_b[l]);
    const int QD = g->qdim, KD = g->kvdim;
    auto q = mx::slice(qkv, mx::Shape{0, 0}, mx::Shape{n, QD});
    auto k = mx::slice(qkv, mx::Shape{0, QD}, mx::Shape{n, QD + KD});
    auto v = mx::slice(qkv, mx::Shape{0, QD + KD}, mx::Shape{n, QD + 2 * KD});

    /* 3) qk-norm（每头 over D，eps 写死 1e-6，与 CPU 一致）*/
    if (g->q_norm[l]) {
      auto qh = mx::fast::rms_norm(mx::reshape(q, mx::Shape{n * H, D}), *g->q_norm[l], 1e-6f);
      q = mx::reshape(qh, mx::Shape{n, g->qdim});
    }
    if (g->k_norm[l]) {
      auto kh = mx::fast::rms_norm(mx::reshape(k, mx::Shape{n * KVH, D}), *g->k_norm[l], 1e-6f);
      k = mx::reshape(kh, mx::Shape{n, g->kvdim});
    }

    /* 4) 排成 [1,Hh,S,D]（rope 的位置轴是 -2）*/
    auto q4 = mx::transpose(mx::reshape(q, mx::Shape{1, n, H, D}), std::vector<int>{0, 2, 1, 3});
    auto k4 = mx::transpose(mx::reshape(k, mx::Shape{1, n, KVH, D}), std::vector<int>{0, 2, 1, 3});
    auto v4 = mx::transpose(mx::reshape(v, mx::Shape{1, n, KVH, D}), std::vector<int>{0, 2, 1, 3});

    /* 5) rope：标准 RoPE 且**整次调用共用一个位置**时走 fast::rope（1 个 kernel）。
     * mx::fast::rope 的 offset 只能是标量或长度 1 的向量（实测报错：
     *   "[rope] offset must be a scalar or vector with 1 elements but has shape (8)"）
     * —— 所以 prefill 的 n>1（每个位置各不相同）必须回退手写。prefill 是一次性的
     * 大计算，op 数不是瓶颈；decode 每步一个位置，正好命中快路。*/
    if (g->rope_base[l] > 0 && n == 1) {
      q4 = rope_fast(g, l, q4, D, pos[0]);
      k4 = rope_fast(g, l, k4, D, pos[0]);
    } else {
      rope_cs(g, l, pos.data(), n, cs, sn);
      mx::Shape cs_shape{1, 1, n, g->rot[l] / 2};
      q4 = rope_apply(q4, mx::Shape{1, H, n, D}, D, g->rot[l], cs, sn, cs_shape);
      k4 = rope_apply(k4, mx::Shape{1, KVH, n, D}, D, g->rot[l], cs, sn, cs_shape);
    }

    /* 6) 写 KV：[1,Hh,n,D] → [n,Hh,D] → slice_update 到池子的 [kb+pos0, +n) 行 */
    auto krows = mx::reshape(mx::transpose(k4, std::vector<int>{0, 2, 1, 3}),
                             mx::Shape{n, KVH, D});
    auto vrows = mx::reshape(mx::transpose(v4, std::vector<int>{0, 2, 1, 3}),
                             mx::Shape{n, KVH, D});
    if (g->kdt != g->wdt) {
      krows = mx::astype(krows, g->kdt);
      vrows = mx::astype(vrows, g->kdt);
    }
    if (eval_mask() & 1) {
      krows.eval();
      vrows.eval();
    }
    const int w0 = kb + pos0;
    g->kc[l] = mx::slice_update(g->kc[l], krows, mx::Shape{w0, 0, 0},
                                mx::Shape{w0 + n, KVH, D});
    g->vc[l] = mx::slice_update(g->vc[l], vrows, mx::Shape{w0, 0, 0},
                                mx::Shape{w0 + n, KVH, D});

    /* 7) 注意力：读本段 [kb, kb+Sk) 行 */
    tmax(q4, "q.rope");
    tmax(k4, "k.rope");
    auto Kf = mx::slice(g->kc[l], mx::Shape{kb, 0, 0}, mx::Shape{kb + Sk, KVH, D});
    auto Vf = mx::slice(g->vc[l], mx::Shape{kb, 0, 0}, mx::Shape{kb + Sk, KVH, D});
    if (g->kdt != g->wdt) {
      Kf = mx::astype(Kf, g->wdt);
      Vf = mx::astype(Vf, g->wdt);
    }
    auto K4 = mx::transpose(mx::reshape(Kf, mx::Shape{1, Sk, KVH, D}),
                            std::vector<int>{0, 2, 1, 3}); /* [1,KVH,Sk,D] */
    auto V4 = mx::transpose(mx::reshape(Vf, mx::Shape{1, Sk, KVH, D}),
                            std::vector<int>{0, 2, 1, 3});
    if (g->rep > 1) { /* GQA：kv 头 h/rep 被 rep 个 q 头共用，repeat 后逐头对齐 */
      K4 = mx::repeat(K4, g->rep, 1);
      V4 = mx::repeat(V4, g->rep, 1);
    }
    /* 分数矩阵必须 f32：hidden 的 max|x| 在 7B-4bit 上到 12032，QK^T 若存回
     * fp16 直接溢出成 inf（Metal matmul 内部 f32 累加但输出按输入 dtype 存储；
     * 官方 sdpa 的分数始终留在 f32 寄存器）。0.5B 激活小从未触发，一上 7B 就 NaN。*/
    auto sc = mx::matmul(mx::astype(q4, mx::float32),
                         mx::transpose(mx::astype(K4, mx::float32),
                                       std::vector<int>{0, 1, 3, 2})); /* [1,H,n,Sk] f32 */
    sc = mx::multiply(sc, mx::array(inv_sqrtD));
    tmax(sc, "scores");
    sc = mx::add(sc, *mc.get(g->layer_window[l], n, pos0, Sk));
    if (eval_mask() & 2) sc.eval();
    auto aw = mx::astype(mx::softmax(sc, std::vector<int>{-1}, true), g->wdt);
    tmax(aw, "softmax");
    auto ao = mx::matmul(aw, V4);                                              /* [1,H,n,D] */
    tmax(ao, "pv");
    auto attn = mx::reshape(mx::transpose(ao, std::vector<int>{0, 2, 1, 3}),
                            mx::Shape{n, g->qdim});

    /* 8) o 投影 + 残差 */
    h = proj(attn, g->o_w[l]); /* [n,E] */
    tmax(h, "o_proj");
    x = mx::add(x, h);
    tmax(x, "x.attn_out");

    /* 9) MLP：SwiGLU = (silu(gate) * up) @ down^T */
    auto h2 = mx::fast::rms_norm(x, g->post_norm[l], g->eps);
    auto gu = proj(h2, g->gu_w[l]); /* [n,2I] */
    tmax(gu, "gate_up");
    auto gt = mx::slice(gu, mx::Shape{0, 0}, mx::Shape{n, g->I});
    auto up = mx::slice(gu, mx::Shape{0, g->I}, mx::Shape{n, 2 * g->I});
    auto act = mx::multiply(mx::multiply(gt, mx::sigmoid(gt)), up);
    tmax(act, "silu*up");
    auto dn = proj(act, g->down_w[l]);
    tmax(dn, "down");
    x = mx::add(x, dn);
    if (eval_mask() & 4) x.eval();

    /* 逐层 trace（SGL_MLX_TRACE=1）：与 Python 侧对照定位分叉层。强制同步，
     * 只在排查时开。*/
    if (trace_on()) {
      auto t = mx::sum(mx::astype(x, mx::float32));
      auto am = mx::max(mx::abs(mx::astype(x, mx::float32)));
      auto both = std::pair<mx::array, mx::array>(t, am);
      both.first.eval();
      both.second.eval();
      fprintf(stderr, "[mlx.trace] L%d x.sum=%.8g max|x|=%.6g size=%zu shape=[%d,%d]\n",
              l, both.first.item<float>(), both.second.item<float>(), x.dtype().size(),
              x.shape(0), x.shape(1));
    }
  }

  /* 10) 末层归一化 + lm_head */
  auto hf = mx::fast::rms_norm(x, *g->final_norm, g->eps);
  if (trace_on()) {
    auto tf = mx::astype(hf, mx::float32);
    auto ts = mx::sum(tf);
    ts.eval();
    const float *p = tf.data<float>();
    fprintf(stderr, "[mlx.trace] hf x.sum=%.8g  hf[0,:4]=%.5f %.5f %.5f %.5f\n",
            ts.item<float>(), p[0], p[1], p[2], p[3]);
  }
  if (out_rows <= 1) {
    /* 只要最后一行：lm_head 只跑 M=1（V 可能 15 万，省掉 (n-1)*V 的算力与内存）*/
    auto last = mx::slice(hf, mx::Shape{n - 1, 0}, mx::Shape{n, E});
    auto lg = mx::astype(head(g, last), mx::float32); /* [1,V] */
    lg.eval();
    if (logits) memcpy(logits, lg.data<float>(), (size_t)g->V * sizeof(float));
  } else {
    auto lg = mx::astype(head(g, hf), mx::float32); /* [n,V] */
    lg.eval();
    if (logits) memcpy(logits, lg.data<float>(), (size_t)n * g->V * sizeof(float));
  }
  return 0;
}

int sgl_mlx_forward_seg(sgl_mlx_model *m, const int *tokens, int n, int pos0, int kv_base,
                        float *logits_out) {
  return forward_impl(m, NULL, tokens, n, pos0, kv_base, logits_out, logits_out ? 1 : 0);
}

int sgl_mlx_forward_seg_rows(sgl_mlx_model *m, const int *tokens, int n, int pos0, int kv_base,
                             float *logits_out) {
  return forward_impl(m, NULL, tokens, n, pos0, kv_base, logits_out, n);
}

int sgl_mlx_forward_seg_emb(sgl_mlx_model *m, const float *emb, int n, int pos0, int kv_base,
                            float *logits_out) {
  return forward_impl(m, emb, NULL, n, pos0, kv_base, logits_out, logits_out ? 1 : 0);
}

/* ============================ 批量 decode ============================
 * B 条序列各 1 token。与 prefill 的差别只有三处：
 *   - 位置逐行不同（lens[b]），rope 的 cos/sin 排成 [B,1,1,half]；
 *   - KV 写入是 B 个散点行（各序列段基址不同）→ B 次 slice_update；
 *   - mask 是 [B,1,1,Smax]，每行只放开自己段内的 [from, lens[b]]。
 * q/k/v 天然是 [B, Hh*1*D] → reshape 成 [B,Hh,1,D] 就是 rope 要的排布，不用转置。*/
/* AB 诊断位（SGL_MLX_AB）：只用于定位性能，不改变正确性路径的默认值
 *   1 = 不算 lm_head（看它占多少）
 *   2 = 不写 KV（跳过 slice_update）
 *   4 = GQA 用 broadcast view 代替 repeat（看重复拷贝的开销）*/
static int ab_mask(void) {
  static int v = -1;
  if (v < 0) {
    const char *e = getenv("SGL_MLX_AB");
    v = (e && *e) ? atoi(e) : 0;
  }
  return v;
}

int sgl_mlx_decode_batch(sgl_mlx_model *g, const int *tokens, const int *base_pos,
                         const int *lens, int B, float *logits_out) {
  if (!g || B <= 0 || !tokens || !base_pos || !lens || !logits_out) return -1;
  const int H = g->H, KVH = g->KVH, D = g->D, V = g->V;

  int Smax = 0;
  int span_lo = 1 << 30, span_hi = 0, maxlen = 0; /* 紧凑化用：活跃行的最小/最大位置 */
  for (int b = 0; b < B; b++) {
    if (lens[b] < 0 || base_pos[b] < 0 || base_pos[b] + lens[b] + 1 > g->ctx) {
      MLX_ERR("decode_batch 越界：b=%d base=%d len=%d ctx=%d", b, base_pos[b], lens[b], g->ctx);
      return -1;
    }
    int end = base_pos[b] + lens[b] + 1;
    if (end > Smax) Smax = end;
    if (base_pos[b] < span_lo) span_lo = base_pos[b];
    if (end > span_hi) span_hi = end;
    if (lens[b] + 1 > maxlen) maxlen = lens[b] + 1;
  }

  /* ---- 紧凑化：段大、序列短时，直接 slice [0, Smax) 会把整段跨度都扫一遍。
   * 实测（4-bit，Qwen2.5-0.5B，B=4，lens≈100）：
   *   seg=256  → Smax=896   → 25.5 ms/步（91 tok/s）
   *   seg=4096 → Smax=12416 → 137 ms/步（26 tok/s）
   * 差 5.4 倍，全是白扫的行。
   * 所以当跨度远大于实际需要的长度时，用 take 把每个序列的活跃行 gather 成
   * 一个紧凑张量 [B, maxlen, KVH, D]：搬运量 B*maxlen*KVH*D（≈100KB/层）
   * 远小于扫 span（≈1.6MB/层）。判定阈值取 2x，避免小段时徒增开销。*/
  const int span = span_hi - span_lo;
  static int compact_on = -1; /* SGL_MLX_COMPACT=0 关掉（对照用，见 NOTES_mlx_q4.md）*/
  if (compact_on < 0) {
    const char *e = getenv("SGL_MLX_COMPACT");
    compact_on = (e && *e) ? atoi(e) : 1;
  }
  const bool compact = compact_on && (B > 1) && (maxlen > 0) && (span > 2 * maxlen + 64);
  std::vector<int32_t> gidx;
  if (compact) {
    gidx.resize((size_t)B * maxlen);
    for (int b = 0; b < B; b++)
      for (int p = 0; p < maxlen; p++) {
        int q = base_pos[b] + p;
        /* 超出该序列实际长度的行指向自己最后一个有效行（否则会读到别的序列），
         * 反正 mask 会把它们屏蔽掉。*/
        if (q >= base_pos[b] + lens[b] + 1) q = base_pos[b] + lens[b];
        gidx[(size_t)b * maxlen + p] = q;
      }
  }
  /* 把 Smax 对齐到固定档位（SGL_MLX_BUCKET，默认 32）。
   *
   * 为什么：MLX 是 lazy + JIT，**kernel 按 shape 特化**。decode 每步 lens 长 1，
   * 若 Smax 每步都不同，则每层每步都要重新选/编 kernel、重新做图规划 ——
   * 实测这是单步耗时的大头（见 docs/NOTES_mlx_t2.md 的 AB 矩阵：
   * 不写 KV、不重复 K/V 都几乎不省，但把所有 shape 固定后就降得下来）。
   * mask 对 [lens+1, Smax) 本来就会给出 -inf，所以多算的行不影响结果，只会跑一次
   * 便宜的空注意力。默认 32：max 32 个 kernel 变体、每个只编一次。
   * 设 SGL_MLX_BUCKET=1 回到旧行为（对照用）。*/
  {
    static int bucket = -1;
    if (bucket < 0) {
      const char *e = getenv("SGL_MLX_BUCKET");
      bucket = (e && *e) ? atoi(e) : 32;
      if (bucket < 1) bucket = 1;
    }
    if (bucket > 1) {
      int s2 = ((Smax + bucket - 1) / bucket) * bucket;
      if (s2 > g->ctx) s2 = g->ctx;
      if (s2 > Smax) Smax = s2;
    }
  }

  auto idx = mx::array(tokens, mx::Shape{B}, mx::int32);
  /* SGL_MLX_PROF=1：把每次 decode_batch 的入参和时间打出来。
   * 服务路径与 bench 的数字对不上时，先看这里：是 Smax 变大了、B 不是 1，
   * 还是同一组入参本身就慢（那就是线程/环境的事）。*/
  static int prof = -1;
  if (prof < 0) {
    const char *e = getenv("SGL_MLX_PROF");
    prof = (e && *e) ? atoi(e) : 0;
  }
  struct timespec tp0;
  if (prof) clock_gettime(CLOCK_MONOTONIC, &tp0);
  auto x = emb_lookup(g, idx); /* [B,E] */
  if (g->emb_softcap > 0)
    x = mx::multiply(mx::array(g->emb_softcap),
                     mx::tanh(mx::divide(x, mx::array(g->emb_softcap))));
  if (g->gemma_scale > 0) x = mx::multiply(x, mx::array(g->gemma_scale));

  /* 注意力的序列长度：紧凑时 = maxlen，否则 = Smax。mask 与 KV 都按它对齐。*/
  const int S = compact ? maxlen : Smax;

  std::vector<int> pos((size_t)B);
  for (int b = 0; b < B; b++) pos[(size_t)b] = lens[b];
  std::vector<float> cs, sn;

  /* mask：对序列 b，相对位置 p 可用 ⇔ rel_start[b] <= p <= rel_start[b]+lens[b]
   * （紧凑模式里 base 已经被 gather 掉，所以 rel_start=0）。*/
  const float NEG = -1e30f;
  std::vector<float> mk((size_t)B * S, 0.f);
  const float inv_sqrtD = 1.0f / std::sqrt((float)D);
  auto rel_start = [&](int b) { return compact ? 0 : base_pos[b]; };
  for (int b = 0; b < B; b++) {
    int upto = lens[b] + 1;
    int from = rel_start(b);
    float *r = mk.data() + (size_t)b * S;
    for (int p = 0; p < S; p++) {
      int rel = p - from;
      r[p] = (rel >= 0 && rel < upto) ? 0.f : NEG;
    }
  }
  /* 逐层窗口不同（gemma-2）时，decode 这一行也要按窗口收窄：单独缓存 */
  std::vector<int> wins_seen;
  std::vector<mx::array> masks;
  auto get_mask = [&](int win) -> const mx::array & {
    for (size_t i = 0; i < wins_seen.size(); i++)
      if (wins_seen[i] == win) return masks[i];
    std::vector<float> m2 = mk;
    if (win > 0) {
      for (int b = 0; b < B; b++) {
        int upto = lens[b] + 1;
        int from = upto > win ? upto - win : 0;
        float *r = m2.data() + (size_t)b * S;
        int rs = rel_start(b);
        for (int p = 0; p < S; p++) {
          int rel = p - rs;
          if (rel < from) r[p] = NEG;
        }
      }
    }
    wins_seen.push_back(win);
    masks.push_back(mx::array(m2.data(), mx::Shape{B, 1, 1, S}, mx::float32));
    return masks.back();
  };

  for (int l = 0; l < g->L; l++) {
    auto h = mx::fast::rms_norm(x, g->inp_norm[l], g->eps);
    auto qkv = proj(h, g->qkv_w[l]); /* [B, qdim+2kvdim]，拼成一份权重（见 forward_impl）*/
    if (g->qkv_b[l]) qkv = mx::add(qkv, *g->qkv_b[l]);
    const int QD = g->qdim, KD = g->kvdim;
    auto q = mx::slice(qkv, mx::Shape{0, 0}, mx::Shape{B, QD});
    auto k = mx::slice(qkv, mx::Shape{0, QD}, mx::Shape{B, QD + KD});
    auto v = mx::slice(qkv, mx::Shape{0, QD + KD}, mx::Shape{B, QD + 2 * KD});

    if (g->q_norm[l]) {
      auto qh = mx::fast::rms_norm(mx::reshape(q, mx::Shape{B * H, D}), *g->q_norm[l], 1e-6f);
      q = mx::reshape(qh, mx::Shape{B, g->qdim});
    }
    if (g->k_norm[l]) {
      auto kh = mx::fast::rms_norm(mx::reshape(k, mx::Shape{B * KVH, D}), *g->k_norm[l], 1e-6f);
      k = mx::reshape(kh, mx::Shape{B, g->kvdim});
    }

    auto q4 = mx::reshape(q, mx::Shape{B, H, 1, D});
    auto k4 = mx::reshape(k, mx::Shape{B, KVH, 1, D});
    auto v4 = mx::reshape(v, mx::Shape{B, KVH, 1, D});

    /* fast::rope 的 offset 只能是标量或长度 1 的向量，所以只在「所有序列同一位置」
     * 时命中（B=1 的主路径正好如此）；批量 decode 里位置各行不同 → 回退手写。*/
    {
      int same = 1;
      for (int b = 1; b < B; b++)
        if (pos[(size_t)b] != pos[0]) same = 0;
      if (g->rope_base[l] > 0 && same) {
        q4 = rope_fast(g, l, q4, D, pos[0]);
        k4 = rope_fast(g, l, k4, D, pos[0]);
      } else {
        rope_cs(g, l, pos.data(), B, cs, sn);
        mx::Shape cs_shape{B, 1, 1, g->rot[l] / 2};
        q4 = rope_apply(q4, mx::Shape{B, H, 1, D}, D, g->rot[l], cs, sn, cs_shape);
        k4 = rope_apply(k4, mx::Shape{B, KVH, 1, D}, D, g->rot[l], cs, sn, cs_shape);
      }
    }

    /* 写 KV：B 个散点行 */
    auto krows = mx::reshape(mx::transpose(k4, std::vector<int>{0, 2, 1, 3}),
                             mx::Shape{B, KVH, D});
    auto vrows = mx::reshape(mx::transpose(v4, std::vector<int>{0, 2, 1, 3}),
                             mx::Shape{B, KVH, D});
    if (g->kdt != g->wdt) {
      krows = mx::astype(krows, g->kdt);
      vrows = mx::astype(vrows, g->kdt);
    }
    /* 不在这里 eval：krows/vrows 与 slice_update 有自然依赖，MLX 会自己排序。
     * 原来是 krows.eval(); vrows.eval(); —— 每层两次强制同步，会打断 MLX 的异步
     * 流水（24 层 = 48 个同步点/步）。实测去掉后 decode 从 45.4 → 41.2 ms/步。*/
    for (int b = 0; b < B && !(ab_mask() & 2); b++) {
      int row = base_pos[b] + lens[b];
      auto kb1 = mx::slice(krows, mx::Shape{b, 0, 0}, mx::Shape{b + 1, KVH, D});
      auto vb1 = mx::slice(vrows, mx::Shape{b, 0, 0}, mx::Shape{b + 1, KVH, D});
      g->kc[l] = mx::slice_update(g->kc[l], kb1, mx::Shape{row, 0, 0},
                                  mx::Shape{row + 1, KVH, D});
      g->vc[l] = mx::slice_update(g->vc[l], vb1, mx::Shape{row, 0, 0},
                                  mx::Shape{row + 1, KVH, D});
    }

    /* mx::array 没有默认构造函数（见前面 arr_from 的注释），所以两个分支都得在
     * 初始化表达式里生成，不能声明后再赋值。*/
    auto kv = [&]() -> std::pair<mx::array, mx::array> {
      if (compact) {
        /* 只取活跃行：[B, maxlen, KVH, D]。搬运 B*maxlen*KVH*D*2 字节/层，
         * 而扫 span 要 span*KVH*D*2 —— 典型 4 倍到几十倍差距。*/
        auto kidx = mx::array(gidx.data(), mx::Shape{B, maxlen}, mx::int32);
        mx::array k = mx::take(g->kc[l], kidx, 0);
        mx::array v = mx::take(g->vc[l], kidx, 0);
        if (g->kdt != g->wdt) {
          k = mx::astype(k, g->wdt);
          v = mx::astype(v, g->wdt);
        }
        return {mx::transpose(k, std::vector<int>{0, 2, 1, 3}),
                mx::transpose(v, std::vector<int>{0, 2, 1, 3})};
      }
      auto Kf = mx::slice(g->kc[l], mx::Shape{0, 0, 0}, mx::Shape{Smax, KVH, D});
      auto Vf = mx::slice(g->vc[l], mx::Shape{0, 0, 0}, mx::Shape{Smax, KVH, D});
      if (g->kdt != g->wdt) {
        Kf = mx::astype(Kf, g->wdt);
        Vf = mx::astype(Vf, g->wdt);
      }
      return {mx::transpose(mx::reshape(Kf, mx::Shape{1, Smax, KVH, D}),
                            std::vector<int>{0, 2, 1, 3}),
              mx::transpose(mx::reshape(Vf, mx::Shape{1, Smax, KVH, D}),
                            std::vector<int>{0, 2, 1, 3})};
    }();
    mx::array K4 = kv.first, V4 = kv.second;
    /* 注意力：mx::fast::scaled_dot_product_attention 一步做完 qk+scale+mask+softmax+av。
     * MLX 的 sdpa 自己处理 GQA（q 的 H 与 k 的 kv_heads 不同就内部重复），不用
     * repeat 物化一份 [H,Smax,D]；也不用把 [B,H,1,Smax] 的分数写回显存。
     * 原来是 matmul+multiply+add+softmax+astype+matmul+repeat×2 = 7 个 op，
     * 现在 1 个（加 1 个 mask 构造）。**默认关闭**（SGL_MLX_SDPA=1 可开）：实测在本
     * 后端的主要 shape（Smax<200、D=64、H=14）下 sdpa 没有速度优势（55.1 → 57.7 ms/步，
     * 反而略慢），而 fp16 下与手写 softmax 路径的舍入差到 2.2e-3（qwen fp16 会挂 [4]）。
     * 没收益就不引入第二条精度路径；留开关供以后 MLX 改善了再试。*/
    static int use_sdpa = -1;
    if (use_sdpa < 0) {
      const char *e = getenv("SGL_MLX_SDPA");
      use_sdpa = (e && *e) ? atoi(e) : 0;
    }
    mx::array attn = [&]() -> mx::array {
      if (use_sdpa) {
        /* 签名（fast.h:46）：sdpa(q, k, v, scale, mask_mode, mask_arrs, sinks, stream)
         * mask_mode="array" + mask_arrs={加性 mask}。sdpa 自己处理 GQA。
         * mask 必须能 promote 到输出类型（否则："Mask type must promote to
         * output type float16"），所以这里转成 wdt。*/
        std::vector<mx::array> ma{mx::astype(get_mask(g->layer_window[l]), g->wdt)};
        /* sdpa 要求 q 的 batch 维与 k/v 一致（否则："mismatching batch dimension"）。
         * B>1 时 K/V 还是 [1,KVH,Smax,D]，broadcast 成 view 就行（不拷贝）。*/
        auto Kb = (B > 1) ? mx::broadcast_to(K4, mx::Shape{B, KVH, Smax, D}) : K4;
        auto Vb = (B > 1) ? mx::broadcast_to(V4, mx::Shape{B, KVH, Smax, D}) : V4;
        auto o = mx::fast::scaled_dot_product_attention(q4, Kb, Vb, inv_sqrtD, "array", ma);
        return mx::reshape(o, mx::Shape{B, g->qdim});
      }
      auto qh = q4;
      if (g->rep > 1) {
        K4 = mx::repeat(K4, g->rep, 1);
        V4 = mx::repeat(V4, g->rep, 1);
      }
      /* 同 forward_impl：分数 f32（见那里的注释）*/
      auto sc = mx::matmul(mx::astype(qh, mx::float32),
                           mx::transpose(mx::astype(K4, mx::float32),
                                         std::vector<int>{0, 1, 3, 2})); /* [B,H,1,S] f32 */
      sc = mx::multiply(sc, mx::array(inv_sqrtD));
      sc = mx::add(sc, get_mask(g->layer_window[l]));
      auto aw = mx::astype(mx::softmax(sc, std::vector<int>{-1}, true), g->wdt);
      auto ao = mx::matmul(aw, V4); /* [B,H,1,D] */
      return mx::reshape(ao, mx::Shape{B, g->qdim});
    }();

    h = proj(attn, g->o_w[l]);
    x = mx::add(x, h);

    auto h2 = mx::fast::rms_norm(x, g->post_norm[l], g->eps);
    /* gate/up 也拼成一份：一次 matmul 后切两半 */
    auto gu = proj(h2, g->gu_w[l]); /* [B, 2I] */
    auto gt = mx::slice(gu, mx::Shape{0, 0}, mx::Shape{B, g->I});
    auto up = mx::slice(gu, mx::Shape{0, g->I}, mx::Shape{B, 2 * g->I});
    auto act = mx::multiply(mx::multiply(gt, mx::sigmoid(gt)), up);
    x = mx::add(x, proj(act, g->down_w[l]));
  }

  auto hf = mx::fast::rms_norm(x, *g->final_norm, g->eps);
  if (ab_mask() & 1) { /* AB：不算 lm_head */
    memset(logits_out, 0, (size_t)B * V * sizeof(float));
  } else {
    auto lg = mx::astype(head(g, hf), mx::float32); /* [B,V] */
    lg.eval();
    memcpy(logits_out, lg.data<float>(), (size_t)B * V * sizeof(float));
  }
  if (prof) {
    struct timespec tp1;
    clock_gettime(CLOCK_MONOTONIC, &tp1);
    double ms = (tp1.tv_sec - tp0.tv_sec) * 1e3 + (tp1.tv_nsec - tp0.tv_nsec) / 1e6;
    fprintf(stderr, "[mlxprof] B=%d Smax=%d base0=%d lens0=%d  %.2f ms\n", B, Smax,
            base_pos[0], lens[0], ms);
  }
  return 0;
}

/* ============================ 前缀缓存搬运 ============================ */

int sgl_mlx_copy_kv(sgl_mlx_model *g, int src, int dst, int len) {
  if (!g) return -1;
  if (src == dst || len <= 0) return 0;
  const int KVH = g->KVH, D = g->D;
  if (src < 0 || dst < 0 || len <= 0 || src + len > g->ctx || dst + len > g->ctx) {
    MLX_ERR("copy_kv 越界：src=%d dst=%d len=%d ctx=%d", src, dst, len, g->ctx);
    return -1;
  }
  for (int l = 0; l < g->L; l++) {
    /* 先取源块（惰性 slice 持有旧数组的引用，下面重赋值不会让它失效）*/
    auto kb = mx::slice(g->kc[l], mx::Shape{src, 0, 0}, mx::Shape{src + len, KVH, D});
    auto vb = mx::slice(g->vc[l], mx::Shape{src, 0, 0}, mx::Shape{src + len, KVH, D});
    g->kc[l] = mx::slice_update(g->kc[l], kb, mx::Shape{dst, 0, 0},
                                mx::Shape{dst + len, KVH, D});
    g->vc[l] = mx::slice_update(g->vc[l], vb, mx::Shape{dst, 0, 0},
                                mx::Shape{dst + len, KVH, D});
  }
  for (int l = 0; l < g->L; l++) {
    g->kc[l].eval();
    g->vc[l].eval();
  }
  return 0;
}

/* ============================ 预热 ============================
 * MLX 的 kernel 是首次用到某形状时才编（JIT），第一次请求会明显慢。
 * 起服务前用假 token 把「最大 prefill 块」与「最大 batch 的 decode」各跑一遍。*/
int sgl_mlx_prewarm(sgl_mlx_model *g, int rows, int max_b) {
  if (!g) return -1;
  int done = 0;
  if (rows > 0) {
    int n = rows < g->ctx ? rows : g->ctx;
    if (n > 0) {
      std::vector<int> ids((size_t)n, 1);
      if (forward_impl(g, NULL, ids.data(), n, 0, 0, NULL, 0) == 0) done++;
    }
  }
  if (max_b > 0 && g->ctx >= 2 * max_b) {
    int B = max_b, seg = g->ctx / B;
    std::vector<int> tk((size_t)B, 1), bp((size_t)B, 0), ln((size_t)B, 0);
    for (int b = 0; b < B; b++) bp[(size_t)b] = b * seg;
    std::vector<float> lg((size_t)B * g->V);
    if (sgl_mlx_decode_batch(g, tk.data(), bp.data(), ln.data(), B, lg.data()) == 0) done++;
  }
  printf("[mlx] prewarm: rows=%d max_b=%d → %d 项\n", rows, max_b, done);
  fflush(stdout);
  return done > 0 ? done : -1;
}

/* ============================ 内存预算 ============================ */

long sgl_mlx_estimate_mem_mb(const void *llama_model, int ctx, int rows) {
  const sgl_llama *m = (const sgl_llama *)llama_model;
  if (!m) return -1;
  const int fp16 = env_int("SGL_MLX_FP16", 1);
  const int kv16 = env_int("SGL_MLX_KV16", fp16);
  const double wb = fp16 ? 2.0 : 4.0, kb = kv16 ? 2.0 : 4.0;
  const double L = m->n_layer, E = m->n_embd, I = m->n_inter, V = m->vocab;
  const double H = m->n_head, KVH = m->n_kv_head, D = m->head_dim;
  /* 权重元素数：wte + lm_head + 每层 q/k/v/o + gate/up/down + norm（小，按 4E/层估）
   * tie 时 lm_head 与 wte 共享同一份；wt_on 时另存转置副本，所以还是两份。
   * 量化检查点（有 model.safetensors.index.json 或 mxw 已开）时权重按 4bit 算，
   * 且 tie 不再物化稠密副本（wte_q/lm_head_q 共享），embed 也不解量化。*/
  const int quant = g_mxw != NULL;
  const double qpb = quant ? 0.55 : wb; /* 4bit+scale+bias ≈ 0.55 字节/参数 */
  double w = V * E + ((m->tie_embeddings && (!wt_on() || quant)) ? 0.0 : V * E) +
             L * (H * D * E + 2.0 * KVH * D * E + E * H * D) + L * 3.0 * I * E + L * 4.0 * E;
  double kv = L * 2.0 * (ctx > 0 ? ctx : m->n_ctx) * KVH * D;
  /* 激活：x/q/k/v/MLP 中间量 + 注意力分数（fp32，H*rows*ctx 是最大的一块）+ logits */
  double r = rows > 0 ? rows : 1;
  double act = r * (E + H * D + 2.0 * KVH * D + 2.0 * I) * wb + H * r * (ctx > 0 ? ctx : 1) * 4.0;
  double lg = r * V * 4.0;
  return (long)((w * qpb + kv * kb + act + lg) / 1048576.0);
}
