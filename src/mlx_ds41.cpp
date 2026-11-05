/* mlx_ds41.cpp — DeepSeek-V4.1 mini 金标准的 MLX 前向（对拍用，逐步转录 mini_ref.py）。
 *
 * 语义来源：tools/dsv41_ref/mini_ref.py（官方 inference/{model,kernel,engram}.py 的纯 torch 镜像）。
 * 同一计算图、全 f32；假量化（fp8/fp4）用 mx 算子位精确重写：
 *   - ceil(log2 x)   ≡ 位语义（floor + (lg>floor)）
 *   - 2^e            ≡ 位构造（(e+127)<<23 再 view 成 f32），避免 expf 舍入
 *   - e4m3/e2m1 cast ≡ 手写网格舍入（banker's，与 torch cast 一致）
 * 权重/输入/频率全部经 safetensors 读入（绕开 host 内存造 array 的坑）。
 *
 * 金标准循环（mirror main()）：prefill(ids[0:12]) 存 logits_pre；随后 i=12..15：
 *   喂 toks[i-1]、start_pos=i-1 → logits_i、toks[i]=argmax。
 *   注意第 0 步喂的是最后一个 prefill token、start_pos=11（复算位置 11，与 mirror 一致）。
 */
#include "sgl/ds41.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mlx/array.h"
#include "mlx/io.h"
#include "mlx/ops.h"

using namespace mlx::core;
namespace mx = mlx::core;

namespace {

/* ---- mini 配置（硬编码；对应 mini_ref.CFG）---- */
constexpr int V = 256, D = 64, L = 2, H = 8, DH = 32, RD = 8, G = 2, RORD = 16;
constexpr int WIN = 16;
constexpr int R0 = 2;                       /* compress_ratios[0]；layer1 = 0 */
constexpr float HC_EPS = 1e-6f, N_EPS = 1e-20f;
constexpr int E = 8, TOPK = 2;
constexpr float SWIGLU_LIM = 10.0f, GATE_T = 1.0f, ROUTE_SCALE = 1.5f;
constexpr int IH = 4, IHD = 32, ITOPK = 8;
constexpr int HC = 2, SN_ITERS = 5;
constexpr int MAXSEQ = 64, S_IN = 12, B = 2;
constexpr float FP8_MAX = 448.0f, FP4_MAX = 6.0f;
constexpr float ATTN_SCALE = 0.17677669529663689f;      /* 32^-0.5 */
constexpr float IDX_W_SCALE = 0.08838834764831845f;     /* 32^-0.5 · 4^-0.5 */
constexpr float E4M3_SUBN_FLOOR = 1.1718875e-2f;        /* 6·2^-9 */

std::unordered_map<std::string, mx::array> W;
/* MLX 的 array 无默认构造 → 全局状态用 optional 存，访问函数取引用 */
static std::optional<mx::array> o_wcos, o_wsin, o_ccos, o_csin, o_ids;
static std::optional<mx::array> o_wc[2], o_cc, o_ic, o_ks, o_ss;
inline mx::array& f_win_cos() { return *o_wcos; }
inline mx::array& f_win_sin() { return *o_wsin; }
inline mx::array& f_cmp_cos() { return *o_ccos; }
inline mx::array& f_cmp_sin() { return *o_csin; }
inline mx::array& ids_dev() { return *o_ids; }
inline mx::array& cmp_cache() { return *o_cc; }
inline mx::array& idx_cache() { return *o_ic; }
inline mx::array& kv_state() { return *o_ks; }
inline mx::array& score_state() { return *o_ss; }
inline mx::array& win_cache(int i) { return *o_wc[i]; }
int mult0[3]; /* engram multipliers（host 哈希用，从 st 读） */
int prime0[2];

int id_hist[B][MAXSEQ];

mx::array w(const std::string& n) {
  auto it = W.find(n);
  if (it == W.end()) {
    fprintf(stderr, "[ds41] missing weight: %s\n", n.c_str());
    exit(2);
  }
  return it->second;
}

static int tr_on() {
  static int v = getenv("DS41_TRACE") ? 1 : 0;
  return v;
}
static void tr(const char* tag, const mx::array& a) {
  if (!tr_on()) return;
  auto sm = mx::sum(a);
  auto am = mx::max(mx::abs(a));
  sm.eval();
  am.eval();
  std::string sh;
  for (int d : a.shape()) sh += std::to_string(d) + ",";
  fprintf(stderr, "[cpp:%s] shape=[%s] sum=%.6e amax=%.6e\n", tag, sh.c_str(), sm.item<float>(), am.item<float>());
}
mx::array f32(const mx::array& a) { return mx::astype(a, mx::float32); }
mx::array fa(float v) { return mx::array(v); }
mx::array ia(int v) { return mx::array(v, mx::int32); }

mx::array clipS(const mx::array& a, float lo, float hi) {
  return mx::minimum(mx::maximum(a, fa(lo)), fa(hi));
}

/* ---------- 量化原语（位精确） ---------- */

/* 就近偶数舍入（torch.round 语义） */
mx::array rt_even(const mx::array& x) {
  auto r = mx::floor(x);
  auto f = x - r;
  auto rmod2 = r - mx::floor(r / fa(2.0f)) * fa(2.0f);
  auto inc = mx::logical_or(f > fa(0.5f), mx::logical_and(f == fa(0.5f), rmod2 == fa(1.0f)));
  return r + mx::astype(inc, mx::float32);
}

/* 2^e（e int32 数组）：位构造，精确 */
mx::array pow2_bits(const mx::array& e_i32) {
  auto bits = (e_i32 + ia(127)) * ia(8388608);
  return mx::view(bits, mx::float32);
}

/* ceil(log2 x) ≡ kernel.fast_log2_ceil */
mx::array ceil_log2(const mx::array& x) {
  auto lg = mx::log2(x);
  auto fl = mx::floor(lg);
  return fl + mx::astype(lg > fl, mx::float32);
}

mx::array round_scale(const mx::array& amax, float qmax) {
  return pow2_bits(mx::astype(ceil_log2(amax / fa(qmax)), mx::int32));
}

/* fp8 e4m3 网格 cast（RN 含亚常规），镜像 q_e4m3（torch float8_e4m3fn cast） */
mx::array q_e4m3(const mx::array& x) {
  auto a = mx::abs(x);
  auto sub = rt_even(a * fa(512.0f)) * fa(1.0f / 512.0f);
  auto lg = mx::floor(mx::log2(mx::maximum(a, fa(1e-30f))));
  auto e = mx::astype(lg, mx::int32);
  auto m = a * pow2_bits(ia(3) - e);
  auto qm = rt_even(m);
  /* m∈[8,16)：舍入到 16 时进位为 (1.000, e+1) = 2^(e+1) */
  auto nrm = mx::where(qm == fa(16.0f), pow2_bits(e + ia(1)), qm * pow2_bits(e - ia(3)));
  auto out = mx::where(lg >= fa(-6.0f), nrm, sub);
  return mx::where(x < fa(0.0f), -out, out);
}

/* fp4 e2m1 网格 cast，镜像 q_e2m1 */
mx::array q_e2m1(const mx::array& x) {
  auto a = mx::minimum(mx::abs(x), fa(FP4_MAX));
  auto sub = rt_even(a * fa(2.0f)) * fa(0.5f); /* 网格 {0,.5,1,1.5} */
  auto three = mx::logical_and(a >= fa(2.5f), a < fa(3.5f));
  auto four = mx::logical_and(a >= fa(3.5f), a < fa(5.0f));
  auto six = a >= fa(5.0f);
  auto nrm = fa(2.0f) + astype(three, mx::float32) * fa(1.0f) +
             astype(four, mx::float32) * fa(2.0f) + astype(six, mx::float32) * fa(4.0f);
  auto out = mx::where(a >= fa(2.0f), nrm, sub);
  return mx::where(x < fa(0.0f), -out, out);
}

/* fp8 假量化：末维 32 分块 + ue8m0 scale（镜像 act_quant_dequant） */
mx::array act_quant_dq(const mx::array& x0) {
  int last = (int)x0.shape().back();
  auto x = mx::reshape(f32(x0), {-1, last});
  int n = x.shape(0);
  auto g = mx::reshape(x, {n, last / 32, 32});
  auto amax = mx::maximum(mx::max(mx::abs(g), {-1}, true), fa(1e-4f));
  auto s = round_scale(amax, FP8_MAX);
  if (tr_on()) {
    auto ssm = mx::sum(s), smx = mx::max(mx::abs(s));
    ssm.eval();
    smx.eval();
    fprintf(stderr, "[cpp:qscale] sum=%.6e amax=%.6e\n", ssm.item<float>(), smx.item<float>());
  }
  auto q = q_e4m3(clipS(g / s, -FP8_MAX, FP8_MAX));
  if (tr_on()) {
    auto qsm = mx::sum(q);
    qsm.eval();
    float f8[8];
    auto q8 = mx::reshape(mx::slice(mx::reshape(q, {(int)q.size()}), Shape{0}, Shape{8}), {8});
    q8.eval();
    memcpy(f8, q8.data<float>(), sizeof f8);
    fprintf(stderr, "[cpp:qq] sum=%.6e first8=[%g,%g,%g,%g,%g,%g,%g,%g]\n", qsm.item<float>(), f8[0], f8[1], f8[2],
            f8[3], f8[4], f8[5], f8[6], f8[7]);
  }
  return mx::reshape(q * s, x0.shape());
}

/* fp4 假量化（镜像 fp4_act_quant_dequant）；e4m3_scale=false → e8m0 */
mx::array fp4_act_quant_dq(const mx::array& x0, int bs, bool e4m3_scale) {
  int last = (int)x0.shape().back();
  auto x = mx::reshape(f32(x0), {-1, last});
  int n = x.shape(0);
  auto g = mx::reshape(x, {n, last / bs, bs});
  auto amax = mx::max(mx::abs(g), {-1}, true);
  mx::array s = e4m3_scale ? q_e4m3(mx::maximum(amax, fa(E4M3_SUBN_FLOOR)) / fa(FP4_MAX))
                           : round_scale(mx::maximum(amax, fa(6.0f * 1.1754944e-38f)), FP4_MAX);
  auto q = q_e2m1(clipS(g / s, -FP4_MAX, FP4_MAX));
  return mx::reshape(q * s, x0.shape());
}

/* ---------- 基础件 ---------- */

mx::array rms_norm(const mx::array& x, const mx::array& wt) {
  auto xf = f32(x);
  auto var = mx::mean(mx::square(xf), {-1}, true);
  return wt * (xf * mx::rsqrt(var + fa(N_EPS)));
}

mx::array linear(const mx::array& x, const mx::array& wt) { return mx::matmul(f32(x), mx::transpose(f32(wt), {1, 0})); }

/* Shape 拼接（SmallVector 无 operator+） */
static Shape shp_cat(Shape a, const Shape& b) {
  for (auto v : b) a.push_back(v);
  return a;
}
/* 最后一维取 [s0,s1) 的全秩 start/stop */
static std::pair<Shape, Shape> last_slice(const mx::array& a, int s0, int s1) {
  Shape st, sp = a.shape();
  for (int i = 0; i < a.ndim(); i++) st.push_back(0);
  st[a.ndim() - 1] = s0;
  sp[a.ndim() - 1] = s1;
  return {st, sp};
}
/* 读第 i 个元素（任意形状，先展平；这版 MLX 的 item<T>() 仅限 size-1） */
static int item_i(const mx::array& a, int i) {
  auto flat = mx::reshape(a, {(int)a.size()});
  auto e = mx::reshape(mx::slice(flat, Shape{i}, Shape{i + 1}), {});
  return e.item<int>();
}
static float item_f(const mx::array& a, int i) {
  auto flat = mx::reshape(a, {(int)a.size()});
  auto e = mx::reshape(mx::slice(flat, Shape{i}, Shape{i + 1}), {});
  return e.item<float>();
}

/* RoPE：x [B,S,rd]（3D）或 [B,S,H,rd]（4D）；cs/sn 已切好 [S,rd/2] */
mx::array rotary(const mx::array& x, const mx::array& cs, const mx::array& sn, bool inv) {
  auto full = x.shape();
  int nd = x.ndim();
  int rd = full[nd - 1];
  Shape mid(full.begin(), full.end() - 1);
  auto pair = mx::reshape(x, shp_cat(mid, {rd / 2, 2}));
  auto sl0 = last_slice(pair, 0, 1); /* [..., rd/2, 1] */
  auto x0 = mx::reshape(mx::slice(pair, sl0.first, sl0.second), shp_cat(mid, {rd / 2}));
  auto sl1 = last_slice(pair, 1, 2);
  auto x1 = mx::reshape(mx::slice(pair, sl1.first, sl1.second), shp_cat(mid, {rd / 2}));
  auto s = inv ? fa(-1.0f) : fa(1.0f);
  if (nd == 3 && getenv("DS41_DBG")) {
    fprintf(stderr, "[rotary3] x0=[%d,%d,%d] cs=%d\n", (int)x0.shape(0), (int)x0.shape(1), (int)x0.shape(2),
            (int)cs.shape(0));
  }
  if (nd == 4 && getenv("DS41_DBG")) {
    fprintf(stderr, "[rotary4] x0=[%d,%d,%d,%d] cs=%d\n", (int)x0.shape(0), (int)x0.shape(1), (int)x0.shape(2),
            (int)x0.shape(3), (int)cs.shape(0));
  }
  auto cB = (nd == 3) ? mx::broadcast_to(mx::reshape(cs, {1, cs.shape(0), cs.shape(1)}), x0.shape())
                      : mx::broadcast_to(mx::reshape(cs, {1, cs.shape(0), 1, cs.shape(1)}), x0.shape());
  auto sB = (nd == 3) ? mx::broadcast_to(mx::reshape(f32(sn) * s, {1, sn.shape(0), sn.shape(1)}), x0.shape())
                      : mx::broadcast_to(mx::reshape(f32(sn) * s, {1, sn.shape(0), 1, sn.shape(1)}), x0.shape());
  auto y0 = x0 * cB - x1 * sB;
  auto y1 = x0 * sB + x1 * cB;
  return mx::reshape(mx::stack({y0, y1}, -1), full);
}

/* 把 x 最后 RD 维替换为旋转版（镜像 x[..., -rd:] = rotary(...)） */
mx::array rotary_tail(const mx::array& x, const mx::array& cs, const mx::array& sn, bool inv) {
  auto full = x.shape();
  int last = full[full.size() - 1];
  auto sl_h = last_slice(x, 0, last - RD);
  auto x_head = mx::slice(x, sl_h.first, sl_h.second);
  auto sl_t = last_slice(x, last - RD, last);
  auto x_tail = mx::slice(x, sl_t.first, sl_t.second);
  return mx::concatenate({x_head, rotary(x_tail, cs, sn, inv)}, x.ndim() - 1);
}

mx::array slice_win_freqs(const mx::array& tbl, int sp, int s) {
  return mx::slice(tbl, Shape{sp, 0}, Shape{sp + s, RD / 2});
}
mx::array slice_cmp_freqs(const mx::array& tbl, int sp, int s) {
  if (sp == 0) {
    int cnt = std::max(s - s % R0, 0) / R0;
    /* MLX slice：stop 按索引算，带 stride R0 时取 cnt 个元素需要 stop=cnt*R0 */
    return mx::slice(tbl, Shape{0, 0}, Shape{cnt * R0, RD / 2}, Shape{R0, 1});
  }
  return mx::slice(tbl, Shape{sp + 1 - R0, 0}, Shape{sp + 2 - R0, RD / 2});
}

/* ---------- sparse attention（镜像 sparse_attn_mirror） ---------- */
mx::array sparse_attn(const mx::array& q, const mx::array& kv, const mx::array& sink, const mx::array& idxs) {
  int b = q.shape(0), s = q.shape(1), h = q.shape(2);
  int n = kv.shape(1), t = idxs.shape(2);
  auto idx = mx::maximum(idxs, ia(0));
  auto flat = idx + mx::reshape(mx::arange(0, b, 1, mx::int32), {b, 1, 1}) * ia(n);
  auto kvsel = mx::reshape(mx::take(mx::reshape(kv, {b * n, DH}), mx::reshape(flat, {-1}), 0), {b, s, t, DH});
  auto lg = mx::sum(mx::reshape(q, {b, s, h, 1, DH}) * mx::reshape(kvsel, {b, s, 1, t, DH}), {-1}) * fa(ATTN_SCALE);
  auto valid = mx::reshape(idxs >= ia(0), {b, s, 1, t});
  lg = mx::where(valid, lg, fa(-1e30f));
  auto mxv = mx::max(lg, {-1}, true);
  auto hasv = mx::reshape(mx::max(idxs, {-1}, true) >= ia(0), {b, s, 1, 1});
  mxv = mx::where(hasv, mxv, fa(-1e30f));
  auto p = mx::exp(lg - mxv);
  auto denom = mx::sum(p, {-1}, true) + mx::exp(mx::reshape(sink, {1, 1, h, 1}) - mxv);
  auto o = mx::sum(mx::reshape(p, {b, s, h, t, 1}) * mx::reshape(kvsel, {b, s, 1, t, DH}), {-2}) / denom;
  return o;
}

/* ---------- window topk 索引（镜像 get_window_topk_idxs） ---------- */
mx::array win_topk(int s, int sp) {
  if (sp == 0) {
    auto end = mx::expand_dims(mx::arange(0, s, 1, mx::int32), 1);
    int tw = std::min(s, WIN);
    auto base = mx::maximum(end - ia(WIN - 1), ia(0)) + mx::expand_dims(mx::arange(0, tw, 1, mx::int32), 0);
    auto idx = mx::where(base > end, ia(-1), base); /* [S,Tw] */
    return mx::broadcast_to(mx::reshape(idx, {1, s, tw}), {B, s, tw});
  }
  int oldest = sp % WIN + 1;
  auto tt = mx::concatenate({mx::arange(oldest, WIN, 1, mx::int32), mx::arange(0, oldest, 1, mx::int32)},
                            0);
  auto idx = mx::where(tt > ia(sp), ia(-1), tt); /* [Tw] */
  return mx::broadcast_to(mx::reshape(idx, {1, 1, (int)idx.shape(0)}), {B, 1, (int)idx.shape(0)});
}

/* ---------- Compressor（镜像 Compressor.forward，仅 kv_source 层=0） ---------- */
std::pair<mx::array, bool> compressor(const mx::array& x, int sp, int s) {
  auto kv = linear(x, w("blocks.0.attn.compressor.wkv.weight"));
  auto score = linear(x, w("blocks.0.attn.compressor.wgate.weight"));
  std::optional<mx::array> grouped;
  if (sp == 0) {
    int rem = s % R0, cut = s - rem;
    if (rem) {
      kv_state() = mx::slice_update(kv_state(), mx::reshape(mx::slice(kv, Shape{0, cut}, Shape{B, s}), {B, rem, DH}),
                                  Shape{0, 0, 0}, Shape{B, rem, DH});
      score_state() = mx::slice_update(score_state(),
                                     mx::reshape(mx::slice(score, Shape{0, cut}, Shape{B, s}), {B, rem, DH}),
                                     Shape{0, 0, 0}, Shape{B, rem, DH});
      kv = mx::slice(kv, Shape{0, 0}, Shape{B, cut});
      score = mx::slice(score, Shape{0, 0}, Shape{B, cut});
    }
    auto kvg = mx::reshape(kv, {B, cut / R0, R0, DH});
    auto scg = mx::softmax(mx::reshape(score, {B, cut / R0, R0, DH}), 2);
    grouped = mx::sum(kvg * scg, 2);
  } else {
    int slot = sp % R0;
    bool ok = (sp + 1) % R0 == 0;
    kv_state() = mx::slice_update(kv_state(), kv, Shape{0, slot, 0}, Shape{B, slot + 1, DH});
    score_state() = mx::slice_update(score_state(), score, Shape{0, slot, 0}, Shape{B, slot + 1, DH});
    if (!ok) return {kv_state(), false};
    auto sm = mx::softmax(score_state(), 1);
    grouped = mx::sum(kv_state() * sm, 1, true);
  }
  return {rms_norm(*grouped, w("blocks.0.attn.compressor.norm.weight")), true};
}

/* ---------- Indexer（镜像 Indexer.forward） ---------- */
mx::array indexer(int l, const mx::array& x, const mx::array& qr, const mx::array& latent, bool lat_ok, int sp,
                  int s, int offset) {
  std::string p = "blocks." + std::to_string(l) + ".attn.indexer.";
  int end_pos = sp + s;
  if (lat_ok) {
    auto f0 = slice_cmp_freqs(f_cmp_cos(), sp, s);
    auto f1 = slice_cmp_freqs(f_cmp_sin(), sp, s);
    auto k = rms_norm(linear(latent, w(p + "wk.weight")), w(p + "k_norm.weight"));
    k = rotary_tail(k, f0, f1, false);
    k = fp4_act_quant_dq(k, 32, false);
    int kn = (int)k.shape(1);
    idx_cache() = mx::slice_update(idx_cache(), k, Shape{0, sp / R0, 0}, Shape{B, sp / R0 + kn, IHD});
  }
  auto q = mx::reshape(linear(qr, w(p + "wq_b.weight")), {B, s, IH, IHD});
  q = rotary_tail(q, slice_win_freqs(f_win_cos(), sp, s), slice_win_freqs(f_win_sin(), sp, s), false);
  q = fp4_act_quant_dq(q, 32, false);
  int tk_len = end_pos / R0;
  auto index_k = mx::slice(idx_cache(), Shape{0, 0, 0}, Shape{B, tk_len, IHD});
  auto wt = linear(x, w(p + "weights_proj.weight")) * fa(IDX_W_SCALE); /* [B,S,IH] */
  auto sc = mx::sum(mx::reshape(q, {B, s, IH, 1, IHD}) * mx::reshape(index_k, {B, 1, 1, tk_len, IHD}), {-1});
  sc = mx::sum(mx::maximum(sc, fa(0.0f)) * mx::reshape(wt, {B, s, IH, 1}), 2); /* [B,S,T] */
  if (sp == 0) {
    auto clens = mx::floor_divide(mx::arange(1, s + 1, 1, mx::int32), ia(R0)); /* [s]： (i+1)//r */
    clens = mx::reshape(clens, {s, 1});
    auto mask = mx::reshape(mx::arange(0, s / R0, 1, mx::int32), {1, s / R0}) >= clens; /* [s,T] */
    sc = mx::where(mask, fa(-1e30f), sc); /* 尾维广播到 [B,s,T] */
  }
  int tk = std::min(ITOPK, tk_len);
  if (tr_on() && sp == 0) {
    auto r3 = mx::reshape(mx::slice(sc, Shape{0, 3, 0}, Shape{1, 4, tk_len}), {tk_len});
    r3.eval();
    fprintf(stderr, "[cpp:scrow3]=[");
    for (int i = 0; i < tk_len; i++) fprintf(stderr, "%g,", item_f(r3, i));
    fprintf(stderr, "]\n");
  }
  std::vector<mx::array> picks;
  auto cur = sc;
  auto picked = mx::zeros({B, s, tk_len}, mx::float32); /* topk 索引不重复（含平局下探） */
  for (int i = 0; i < tk; i++) {
    auto c2 = mx::where(picked > fa(0.5f), fa(-2e30f), cur); /* 已挑的比可见性 mask 更低 → 平局按序下探 */
    auto am = mx::astype(mx::argmax(c2, {-1}), mx::int32); /* [B,S] */
    picks.push_back(am);
    auto onehot = mx::astype(
        mx::reshape(mx::arange(0, tk_len, 1, mx::int32), {1, 1, tk_len}) == mx::reshape(am, {B, s, 1}),
        mx::float32);
    picked = mx::maximum(picked, onehot);
  }
  auto idxs = mx::sort(mx::stack(picks, -1), -1); /* [B,S,tk] 升序 */
  std::optional<mx::array> vis;
  if (sp == 0) {
    auto clens = mx::floor_divide(mx::arange(1, s + 1, 1, mx::int32), ia(R0)); /* [s] */
    clens = mx::reshape(clens, {s, 1});
    vis = idxs < mx::reshape(clens, {1, s, 1});
  } else {
    vis = mx::reshape(idxs < ia(tk_len), {B, s, tk});
  }
  return mx::where(*vis, idxs + ia(offset), ia(-1));
}

/* ---------- Attention ---------- */
mx::array attn_forward(int l, const mx::array& x, int sp, int s) {
  std::string p = "blocks." + std::to_string(l) + ".attn.";
  auto fc0 = slice_win_freqs(f_win_cos(), sp, s);
  auto fc1 = slice_win_freqs(f_win_sin(), sp, s);
  auto qr = rms_norm(linear(x, w(p + "wq_a.weight")), w(p + "q_norm.weight"));
  auto q = mx::reshape(linear(qr, w(p + "wq_b.weight")), {B, s, H, DH});
  q = rotary_tail(q, fc0, fc1, false);
  tr("q", q);

  auto kvn = rms_norm(linear(x, w(p + "wkv.weight")), w(p + "kv_norm.weight"));
  kvn = rotary_tail(kvn, fc0, fc1, false);
  tr("kv_rot", kvn);
  kvn = act_quant_dq(kvn);
  tr("kv_q", kvn);
  std::optional<mx::array> wkv;
  if (sp == 0) {
    if (s <= WIN) {
      win_cache(l) = mx::slice_update(win_cache(l), kvn, Shape{0, 0, 0}, Shape{B, s, DH});
      wkv = mx::slice(win_cache(l), Shape{0, 0, 0}, Shape{B, s, DH});
    } else {
      int cut = s % WIN;
      win_cache(l) = mx::slice_update(win_cache(l), mx::slice(kvn, Shape{0, s - WIN + cut}, Shape{B, s}),
                                      Shape{0, cut, 0}, Shape{B, WIN, DH});
      win_cache(l) = mx::slice_update(win_cache(l), mx::slice(kvn, Shape{0, s - cut}, Shape{B, s}), Shape{0, 0, 0},
                                      Shape{B, cut, DH});
      wkv = mx::slice(win_cache(l), Shape{0, 0, 0}, Shape{B, WIN, DH});
    }
  } else {
    int row = sp % WIN;
    win_cache(l) = mx::slice_update(win_cache(l), kvn, Shape{0, row, 0}, Shape{B, row + 1, DH});
    wkv = mx::slice(win_cache(l), Shape{0, 0, 0}, Shape{B, WIN, DH});
  }
  auto widx = win_topk(s, sp);
  tr("kv", *wkv);

  mx::array kv = *wkv, idxs = widx;
  if (l == 0) { /* kv_source=index_source=0 */
    auto cpr = compressor(x, sp, s);
    int offset = (int)wkv->shape(1);
    if (cpr.second) {
      auto latent = rotary_tail(cpr.first, slice_cmp_freqs(f_cmp_cos(), sp, s), slice_cmp_freqs(f_cmp_sin(), sp, s), false);
      latent = fp4_act_quant_dq(latent, 16, true);
      int ln = (int)latent.shape(1);
      cmp_cache() = mx::slice_update(cmp_cache(), latent, Shape{0, sp / R0, 0}, Shape{B, sp / R0 + ln, DH});
    }
    int clen = (sp + s) / R0;
    kv = mx::concatenate({*wkv, mx::slice(cmp_cache(), Shape{0, 0, 0}, Shape{B, clen, DH})}, 1);
    auto cidx = indexer(l, x, qr, cpr.first, cpr.second, sp, s, offset);
    idxs = mx::concatenate({widx, cidx}, 2);
  }

  if (tr_on()) {
    auto vc = mx::sum(idxs >= ia(0));
    vc.eval();
    int r0[24], rE[24];
    auto row0 = mx::reshape(mx::slice(idxs, Shape{0, 0, 0}, Shape{1, 1, (int)idxs.shape(2)}), {(int)idxs.shape(2)});
    auto rowE =
        mx::reshape(mx::slice(idxs, Shape{0, s - 1, 0}, Shape{1, s, (int)idxs.shape(2)}), {(int)idxs.shape(2)});
    row0.eval();
    rowE.eval();
    memcpy(r0, row0.data<int>(), sizeof(int) * idxs.shape(2));
    memcpy(rE, rowE.data<int>(), sizeof(int) * idxs.shape(2));
    fprintf(stderr, "[cpp:idxs] valid=%d t=%d\n", vc.item<int>(), (int)idxs.shape(2));
    fprintf(stderr, "[cpp:idxrow0]=[");
    for (int i = 0; i < (int)idxs.shape(2); i++) fprintf(stderr, "%d,", r0[i]);
    fprintf(stderr, "]\n[cpp:idxrowE]=[");
    for (int i = 0; i < (int)idxs.shape(2); i++) fprintf(stderr, "%d,", rE[i]);
    fprintf(stderr, "]\n");
    if (s > 6) { /* row3/row6 只在 prefill 有意义 */
      auto row3 =
          mx::reshape(mx::slice(idxs, Shape{0, 3, 0}, Shape{1, 4, (int)idxs.shape(2)}), {(int)idxs.shape(2)});
      auto row6 =
          mx::reshape(mx::slice(idxs, Shape{0, 6, 0}, Shape{1, 7, (int)idxs.shape(2)}), {(int)idxs.shape(2)});
      row3.eval();
      row6.eval();
      fprintf(stderr, "[cpp:idxrow3]=[");
      for (int i = 0; i < (int)idxs.shape(2); i++) fprintf(stderr, "%d,", item_i(row3, i));
      fprintf(stderr, "]\n[cpp:idxrow6]=[");
      for (int i = 0; i < (int)idxs.shape(2); i++) fprintf(stderr, "%d,", item_i(row6, i));
      fprintf(stderr, "]\n");
    }
  }
  auto o = sparse_attn(q, kv, w(p + "attn_sink"), idxs);
  tr("o_raw", o);
  if (tr_on() && sp == 11 && l == 0) { /* decode0 全量转储 */
    auto dq = q;
    dq.eval();
    FILE* fp = fopen("/tmp/d_q_cpp.bin", "wb");
    fwrite(dq.data<float>(), 4, (size_t)dq.size(), fp);
    fclose(fp);
    kv.eval();
    fp = fopen("/tmp/d_kvf_cpp.bin", "wb");
    fwrite(kv.data<float>(), 4, (size_t)kv.size(), fp);
    fclose(fp);
    idxs.eval();
    fp = fopen("/tmp/d_idx_cpp.bin", "wb");
    fwrite(idxs.data<int>(), 4, (size_t)idxs.size(), fp);
    fclose(fp);
    auto oraw = o;
    oraw.eval();
    fp = fopen("/tmp/d_o_cpp.bin", "wb");
    fwrite(oraw.data<float>(), 4, (size_t)oraw.size(), fp);
    fclose(fp);
  }
  o = rotary_tail(o, fc0, fc1, true);
  tr("o_inv", o);
  o = act_quant_dq(mx::reshape(o, {B, s, G, H * DH / G}));
  tr("o_q", o);
  if (tr_on() && sp == 0 && s > 6) {
    auto sl = mx::reshape(mx::slice(o, Shape{0, 3, 0, 0}, Shape{1, 4, 1, 8}), {8});
    sl.eval();
    fprintf(stderr, "[cpp:oq_e]=[%g,%g,%g,%g,%g,%g,%g,%g]\n", item_f(sl, 0), item_f(sl, 1), item_f(sl, 2),
            item_f(sl, 3), item_f(sl, 4), item_f(sl, 5), item_f(sl, 6), item_f(sl, 7));
  }
  auto ot = mx::reshape(mx::transpose(o, {2, 0, 1, 3}), {G, B * s, H * DH / G}); /* [G,B*S,DHg]：batch 对齐 wa */
  auto wa = mx::transpose(f32(w(p + "wo_a")), {0, 2, 1}); /* [G,DHg,R] */
  auto og = mx::matmul(ot, wa);                      /* [G,B*S,R] */
  og = mx::reshape(og, {G, B, s, RORD});
  static int ogN = 0;
  ogN++;
  if (tr_on() && sp == 0 && s > 6) { /* 全量落盘供 numpy 对比 */
    og.eval();
    char fn[64];
    snprintf(fn, sizeof fn, "/tmp/og_cpp_%d.bin", ogN);
    FILE* fp = fopen(fn, "wb");
    fwrite(og.data<float>(), 4, (size_t)(G * B * s * RORD), fp);
    fclose(fp);
  }
  if (tr_on() && sp == 0 && s > 6) {
    auto og0 = mx::reshape(mx::slice(og, Shape{0, 0, 3, 0}, Shape{1, 1, 4, 8}), {8});
    og0.eval();
    fprintf(stderr, "[cpp:og_e]=[%g,%g,%g,%g,%g,%g,%g,%g]\n", item_f(og0, 0), item_f(og0, 1), item_f(og0, 2),
            item_f(og0, 3), item_f(og0, 4), item_f(og0, 5), item_f(og0, 6), item_f(og0, 7));
    auto og1 = mx::reshape(mx::slice(o, Shape{0, 3, 1, 0}, Shape{1, 4, 2, 8}), {8});
    og1.eval();
    fprintf(stderr, "[cpp:oq_g1]=[%g,%g,%g,%g,%g,%g,%g,%g]\n", item_f(og1, 0), item_f(og1, 1), item_f(og1, 2),
            item_f(og1, 3), item_f(og1, 4), item_f(og1, 5), item_f(og1, 6), item_f(og1, 7));
    auto ogg1 = mx::reshape(mx::slice(og, Shape{1, 0, 3, 0}, Shape{2, 1, 4, 8}), {8}); /* g=1,b=0,s=3,r=0..7 */
    ogg1.eval();
    fprintf(stderr, "[cpp:og_g1]=[%g,%g,%g,%g,%g,%g,%g,%g]\n", item_f(ogg1, 0), item_f(ogg1, 1), item_f(ogg1, 2),
            item_f(ogg1, 3), item_f(ogg1, 4), item_f(ogg1, 5), item_f(ogg1, 6), item_f(ogg1, 7));
    auto ogfa = mx::reshape(mx::slice(og, Shape{0, 0, 11, 1}, Shape{1, 1, 12, 5}), {4}); /* g0,b0,s11,r1..4 */
    auto ogfb = mx::reshape(mx::slice(og, Shape{0, 1, 5, 0}, Shape{1, 2, 6, 4}), {4});  /* g0,b1,s5,r0..3 */
    ogfa.eval();
    ogfb.eval();
    fprintf(stderr, "[cpp:og_f]=[%g,%g,%g,%g]+[%g,%g,%g,%g]\n", item_f(ogfa, 0), item_f(ogfa, 1), item_f(ogfa, 2),
            item_f(ogfa, 3), item_f(ogfb, 0), item_f(ogfb, 1), item_f(ogfb, 2), item_f(ogfb, 3));
  }
  og = mx::transpose(og, {1, 2, 0, 3});              /* [B,S,G,R] */
  auto out = linear(mx::reshape(og, {B, s, G * RORD}), w(p + "wo_b.weight"));
  if (tr_on()) {
    char tg[64];
    snprintf(tg, sizeof tg, "L%d.attn sp=%d", l, sp);
    tr(tg, out);
    if (l == 0 && sp == 0 && s > 6) {
      out.eval();
      FILE* fp = fopen("/tmp/at0_cpp.bin", "wb");
      fwrite(out.data<float>(), 4, (size_t)(B * s * D), fp);
      fclose(fp);
    }
  }
  return out;
}

/* ---------- MoE ---------- */
mx::array expert_fwd(const std::string& pfx, const mx::array& xq0, const mx::array& wts) {
  auto xq = act_quant_dq(xq0);
  auto gate = mx::matmul(xq, mx::transpose(f32(w(pfx + "w1.wq")), {1, 0}));
  auto up = mx::matmul(xq, mx::transpose(f32(w(pfx + "w3.wq")), {1, 0}));
  up = clipS(up, -SWIGLU_LIM, SWIGLU_LIM);
  gate = mx::minimum(gate, fa(SWIGLU_LIM));
  auto y = act_quant_dq(gate * mx::sigmoid(gate) * up * wts); /* w2 输入同样过 fp8 假量化 */
  return mx::matmul(y, mx::transpose(f32(w(pfx + "w2.wq")), {1, 0}));
}

mx::array moe_forward(int l, const mx::array& x) {
  std::string p = "blocks." + std::to_string(l) + ".ffn.";
  int n = x.shape(0) * x.shape(1);
  if (tr_on()) {
    char tg[64];
    snprintf(tg, sizeof tg, "L%d.ffn_in", l);
    tr(tg, x);
  }
  auto xf = mx::reshape(x, {n, D});
  auto sc = mx::matmul(f32(xf), mx::transpose(f32(w(p + "gate.w")), {1, 0})) / fa(GATE_T);
  auto e2x = mx::exp(sc);
  sc = mx::where(sc > fa(20.0f), sc, mx::log(fa(1.0f) + e2x)); /* softplus，threshold=20 */
  sc = mx::sqrt(sc);
  auto biased = sc + mx::reshape(f32(w(p + "gate.bias")), {1, E});
  /* topk：k 轮 argmax + 屏蔽 */
  std::vector<mx::array> picks;
  auto cur = biased;
  for (int i = 0; i < TOPK; i++) {
    auto am = mx::astype(mx::argmax(cur, {-1}), mx::int32); /* [n] */
    picks.push_back(am);
    auto onehot = mx::reshape(mx::arange(0, E, 1, mx::int32), {1, E}) == mx::reshape(am, {n, 1});
    cur = mx::where(onehot, fa(-1e30f), cur);
  }
  auto idx = mx::stack(picks, -1); /* [n,k] */
  auto w8 = mx::take_along_axis(sc, idx, 1);
  w8 = w8 / (mx::sum(w8, {-1}, true) + fa(1e-20f));
  w8 = w8 * fa(ROUTE_SCALE);
  auto k0 = mx::slice(idx, Shape{0, 0}, Shape{n, 1});
  auto k1 = mx::slice(idx, Shape{0, 1}, Shape{n, 2});
  auto w0 = mx::slice(w8, Shape{0, 0}, Shape{n, 1});
  auto w1 = mx::slice(w8, Shape{0, 1}, Shape{n, 2});
  auto y = mx::zeros({n, D}, mx::float32);
  for (int i = 0; i < E; i++) {
    auto wts = mx::astype(k0 == ia(i), mx::float32) * w0 + mx::astype(k1 == ia(i), mx::float32) * w1;
    y = y + expert_fwd(p + "experts." + std::to_string(i) + ".", xf, wts);
  }
  y = y + expert_fwd(p + "shared.", xf, mx::ones({n, 1}, mx::float32));
  auto out = mx::reshape(y, x.shape());
  if (tr_on()) {
    char tg[64];
    snprintf(tg, sizeof tg, "L%d.ffn", l);
    tr(tg, out);
    if (l == 0 && x.shape(1) > 6) {
      out.eval();
      FILE* fp = fopen("/tmp/ff0_cpp.bin", "wb");
      fwrite(out.data<float>(), 4, (size_t)(out.size()), fp);
      fclose(fp);
    }
  }
  return out;
}

/* ---------- Engram（哈希在 host） ---------- */
mx::array engram_forward(const mx::array& x, const uint32_t* ids_h, int sp, int s) {
  for (int b = 0; b < B; b++)
    for (int i = 0; i < s; i++) id_hist[b][sp + i] = (int)ids_h[b * s + i];
  std::vector<mx::array> cells;
  cells.reserve(B * s * 2);
  for (int b = 0; b < B; b++) {
    for (int i = 0; i < s; i++) {
      int pos = sp + i;
      /* mirror：pos<shift 时该 token 视为 blocked（pad=7），否则读 cache[pos-shift] */
      int64_t r0 = ((pos - 0) < 0 ? 7 : id_hist[b][pos]) * mult0[0];
      int64_t r1 = r0 ^ (((pos - 1) < 0 ? 7 : id_hist[b][pos - 1]) * mult0[1]);
      int64_t r2 = r1 ^ (((pos - 2) < 0 ? 7 : id_hist[b][pos - 2]) * mult0[2]);
      cells.push_back(ia((int)(r1 % prime0[0])));
      cells.push_back(ia((int)(r2 % prime0[1])));
    }
  }
  auto rows = mx::reshape(mx::stack(cells, 0), {B, s, 2});
  if (tr_on() && s > 6) {
    rows.eval();
    fprintf(stderr, "[cpp:rows]=[%d,%d][%d,%d] hist_b1_5_6_7=[%d,%d,%d]\n", item_i(rows, 3 * 2),
            item_i(rows, 3 * 2 + 1), item_i(rows, 19 * 2), item_i(rows, 19 * 2 + 1), id_hist[1][5], id_hist[1][6],
            id_hist[1][7]); /* 19 = b1,s7 的扁平行号 */
  }
  auto vals = mx::take(w("engram.table"), rows, 0); /* [B,S,2,ENG_EHD] */
  auto kv = linear(mx::reshape(vals, {B, s, 2 * 32}), w("engram.wkv.weight"));
  auto key = mx::reshape(mx::slice(kv, Shape{0, 0, 0}, Shape{B, s, HC * D}), {B, s, HC, D});
  auto value = mx::reshape(mx::slice(kv, Shape{0, 0, HC * D}, Shape{B, s, HC * D + D}), {B, s, 1, D});
  auto wgt = f32(w("engram.q_weight")) * f32(w("engram.k_weight"));
  auto h = f32(x);
  auto rstd = mx::rsqrt(mx::mean(mx::square(h), {-1}) + fa(N_EPS)) *
              mx::rsqrt(mx::mean(mx::square(key), {-1}) + fa(N_EPS)); /* [B,s,HC] */
  auto dot = mx::sum(h * mx::reshape(wgt, {1, 1, HC, D}) * key, {-1}) * rstd * fa(1.0f / std::sqrt((float)D));
  auto g = mx::sigmoid(mx::sign(dot) * mx::sqrt(mx::maximum(mx::abs(dot), fa(1e-6f))));
  auto out = h + mx::expand_dims(g, -1) * value;
  tr("engram", out);
  if (tr_on() && sp == 0 && s > 6) {
    out.eval();
    FILE* fp = fopen("/tmp/eg_cpp.bin", "wb");
    fwrite(out.data<float>(), 4, (size_t)(B * s * HC * D), fp);
    fclose(fp);
  }
  return out;
}

/* ---------- Hyper-Connections ---------- */
std::tuple<mx::array, mx::array, mx::array> sinkhorn_split(const mx::array& mixes, const mx::array& scale,
                                                           const mx::array& base) {
  int n = mixes.shape(0);
  auto pre = mx::sigmoid(mx::slice(mixes, Shape{0, 0}, Shape{n, HC}) * mx::slice(scale, Shape{0}, Shape{1}) +
                         mx::reshape(mx::slice(base, Shape{0}, Shape{HC}), {1, HC})) +
             fa(HC_EPS);
  auto post = fa(2.0f) * mx::sigmoid(mx::slice(mixes, Shape{0, HC}, Shape{n, 2 * HC}) *
                                         mx::slice(scale, Shape{1}, Shape{2}) +
                                         mx::reshape(mx::slice(base, Shape{HC}, Shape{2 * HC}), {1, HC}));
  auto comb = mx::reshape(mx::slice(mixes, Shape{0, 2 * HC}, Shape{n, 6 * HC}), {n, HC, HC}) *
                  mx::reshape(mx::slice(scale, Shape{2}, Shape{3}), {1, 1}) +
              mx::reshape(mx::slice(base, Shape{2 * HC}, Shape{6 * HC}), {HC, HC});
  comb = mx::softmax(comb, -1) + fa(HC_EPS);
  comb = comb / (mx::sum(comb, {-2}, true) + fa(HC_EPS));
  for (int i = 0; i < SN_ITERS - 1; i++) {
    comb = comb / (mx::sum(comb, {-1}, true) + fa(HC_EPS));
    comb = comb / (mx::sum(comb, {-2}, true) + fa(HC_EPS));
  }
  return {mx::reshape(pre, {B, -1, HC}), mx::reshape(post, {B, -1, HC}), mx::reshape(comb, {B, -1, HC, HC})};
}

std::tuple<mx::array, mx::array, mx::array> hc_mixes(int l, const mx::array& x, bool attn) {
  std::string p = "blocks." + std::to_string(l) + ".";
  int s = x.shape(1);
  auto xf = mx::reshape(x, {B * s, HC * D});
  auto rsqrt = mx::rsqrt(mx::mean(mx::square(xf), {-1}, true) + fa(N_EPS));
  auto fn = attn ? w(p + "hc_attn_fn") : w(p + "hc_ffn_fn");
  auto scale = attn ? w(p + "hc_attn_scale") : w(p + "hc_ffn_scale");
  auto base = attn ? w(p + "hc_attn_base") : w(p + "hc_ffn_base");
  auto mixes = mx::matmul(f32(xf), mx::transpose(f32(fn), {1, 0})) * rsqrt;
  auto r = sinkhorn_split(mixes, scale, base);
  return {mx::reshape(std::get<0>(r), {B, s, HC}), mx::reshape(std::get<1>(r), {B, s, HC}),
          mx::reshape(std::get<2>(r), {B, s, HC, HC})};
}

std::pair<mx::array, mx::array> block_forward(int l, const mx::array& x_in, int sp, int s,
                                              const mx::array& pre_mix) {
  std::string p = "blocks." + std::to_string(l) + ".";
  auto residual = x_in;
  auto a = hc_mixes(l, x_in, true);
  if (tr_on()) {
    char tg[64];
    snprintf(tg, sizeof tg, "L%d.a_post", l);
    tr(tg, std::get<1>(a));
    snprintf(tg, sizeof tg, "L%d.a_comb", l);
    tr(tg, std::get<2>(a));
  }
  auto y = mx::sum(mx::expand_dims(pre_mix, -1) * x_in, 2);
  y = rms_norm(y, w(p + "attn_norm.weight"));
  y = attn_forward(l, y, sp, s);
  auto x = mx::expand_dims(std::get<1>(a), -1) * mx::reshape(y, {B, s, 1, D}) +
           mx::sum(mx::expand_dims(std::get<2>(a), -1) * mx::expand_dims(residual, -2), 2);

  residual = x;
  auto f = hc_mixes(l, x, false);
  if (tr_on() && l == 0 && sp == 0 && s > 6) {
    auto fp1 = mx::expand_dims(std::get<1>(f), -1);
    auto fc = std::get<2>(f);
    fp1.eval();
    fc.eval();
    FILE* fp = fopen("/tmp/fpost_cpp.bin", "wb");
    fwrite(fp1.data<float>(), 4, (size_t)(fp1.size()), fp);
    fclose(fp);
    fp = fopen("/tmp/fcomb_cpp.bin", "wb");
    fwrite(fc.data<float>(), 4, (size_t)(fc.size()), fp);
    fclose(fp);
    auto r2 = mx::expand_dims(residual, -3);
    r2.eval();
    fp = fopen("/tmp/res2_cpp.bin", "wb");
    fwrite(r2.data<float>(), 4, (size_t)(r2.size()), fp);
    fclose(fp);
  }
  y = mx::sum(mx::expand_dims(std::get<0>(a), -1) * x, 2);
  y = rms_norm(y, w(p + "ffn_norm.weight"));
  y = moe_forward(l, y);
  if (tr_on() && l == 0 && sp == 0 && s > 6) {
    y.eval();
    FILE* fp = fopen("/tmp/y_cpp.bin", "wb");
    fwrite(y.data<float>(), 4, (size_t)(y.size()), fp);
    fclose(fp);
  }
  x = mx::expand_dims(std::get<1>(f), -1) * mx::reshape(y, {B, s, 1, D}) +
      mx::sum(mx::expand_dims(std::get<2>(f), -1) * mx::expand_dims(residual, -2), 2);
  if (tr_on() && l == 0 && sp == 0 && s > 6) {
    x.eval();
    FILE* fp = fopen("/tmp/blk0_cpp.bin", "wb");
    fwrite(x.data<float>(), 4, (size_t)(x.size()), fp);
    fclose(fp);
  }
  return {x, std::get<0>(f)};
}

mx::array forward_full(const mx::array& ids, int sp, int s, const uint32_t* ids_h) {
  auto h = mx::take(w("embed.weight"), ids, 0); /* [B,S,D] */
  tr("embed", h);
  h = mx::repeat(mx::reshape(h, {B, s, 1, D}), HC, 2);
  auto pre_mix = mx::astype(
      mx::broadcast_to(mx::reshape(mx::arange(0, HC, 1, mx::int32) == ia(0), {1, 1, HC}), {B, s, HC}),
      mx::float32);
  for (int l = 0; l < L; l++) {
    if (l == 1) h = engram_forward(h, ids_h, sp, s); /* engram_layer_ids=(1,)，Block 之前 */
    auto r = block_forward(l, h, sp, s, pre_mix);
    h = r.first;
    pre_mix = r.second;
  }
  auto hc = mx::sum(mx::expand_dims(pre_mix, -1) * h, 2);
  auto hn = rms_norm(hc, w("norm.weight"));
  return mx::matmul(hn, mx::transpose(f32(w("head_w")), {1, 0})); /* [B,S,V] */
}

} // namespace

int sgl_ds41_load(const char* dir) {
  std::string d(dir);
  W = mx::load_safetensors(d + "/mini_ref.st").first;
  o_wcos = W.at("f_win_cos");
  o_wsin = W.at("f_win_sin");
  o_ccos = W.at("f_cmp_cos");
  o_csin = W.at("f_cmp_sin");
  o_ids = W.at("ids");
  for (int i = 0; i < 3; i++) mult0[i] = item_i(W.at("engram_mult_0"), i);
  prime0[0] = item_i(W.at("engram_primes"), 0);
  prime0[1] = item_i(W.at("engram_primes"), 1);
  sgl_ds41_reset();
  return 0;
}

void sgl_ds41_reset(void) {
  win_cache(0) = mx::zeros({B, WIN, DH}, mx::float32);
  win_cache(1) = mx::zeros({B, WIN, DH}, mx::float32);
  cmp_cache() = mx::zeros({B, MAXSEQ / R0, DH}, mx::float32);
  idx_cache() = mx::zeros({B, MAXSEQ / R0, IHD}, mx::float32);
  kv_state() = mx::zeros({B, R0, DH}, mx::float32);
  score_state() = mx::full({B, R0, DH}, -INFINITY, mx::float32); /* py: torch.full(..., -inf) */
  memset(id_hist, 0, sizeof(id_hist));
}

int sgl_ds41_run(int n_dec, float* logits_pre, uint32_t* toks, float* logits_dec) {
  std::vector<uint32_t> ids_h(B * S_IN);
  for (int i = 0; i < B * S_IN; i++) ids_h[i] = (uint32_t)item_i(ids_dev(), i);

  /* prefill */
  auto logits = forward_full(ids_dev(), 0, S_IN, ids_h.data());
  auto lg_pre = mx::reshape(mx::slice(logits, Shape{0, S_IN - 1, 0}, Shape{B, S_IN, V}), {B * V});
  lg_pre.eval();
  memcpy(logits_pre, lg_pre.data<float>(), sizeof(float) * B * V);
  /* decode：py 语义——i=0 在位置 S_IN-1 重喂 prompt 末位 token（重算该位置），
     其后每步喂上一步的 argmax；toks 存每步的 argmax（= npz 的 tok_j）。 */
  uint32_t prv[2] = {ids_h[S_IN - 1], ids_h[(B - 1) * S_IN + S_IN - 1]};
  for (int i = 0; i < n_dec; i++) {
    int pos = S_IN - 1 + i;
    std::vector<uint32_t> tmp(B);
    for (int b = 0; b < B; b++) tmp[b] = prv[b];
    auto ids1 = mx::reshape(mx::stack({ia((int)prv[0]), ia((int)prv[1])}, 0), {B, 1});
    auto lg = forward_full(ids1, pos, 1, tmp.data()); /* [B,1,V] */
    auto lastd = mx::reshape(lg, {B * V});
    lastd.eval();
    memcpy(logits_dec + (size_t)i * B * V, lastd.data<float>(), sizeof(float) * B * V);
    auto nxtd = mx::astype(mx::argmax(lg, {-1}), mx::uint32);
    nxtd.eval();
    prv[0] = (uint32_t)item_i(nxtd, 0);
    prv[1] = (uint32_t)item_i(nxtd, 1);
    toks[2 * i] = prv[0];
    toks[2 * i + 1] = prv[1];
  }
  return 0;
}
