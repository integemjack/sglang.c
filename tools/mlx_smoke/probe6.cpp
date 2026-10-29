// probe6: 验证「读 mlx-community 4bit 三件套 + MLX dequantize」这条链的数值。
// 用 MLX 自己的 dequantize 当参照，看手工拆包能不能对上——对不上就说明布局理解错。
#include <mlx/mlx.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>
#include <map>

namespace mx = mlx::core;

static std::string g_file;
static size_t g_base = 0;
static std::string g_hdr;

// 极简 safetensors：只做 main_meta 解析 + 按名字取原始字节
struct Meta { size_t a, b; std::string dtype; std::vector<int> shape; };
static std::map<std::string, Meta> g_meta;

static bool load_header(const char* path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  uint64_t n = 0;
  f.read((char*)&n, 8);
  g_hdr.resize(n);
  f.read(&g_hdr[0], n);
  g_base = 8 + n;
  // 手工扫 JSON（够用）：找 "<name>":{...}
  size_t i = 0;
  while ((i = g_hdr.find("\"dtype\"", i)) != std::string::npos) {
    size_t obj = g_hdr.rfind('{', i);
    size_t key_q2 = g_hdr.rfind('"', obj - 2);  // key 的结束引号
    size_t key_q1 = g_hdr.rfind('"', key_q2 - 1);
    std::string name = g_hdr.substr(key_q1 + 1, key_q2 - key_q1 - 1);
    size_t doff = g_hdr.find("data_offsets", i);
    size_t lb = g_hdr.find('[', doff), comma = g_hdr.find(',', lb), rb = g_hdr.find(']', lb);
    Meta m;
    m.a = strtoull(g_hdr.c_str() + lb + 1, nullptr, 10);
    m.b = strtoull(g_hdr.c_str() + comma + 1, nullptr, 10);
    size_t d1 = g_hdr.find('"', i + 7), d2 = g_hdr.find('"', d1 + 1);
    m.dtype = g_hdr.substr(d1 + 1, d2 - d1 - 1);
    size_t so = g_hdr.find('"', g_hdr.find("shape", i) + 5);
    size_t se = g_hdr.find(']', so);
    std::string sh = g_hdr.substr(so, se - so);
    for (size_t p = 0; p < sh.size(); p++) {
      if (sh[p] >= '0' && sh[p] <= '9') {
        int v = 0;
        while (p < sh.size() && sh[p] >= '0' && sh[p] <= '9') v = v * 10 + (sh[p++] - '0');
        m.shape.push_back(v);
      }
    }
    g_meta[name] = m;
    i = d2;
  }
  return !g_meta.empty();
}

static std::vector<char> raw(const std::string& name) {
  auto it = g_meta.find(name);
  if (it == g_meta.end()) return {};
  std::ifstream f(g_file, std::ios::binary);
  f.seekg(g_base + it->second.a);
  std::vector<char> v(it->second.b - it->second.a);
  f.read(v.data(), v.size());
  return v;
}

int main(int argc, char** argv) {
  if (argc < 2) return 1;
  g_file = std::string(argv[1]) + "/model.safetensors";
  if (!load_header(g_file.c_str())) { printf("header fail\n"); return 1; }
  const std::string base = "model.layers.0.self_attn.q_proj";
  auto wraw = raw(base + ".weight"), sraw = raw(base + ".scales"), braw = raw(base + ".biases");
  Meta mw = g_meta[base + ".weight"], ms = g_meta[base + ".scales"];
  printf("w %s [%d,%d]  scales %s [%d,%d]  bytes w=%zu s=%zu b=%zu\n",
         mw.dtype.c_str(), mw.shape[0], mw.shape[1], ms.dtype.c_str(), ms.shape[0],
         ms.shape[1], wraw.size(), sraw.size(), braw.size());
  const int N = mw.shape[0], KP = mw.shape[1];
  const int K = KP * 32 / 4;
  const int G = K / ms.shape[1];
  printf("N=%d KP=%d K=%d group=%d\n", N, KP, K, G);
  if ((int)wraw.size() != N * KP * 4 || (int)sraw.size() != ms.shape[0] * ms.shape[1] * 2) {
    printf("size mismatch\n");
    return 1;
  }

  // MLX 侧
  mx::array w = mx::zeros(mx::Shape{N, KP}, mx::uint32);
  w.eval();
  memcpy(w.data<uint32_t>(), wraw.data(), wraw.size());
  mx::array sc = mx::zeros(mx::Shape{ms.shape[0], ms.shape[1]}, mx::float16);
  sc.eval();
  memcpy(sc.data<uint16_t>(), sraw.data(), sraw.size());
  mx::array bi = mx::zeros(mx::Shape{ms.shape[0], ms.shape[1]}, mx::float16);
  bi.eval();
  memcpy(bi.data<uint16_t>(), braw.data(), braw.size());

  mx::array dq = mx::dequantize(w, sc, bi, G, 4);
  dq.eval();
  auto dqf = mx::astype(dq, mx::float32);
  dqf.eval();
  const float* d = dqf.data<float>();
  double mn = 1e30, mxv = -1e30, sum = 0, sq = 0;
  for (int i = 0; i < N * K; i++) {
    double v = d[i];
    if (v < mn) mn = v;
    if (v > mxv) mxv = v;
    sum += v;
    sq += v * v;
  }
  double mean = sum / (N * K);
  printf("MLX dequantize : mean=%+.6f std=%.6f min=%+.4f max=%+.4f\n", mean,
         sqrt(sq / (N * K) - mean * mean), mn, mxv);

  // 手工拆包（每 uint32 8 个 nibble，低->高）
  {
    const uint32_t* u = (const uint32_t*)wraw.data();
    const uint16_t* sh = (const uint16_t*)sraw.data();
    const uint16_t* bh = (const uint16_t*)braw.data();
    auto h2f = [](uint16_t h) {
      uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1F, m = h & 0x3FF;
      if (e == 0) return (float)(m * pow(2.0, -24));
      if (e == 31) return INFINITY;
      return (float)((1 + m / 1024.0) * pow(2.0, (int)e - 15));
    };
    double sum2 = 0, sq2 = 0, mn2 = 1e30, mx2 = -1e30;
    for (int n = 0; n < N; n++) {
      for (int k = 0; k < K; k++) {
        int j = k / 8, i = k % 8;
        int qv = (u[n * KP + j] >> (4 * i)) & 0xF;
        int g = k / G;
        double v = qv * h2f(sh[n * ms.shape[1] + g]) + h2f(bh[n * ms.shape[1] + g]);
        if (v < mn2) mn2 = v;
        if (v > mx2) mx2 = v;
        sum2 += v;
        sq2 += v * v;
      }
    }
    double mean2 = sum2 / (N * K);
    printf("manual unpack  : mean=%+.6f std=%.6f min=%+.4f max=%+.4f\n", mean2,
           sqrt(sq2 / (N * K) - mean2 * mean2), mn2, mx2);
  }

  // scales 的统计（用 MLX 解出来）
  auto scf = mx::astype(sc, mx::float32);
  scf.eval();
  double smn = 1e30, smx = -1e30, ssum = 0; int sneg = 0;
  for (int i = 0; i < ms.shape[0] * ms.shape[1]; i++) {
    double v = scf.data<float>()[i];
    if (v < 0) sneg++;
    if (v < smn) smn = v;
    if (v > smx) smx = v;
    ssum += v;
  }
  printf("scales: mean=%.6f min=%.6f max=%.6f negative=%d/%d\n", ssum / (ms.shape[0] * ms.shape[1]),
         smn, smx, sneg, ms.shape[0] * ms.shape[1]);
  return 0;
}