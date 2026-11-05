// ds41_conv: DeepSeek-V4.1-Flash GGUF(分片) -> MLX safetensors（稠密/注意力权重）
// 专家(*_exps)和 Engram 表(per_layer_token_embd)留在 GGUF，由引擎按行 mmap 读取。
// 量化：>8M 参数 -> affine bits=4 g64；>64K -> f16；其余 f32。
// MLX qmm 语义（源码确认）：dequant = q*scale + bias，LSB-first U32 打包。
// 构建: c++ -O2 -std=c++17 ds41_conv.cpp -o ds41_conv
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;

static float f16_to_f32(u16 h) {
    u32 sign = (h & 0x8000) << 16, exp = (h >> 10) & 0x1f, man = h & 0x3ff;
    u32 f;
    if (exp == 0) {
        if (man == 0) { f = sign; }
        else { int e = -1; u32 m = man; do { e++; m <<= 1; } while (!(m & 0x400)); m &= 0x3ff;
               f = sign | ((127 - 15 - e + 1) << 23) | (m << 13); }
    } else if (exp == 31) { f = sign | 0x7f800000 | (man << 13);
    } else { f = sign | ((exp - 15 + 127) << 23) | (man << 13); }
    float o; memcpy(&o, &f, 4); return o;
}
static u16 f32_to_f16(float v) {
    u32 x; memcpy(&x, &v, 4);
    u32 sign = (x >> 16) & 0x8000; i32 e = (i32)((x >> 23) & 0xff) - 127 + 15;
    u32 man = x & 0x7fffff;
    if (((x >> 23) & 0xff) == 0xff) return (u16)(sign | 0x7c00 | (man ? 0x200 : 0));
    if (e >= 31) return (u16)(sign | 0x7c00);
    if (e <= 0) return (u16)sign;
    return (u16)(sign | ((u32)e << 10) | (man >> 13));
}
static float bf16_to_f32(u16 h) {
    u32 f = (u32)h << 16; float o; memcpy(&o, &f, 4); return o;
}

// ---------------- GGUF ----------------
struct GTensor { int fi; u64 off; u32 gtype; std::vector<u64> ne; u64 nbytes; };

struct GGUF {
    std::vector<int> fds;
    std::vector<u8*> maps;
    std::vector<size_t> sizes;
    std::unordered_map<std::string, GTensor> tensors;
    std::map<std::string, std::vector<double>> arr_kv;   // 数值数组
    std::map<std::string, double> scalar_kv;
    std::vector<u32> token_map;
    std::vector<u64> data_base; /* 每分片头尾对齐后的数据基址 */
    u32 alignment = 32;

    void open(const std::string& prefix, int n) {
        for (int i = 1; i <= n; i++) {
            char p[1024];
            snprintf(p, sizeof p, prefix.c_str(), i);
            int fd = ::open(p, O_RDONLY);
            if (fd < 0) { fprintf(stderr, "open %s 失败\n", p); exit(1); }
            struct stat st; fstat(fd, &st);
            u8* m = (u8*)mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
            if (m == MAP_FAILED) { fprintf(stderr, "mmap %s 失败\n", p); exit(1); }
            madvise(m, st.st_size, MADV_WILLNEED);
            fds.push_back(fd); maps.push_back(m); sizes.push_back(st.st_size);
            data_base.push_back(0); /* 占位，index_file 里填 */
            index_file(i - 1);
        }
    }
    static u64 rd64(u8* p) { u64 v; memcpy(&v, p, 8); return v; }
    static u32 rd32(u8* p) { u32 v; memcpy(&v, p, 4); return v; }
    std::string rstr(u8*& p) { u64 n = rd64(p); p += 8; std::string s((char*)p, n); p += n; return s; }

    void index_file(int fi) {
        u8* p = maps[fi];
        if (memcmp(p, "GGUF", 4)) { fprintf(stderr, "非 GGUF\n"); exit(1); }
        p += 4; p += 4; /* 版本 */
        u64 n_ten = rd64(p); p += 8;
        u64 n_kv = rd64(p); p += 8;
        if (fi == 0) {
            for (u64 k = 0; k < n_kv; k++) {
                std::string key = rstr(p);
                u32 t = rd32(p); p += 4;
                parse_kv(fi, key, t, p);
            }
        } else {
            for (u64 k = 0; k < n_kv; k++) { rstr(p); u32 t = rd32(p); p += 4; skip_val(t, p); }
        }
        for (u64 k = 0; k < n_ten; k++) {
            std::string name = rstr(p);
            u32 nd = rd32(p); p += 4;
            GTensor t; t.fi = fi;
            t.ne.resize(nd);
            u64 total = 1;
            for (u32 d = 0; d < nd; d++) { t.ne[d] = rd64(p); p += 8; total *= t.ne[d]; }
            t.gtype = rd32(p); p += 4;
            t.off = rd64(p); p += 8;
            t.nbytes = ggml_nbytes(t);
            tensors[name] = t;
        }
        /* 数据基址 = 张量索引结束处按 alignment 向上对齐；统一加到本分片张量的 off 上 */
        auto it_al = scalar_kv.find("general.alignment");
        if (it_al != scalar_kv.end() && it_al->second >= 1) alignment = (u32)it_al->second;
        u64 hb = (u64)(p - maps[fi]);
        hb = (hb + alignment - 1) / alignment * alignment;
        data_base.push_back(hb);
        for (auto& kv2 : tensors)
            if (kv2.second.fi == fi) kv2.second.off += hb;
    }
    void parse_kv(int fi, const std::string& key, u32 t, u8*& p) {
        if (t == 8) { rstr(p); return; }
        if (t == 9) { /* 数组 */
            u32 et = rd32(p); p += 4;
            u64 n = rd64(p); p += 8;
            if (et == 8) { for (u64 i = 0; i < n; i++) rstr(p); return; }
            size_t es = vt_size(et);
            if (key == "deepseek41.engram.token_map") {
                token_map.resize(n);
                for (u64 i = 0; i < n; i++) token_map[i] = et == 4 ? rd32(p + i * es) : (u32)read_int(et, p + i * es);
                p += n * es; return;
            }
            std::vector<double> v(n);
            for (u64 i = 0; i < n; i++) v[i] = read_num(et, p + i * es);
            arr_kv[key] = v;
            p += n * es;
            return;
        }
        size_t es = vt_size(t);
        if (es == 0) { fprintf(stderr, "未知 kv 类型 %u (%s)\n", t, key.c_str()); exit(1); }
        scalar_kv[key] = read_num(t, p);
        p += es;
    }
    void skip_val(u32 t, u8*& p) {
        if (t == 8) { rstr(p); return; }
        if (t == 9) { u32 et = rd32(p); p += 4; u64 n = rd64(p); p += 8;
                      if (et == 8) { for (u64 i = 0; i < n; i++) rstr(p); }
                      else p += n * vt_size(et); return; }
        p += vt_size(t);
    }
    static size_t vt_size(u32 t) {
        switch (t) { case 0: case 1: case 7: return 1; case 2: case 3: return 2;
                     case 4: case 5: case 6: return 4; case 10: case 11: case 12: return 8; }
        return 0;
    }
    static double read_num(u32 t, u8* p) {
        switch (t) { case 0: return *(i8*)p; case 1: return *(u8*)p; case 2: /* u16 */ { u16 v; memcpy(&v, p, 2); return v; }
                     case 3: { i16 v; memcpy(&v, p, 2); return v; } case 4: return rd32(p);
                     case 5: { i32 v; memcpy(&v, p, 4); return v; } case 6: { float v; memcpy(&v, p, 4); return v; }
                     case 7: return *(u8*)p; case 10: return rd64(p); case 11: { i64 v; memcpy(&v, p, 8); return (double)v; }
                     case 12: { double v; memcpy(&v, p, 8); return v; } }
        return 0;
    }
    static i64 read_int(u32 t, u8* p) { return (i64)read_num(t, p); }

    static u64 ggml_nbytes(const GTensor& t) {
        u64 n = 1; for (size_t d = 0; d + 1 < t.ne.size(); d++) n *= t.ne[d];
        u64 ne0 = t.ne[0];
        auto bp = [&](u32 gt, u64 nEl) -> u64 {
            switch (gt) { case 0: return nEl * 4; case 1: return nEl * 2; case 30: return nEl * 2; case 28: return nEl * 8;
                case 8: return nEl / 32 * 34; case 12: return nEl / 256 * 144; case 13: return nEl / 256 * 176;
                case 14: return nEl / 256 * 210; case 16: return nEl / 256 * 66; case 17: return nEl / 256 * 74;
                case 19: return nEl / 256 * 50; case 20: return nEl / 256 * 136; case 23: return nEl / 256 * 136;
                case 29: return nEl / 256 * 56; case 24: return nEl; case 26: return nEl * 4; }
            fprintf(stderr, "未知 gtype %u\n", gt); exit(1);
        };
        return bp(t.gtype, n * ne0);
    }

    std::vector<float> read_f32(const std::string& name) { /* 返回 [ne1 行, ne0 列] 行主序 */
        auto it = tensors.find(name);
        if (it == tensors.end()) { fprintf(stderr, "缺张量 %s\n", name.c_str()); exit(1); }
        const GTensor& t = it->second;
        u64 nEl = 1; for (auto d : t.ne) nEl *= d;
        std::vector<float> out(nEl);
        const u8* src = maps[t.fi] + t.off;
        dequant(src, t, out.data());
        /* 反量化防线：权重出现非有限值或异常量级，说明读取错位，别把垃圾写进 safetensors */
        {
            double mx = 0; long bad = 0;
            for (u64 i = 0; i < nEl; i++) {
                float v = out[i];
                if (!(v == v && v > -1e30 && v < 1e30)) { if (bad < 4) fprintf(stderr, "[conv] %s: 非有限值 @%llu\n", name.c_str(), (unsigned long long)i); bad++; }
                else { double a = v < 0 ? -(double)v : (double)v; if (a > mx) mx = a; }
            }
            if (bad || mx > 1e4) { fprintf(stderr, "[conv] %s 反量化异常（bad=%ld maxabs=%g）——检查 gtype/块大小\n", name.c_str(), bad, mx); exit(1); }
        }
        return out;
    }

    static void dequant(const u8* src, const GTensor& t, float* y) {
        u64 K = t.ne[0], N = 1; for (size_t d = 1; d < t.ne.size(); d++) N *= t.ne[d];
        switch (t.gtype) {
        case 0: memcpy(y, src, K * N * 4); break;
        case 1: { const u16* s = (const u16*)src; for (u64 i = 0; i < K * N; i++) y[i] = f16_to_f32(s[i]); break; }
        case 30: { const u16* s = (const u16*)src; for (u64 i = 0; i < K * N; i++) y[i] = bf16_to_f32(s[i]); break; }
        case 8: { /* Q8_0: 每行 32 elt 一块，块 34B（2B f16 d + 32 i8）
            （曾把行跨度写成 18B/块 —— 那是 Q4_0 的块大小，Q8_0 是 34B。
             行 0 正确、行 1 起全部错位，2026-11-01 修） */
            for (u64 r = 0; r < N; r++) {
                const u8* p = src + r * (K / 32 * 34);
                for (u64 b = 0; b < K / 32; b++) {
                    float d = f16_to_f32(*(const u16*)p); p += 2;
                    for (int j = 0; j < 32; j++) { y[r * K + b * 32 + j] = d * (i8)p[j]; }
                    p += 32;
                }
            }
            break;
        }
        case 13: { /* Q5_K：块 176B = d(2)+dmin(2)+scales(12)+qh(32)+qs(128) */
            for (u64 r = 0; r < N; r++) {
                const u8* p = src + r * (K / 256 * 176);
                float* yr = y + r * K;
                for (u64 b = 0; b < K / 256; b++) {
                    float d = f16_to_f32(*(const u16*)p);
                    float dm = f16_to_f32(*(const u16*)(p + 2));
                    const u8* qs = p + 4;
                    const u8* qh = p + 16;
                    const u8* ql = p + 48;
                    int is = 0; u8 u1 = 1, u2 = 2;
                    float* yy = yr + b * 256;
                    for (int j = 0; j < 256; j += 64) {
                        u8 sc, m;
                        get_scale_min_k4(is + 0, qs, &sc, &m);
                        float d1 = d * sc, m1 = dm * m;
                        get_scale_min_k4(is + 1, qs, &sc, &m);
                        float d2 = d * sc, m2 = dm * m;
                        for (int l = 0; l < 32; l++) yy[j + l] = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
                        for (int l = 0; l < 32; l++) yy[j + 32 + l] = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
                        ql += 32; is += 2; u1 <<= 2; u2 <<= 2;
                    }
                    p += 176;
                }
            }
            break;
        }
        default: fprintf(stderr, "转换器不支持 gtype %u（专家/Engram 不走这里）\n", t.gtype); exit(1);
        }
    }
    static void get_scale_min_k4(int j, const u8* q, u8* d, u8* m) {
        if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
        else { *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4); }
    }
};

// ---------------- affine 量化（MLX 语义 dequant = q*scale + bias） ----------------
struct QOut { std::vector<u32> q; std::vector<u16> scales, biases; int bits, gs; };

static QOut quantize_affine(const float* w, u64 K, u64 N, int bits, int gs) {
    QOut o; o.bits = bits; o.gs = gs;
    int pf = 32 / bits;                 /* 每 u32 装几个 */
    u64 wordsPerRow = K / pf;
    o.q.resize(wordsPerRow * N);
    o.scales.resize(K / gs * N);
    o.biases.resize(K / gs * N);
    int qmax = (1 << bits) - 1;
    double rms = 0, mx = 0;
    for (u64 r = 0; r < N; r++) {
        const float* wr = w + r * K;
        for (u64 g = 0; g < K / gs; g++) {
            float mn = 1e30f, mxx = -1e30f;
            for (int j = 0; j < gs; j++) { float v = wr[g * gs + j]; mn = std::min(mn, v); mxx = std::max(mxx, v); }
            float step = (mxx - mn) / qmax;
            u16 s16 = f32_to_f16(step), b16 = f32_to_f16(mn);
            float sf = f16_to_f32(s16), bf = f16_to_f32(b16); /* 用 f16 化后的参数编 q，减小往返误差 */
            o.scales[r * (K / gs) + g] = s16;
            o.biases[r * (K / gs) + g] = b16;
            u32* qrow = o.q.data() + r * wordsPerRow;
            for (int j = 0; j < gs; j++) {
                int q = (sf == 0) ? 0 : (int)lrintf((wr[g * gs + j] - bf) / sf);
                q = std::max(0, std::min(qmax, q));
                rms += ((double)q * sf + bf - wr[g * gs + j]) * ((double)q * sf + bf - wr[g * gs + j]);
                mx = std::max(mx, std::fabs((double)q * sf + bf - wr[g * gs + j]));
                u64 wi = g * (u64)gs + j;
                qrow[wi / pf] |= (u32)q << ((wi % pf) * bits);
            }
        }
    }
    printf("    量化噪声 rms=%.5f max=%.4f\n", sqrt(rms / (double)(K * N)), mx);
    return o;
}

// ---------------- safetensors ----------------
struct Blob { std::string name; std::string dtype; std::vector<u64> shape; std::vector<u8> data; };

static void run(const std::string& prefix, int nshards, const char* outdir, int bits) {
    GGUF g;
    g.open(prefix, nshards);
    printf("张量 %zu 项，kv %zu 项，token_map %zu\n", g.tensors.size(), g.arr_kv.size() + g.scalar_kv.size(), g.token_map.size());

    std::vector<Blob> blobs;
    double tot4 = 0, tot16 = 0;
    std::vector<std::string> names;
    for (auto& kv : g.tensors) names.push_back(kv.first);
    std::sort(names.begin(), names.end());
    for (auto& nm : names) {
        if (nm.find("_exps") != std::string::npos) continue;
        if (nm.find("per_layer_token_embd") != std::string::npos) continue;
        const GTensor& t = g.tensors[nm];
        u64 K = t.ne[0], N = 1; for (size_t d = 1; d < t.ne.size(); d++) N *= t.ne[d];
        bool isNorm = t.gtype == 0 || nm.find("norm") != std::string::npos || nm.find("sinks") != std::string::npos ||
                      nm.find("bias") != std::string::npos;
        printf("  %s [%llu,%llu] gtype%u ", nm.c_str(), (unsigned long long)N, (unsigned long long)K, t.gtype);
        if (isNorm) {
            std::vector<float> a = g.read_f32(nm);
            Blob b; b.name = nm; b.dtype = "F32"; b.shape = {N, K};
            b.data.resize(N * K * 4); memcpy(b.data.data(), a.data(), N * K * 4);
            blobs.push_back(std::move(b)); printf("f32\n"); continue;
        }
        u64 nEl = K * N;
        if (nEl > 8ull * 1024 * 1024) {
            std::vector<float> a = g.read_f32(nm);
            QOut q = quantize_affine(a.data(), K, N, bits, 64);
            int pf = 32 / bits;
            Blob bw; bw.name = nm; bw.dtype = "U32"; bw.shape = {N, K / pf};
            bw.data.resize(q.q.size() * 4); memcpy(bw.data.data(), q.q.data(), q.q.size() * 4);
            Blob bs; bs.name = nm + ".scales"; bs.dtype = "F16"; bs.shape = {N, K / 64};
            bs.data.resize(q.scales.size() * 2); memcpy(bs.data.data(), q.scales.data(), q.scales.size() * 2);
            Blob bb; bb.name = nm + ".biases"; bb.dtype = "F16"; bb.shape = {N, K / 64};
            bb.data.resize(q.biases.size() * 2); memcpy(bb.data.data(), q.biases.data(), q.biases.size() * 2);
            blobs.push_back(std::move(bw)); blobs.push_back(std::move(bs)); blobs.push_back(std::move(bb));
            tot4 += (double)nEl * bits / 8 + (double)nEl / 64 * 4;
            printf("q%d -> %.1fMB\n", bits, (double)nEl * bits / 8 / 1e6 + (double)nEl / 64 * 4 / 1e6);
        } else {
            std::vector<float> a = g.read_f32(nm);
            Blob b; b.name = nm; b.dtype = "F16"; b.shape = {N, K};
            b.data.resize(nEl * 2);
            u16* d = (u16*)b.data.data();
            for (u64 i = 0; i < nEl; i++) d[i] = f32_to_f16(a[i]);
            blobs.push_back(std::move(b));
            tot16 += nEl * 2;
            printf("f16 -> %.1fMB\n", nEl * 2.0 / 1e6);
        }
    }
    /* 元数据张量 */
    if (!g.token_map.empty()) {
        Blob b; b.name = "engram_token_map"; b.dtype = "I32"; b.shape = {g.token_map.size()};
        b.data.resize(g.token_map.size() * 4);
        memcpy(b.data.data(), g.token_map.data(), b.data.size());
        blobs.push_back(std::move(b));
        printf("  engram_token_map [%zu] i32\n", g.token_map.size());
    }
    auto put_arr = [&](const char* name, const std::string& key, const std::string& dt) {
        auto it = g.arr_kv.find(key);
        if (it == g.arr_kv.end()) return;
        Blob b; b.name = name; b.dtype = dt; b.shape = {it->second.size()};
        size_t es = dt == "U64" ? 8 : 4;
        b.data.resize(it->second.size() * es);
        for (size_t i = 0; i < it->second.size(); i++) {
            if (dt == "U64") { u64 v = (u64)it->second[i]; memcpy(b.data.data() + i * 8, &v, 8); }
            else { u32 v = (u32)it->second[i]; memcpy(b.data.data() + i * 4, &v, 4); }
        }
        blobs.push_back(std::move(b));
        printf("  %s [%zu] %s\n", name, it->second.size(), dt.c_str());
    };
    put_arr("engram_head_offsets", "deepseek41.ple.head_offsets", "U32");
    put_arr("engram_primes", "deepseek41.ple.head_vocab_sizes", "U32");
    put_arr("engram_mults", "deepseek41.ple.layer_multipliers", "U64");
    put_arr("kv_source_layers", "deepseek41.attention.kv_source_layers", "U32");
    put_arr("index_source_layers", "deepseek41.attention.indexer.source_layers", "U32");
    put_arr("compress_ratios", "deepseek41.attention.compress_ratios", "U32");

    /* 写 safetensors */
    mkdir(outdir, 0755);
    std::string outPath = std::string(outdir) + "/model.safetensors";
    std::ostringstream hdr;
    hdr << "{";
    hdr << "\"__metadata__\":{\"format\":\"ds41-mlx-dense\"}";
    u64 off = 0;
    /* 先算偏移 */
    std::vector<std::pair<u64, u64>> spans(blobs.size());
    for (size_t i = 0; i < blobs.size(); i++) {
        spans[i] = {off, off + blobs[i].data.size()};
        off += blobs[i].data.size();
        hdr << ",\"" << blobs[i].name << "\":{\"dtype\":\"" << blobs[i].dtype << "\",\"shape\":[";
        for (size_t d = 0; d < blobs[i].shape.size(); d++) hdr << (d ? "," : "") << blobs[i].shape[d];
        hdr << "],\"data_offsets\":[" << spans[i].first << "," << spans[i].second << "]}";
    }
    hdr << "}";
    std::string hs = hdr.str();
    while (hs.size() % 8) hs += " ";
    FILE* fo = fopen(outPath.c_str(), "wb");
    u64 hlen = hs.size();
    fwrite(&hlen, 8, 1, fo);
    fwrite(hs.data(), 1, hs.size(), fo);
    for (size_t i = 0; i < blobs.size(); i++) fwrite(blobs[i].data.data(), 1, blobs[i].data.size(), fo);
    fclose(fo);
    printf("写 %s：header %llu B + data %.2f GB（q%d %.2f + f16 %.2f GB）\n",
           outPath.c_str(), (unsigned long long)(hlen + 8), (tot4 + tot16) / 1e9, bits, tot4 / 1e9, tot16 / 1e9);
}

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "用法: %s <gguf前缀含%%d> <分片数> <输出目录> [bits=4]\n", argv[0]); return 1; }
    int bits = argc > 4 ? atoi(argv[4]) : 4;
    run(argv[1], atoi(argv[2]), argv[3], bits);
    return 0;
}
