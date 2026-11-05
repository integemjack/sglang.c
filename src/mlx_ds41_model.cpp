/* DeepSeek-V4.1-Flash MLX 引擎（真实结构，Strata 式专家/Engram 卸载）
 *
 * 数学路径沿用 mlx_ds41.cpp（mini 金标准对拍 prefill/decode 全部 fp32 噪声级 1.2e-07），
 * 结构按 GGUF 实测（40 层双源压缩、HC4+Sinkhorn20、384 专家 top6、Engram@L1/L14），
 * 语义对照 llama.cpp src/models/deepseek41.cpp（与官方 model.py/engram.py 同源）。
 *
 * 权重三处来源：
 *   - mlx-dense/model.safetensors：稠密/注意力（affine 4bit g64 + f16 + f32），mx::load_safetensors
 *   - 专家 GGUF（IQ1_M/IQ2_XXS/IQ2_XS 混合）：mmap，按 (层,专家) 反量化 → fp16 mx::array，LRU 缓存
 *   - Engram 表 GGUF（IQ4_NL [256,768M]）：mmap 逐行取（哈希行号），host 反量化
 *
 * 短上下文精确快路：prompt ≤ MAXCTX 时压缩组数 ≤ index_topk(512)，官方索引器必选全 →
 * 稀疏注意力等价稠密（全压缩行可见），跳过索引器评分。MAXCTX=512，B=1。
 */
#include <mlx/mlx.h>
#include "ds41_iq_tables.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace fs41 {

using namespace mlx::core;
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef int8_t i8;
typedef int32_t i32;
typedef int64_t i64;

#define MAXCTX 512
#define MAXB 1
static const int WIN = 128;

static array ia(int v) { return array({v}, int32); }
static array fa(float v) { return array({v}, float32); }
static array f32(const array& x) { return astype(x, float32); }
static int item_i(const array& a, int i) {
    auto x = astype(slice(reshape(a, {-1}), Shape{i}, Shape{i + 1}), int32);
    eval(x);
    return x.item<int>();
}

/* ---------------- 配置（config.json 迷你提取） ---------------- */
struct Cfg {
    int L = 40, D = 5120, H = 64, DH = 512, ROT = 64, QLORA = 1280;
    int G = 8, ORANK = 1024, HC = 4, SINKHORN = 20;
    int NE = 384, TOPK = 6, FFN = 2304, V = 129280;
    float HC_EPS = 1e-6f, ROUTE_SCALE = 1.5f, NORM_EPS = 1e-6f, SWIGLU_L = 10.f;
    std::vector<int> ratios, kv_src_raw;
    std::vector<int> kv_of; /* 每层 → 其压缩 KV 源层（-1 纯窗口） */
    std::vector<float> clamp_exp, clamp_shexp;
    float base = 10000.f, comp_base = 160000.f, yarn_f = 16.f, beta_fast = 32.f, beta_slow = 1.f;
    int orig_ctx = 65536;
    std::vector<int> eng_layers; /* [1,14] */
    int ngram = 4, hpn = 8, eng_hd = 256, pad_id = 2;
    std::vector<u32> eng_off_h, eng_prime_h;
    std::vector<u64> eng_mult_h;
    std::string dir;
    std::vector<std::string> exp_shards;
    std::string eng_file;
};

static std::string json_slug(const std::string& s, const char* key) {
    std::string pat = std::string("\"") + key + "\"";
    size_t p = s.find(pat);
    if (p == std::string::npos) return "";
    p = s.find(':', p + pat.size());
    if (p == std::string::npos) return "";
    p++;
    while (p < s.size() && (s[p] == ' ' || s[p] == '\n' || s[p] == '\r' || s[p] == '\t')) p++;
    size_t e = p;
    if (s[p] == '[') {
        int dep = 0;
        for (e = p; e < s.size(); e++) {
            if (s[e] == '[') dep++;
            if (s[e] == ']') { dep--; if (!dep) { e++; break; } }
        }
    } else if (s[p] == '"') {
        for (e = p + 1; e < s.size(); e++)
            if (s[e] == '"' && s[e - 1] != '\\') { e++; break; }
    } else {
        while (e < s.size() && s[e] != ',' && s[e] != '}' && s[e] != '\n') e++;
    }
    return s.substr(p, e - p);
}
static std::vector<double> json_arr(const std::string& t) {
    std::vector<double> v;
    std::stringstream ss(t);
    std::string tok;
    while (ss >> tok) {
        tok.erase(std::remove(tok.begin(), tok.end(), '['), tok.end());
        tok.erase(std::remove(tok.begin(), tok.end(), ']'), tok.end());
        tok.erase(std::remove(tok.begin(), tok.end(), ','), tok.end());
        if (tok.empty()) continue;
        try { v.push_back(std::stod(tok)); } catch (...) {}
    }
    return v;
}

static bool load_cfg(Cfg& c, const std::string& dir) {
    c.dir = dir;
    std::ifstream f(dir + "/config.json");
    if (!f) return false;
    std::stringstream ss;
    ss << f.rdbuf();
    std::string s = ss.str();
    auto num = [&](const char* k, double def) {
        std::string v = json_slug(s, k);
        try { return v.empty() ? def : std::stod(v); } catch (...) { return def; }
    };
    c.L = (int)num("deepseek41.block_count", 40);
    c.D = (int)num("deepseek41.embedding_length", 5120);
    c.H = (int)num("deepseek41.attention.head_count", 64);
    c.DH = (int)num("deepseek41.attention.key_length", 512);
    c.QLORA = (int)num("deepseek41.attention.q_lora_rank", 1280);
    c.G = (int)num("deepseek41.attention.output_group_count", 8);
    c.ORANK = (int)num("deepseek41.attention.output_lora_rank", 1024);
    c.HC = (int)num("deepseek41.hyper_connection.count", 4);
    c.SINKHORN = (int)num("deepseek41.hyper_connection.sinkhorn_iterations", 20);
    c.HC_EPS = (float)num("deepseek41.hyper_connection.epsilon", 1e-6);
    c.NE = (int)num("deepseek41.expert_count", 384);
    c.TOPK = (int)num("deepseek41.expert_used_count", 6);
    c.FFN = (int)num("deepseek41.expert_feed_forward_length", 2304);
    c.ROUTE_SCALE = (float)num("deepseek41.expert_weights_scale", 1.5);
    c.SWIGLU_L = (float)num("deepseek41.swiglu_clamp_exp", 10.0);
    c.ROT = (int)num("deepseek41.rope.dimension_count", 64);
    c.base = (float)num("deepseek41.rope.freq_base", 10000);
    c.comp_base = (float)num("deepseek41.attention.compress_rope_freq_base", 160000);
    c.yarn_f = (float)num("deepseek41.rope.scaling.factor", 16);
    c.beta_fast = (float)num("deepseek41.rope.scaling.yarn_beta_fast", 32);
    c.beta_slow = (float)num("deepseek41.rope.scaling.yarn_beta_slow", 1);
    c.orig_ctx = (int)num("deepseek41.rope.scaling.original_context_length", 65536);
    c.ngram = (int)num("deepseek41.ple.ngram_size", 4);
    c.hpn = (int)num("deepseek41.ple.heads_per_ngram", 8);
    c.eng_hd = (int)num("deepseek41.embedding_length_per_layer_input", 256);
    c.pad_id = (int)num("deepseek41.engram.pad_token_id", 2);
    for (auto v : json_arr(json_slug(s, "deepseek41.attention.compress_ratios"))) c.ratios.push_back((int)v);
    for (auto v : json_arr(json_slug(s, "deepseek41.attention.kv_source_layers"))) c.kv_src_raw.push_back((int)v);
    for (auto v : json_arr(json_slug(s, "deepseek41.ple.layers"))) c.eng_layers.push_back((int)v);
    for (auto v : json_arr(json_slug(s, "deepseek41.ple.head_offsets"))) c.eng_off_h.push_back((u32)v);
    for (auto v : json_arr(json_slug(s, "deepseek41.ple.head_vocab_sizes"))) c.eng_prime_h.push_back((u32)v);
    for (auto v : json_arr(json_slug(s, "deepseek41.ple.layer_multipliers"))) c.eng_mult_h.push_back((u64)v);
    if (c.ratios.size() > (size_t)c.L) c.ratios.resize(c.L); /* 后面是 MTP 草稿层的 */
    for (auto v : json_arr(json_slug(s, "deepseek41.swiglu_clamp_exp"))) c.clamp_exp.push_back((float)v);
    for (auto v : json_arr(json_slug(s, "deepseek41.swiglu_clamp_shexp"))) c.clamp_shexp.push_back((float)v);
    c.NORM_EPS = (float)num("deepseek41.attention.layer_norm_rms_epsilon", 1e-6);
    if (c.NORM_EPS <= 0 || c.NORM_EPS > 1e-3) c.NORM_EPS = 1e-6f; /* 元数据 1e-20 不合理时兜底 */
    /* 每层的压缩 KV 源（≤该层的最近源层） */
    c.kv_of.assign(c.L, -1);
    int last = -1;
    for (int l = 0; l < c.L; l++) {
        for (int k : c.kv_src_raw)
            if (k == l) last = l;
        if (c.ratios[l] > 0) c.kv_of[l] = last;
    }
    for (int i = 1; i <= 8; i++) {
        char p[1024];
        snprintf(p, sizeof p, "../DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-%05d-of-00006.gguf", i);
        std::ifstream t(dir + "/" + p);
        if (!t) break;
        c.exp_shards.push_back(dir + "/" + p);
    }
    c.eng_file = dir + "/../DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf";
    if (c.ratios.size() != (size_t)c.L) fprintf(stderr, "cfg: ratios %zu != L %d\n", c.ratios.size(), c.L);
    if (c.exp_shards.empty()) fprintf(stderr, "cfg: no expert shards\n");
    return c.ratios.size() == (size_t)c.L && !c.exp_shards.empty();
}

/* ---------------- GGUF 读取（专家分片 + Engram 表） ---------------- */
struct GTen { int fi = 0; u64 off = 0; u32 gt = 0; std::vector<u64> ne; u64 nbytes = 0; };

struct GG {
    std::vector<int> fds;
    std::vector<u8*> maps;
    std::unordered_map<std::string, GTen> ten;

    static u64 rd64(const u8* p) { u64 v; memcpy(&v, p, 8); return v; }
    static u32 rd32(const u8* p) { u32 v; memcpy(&v, p, 4); return v; }
    static std::string rstr(const u8*& p) { u64 n = rd64(p); p += 8; std::string s((const char*)p, n); p += n; return s; }
    static int elt_sz(int t) { return (t == 2 || t == 3) ? 2 : (t == 7) ? 1 : (t >= 10 && t <= 12) ? 8 : (t >= 4 && t <= 6) ? 4 : 1; }
    static u64 nblk_bytes(u32 gt, u64 nEl) {
        switch (gt) {
        case 0: return nEl * 4;          /* F32 */
        case 1: return nEl;              /* F16 */
        case 30: return nEl * 2;         /* BF16 */
        case 8: return nEl / 32 * 34;    /* Q8_0 */
        case 13: return nEl / 256 * 176; /* Q5_K */
        case 14: return nEl / 256 * 210; /* Q6_K */
        case 12: return nEl / 256 * 144; /* Q4_K */
        case 15: return nEl / 256 * 173; /* IQ2_S */
        case 16: return nEl / 256 * 66;  /* IQ2_XXS */
        case 17: return nEl / 256 * 74;  /* IQ2_XS */
        case 19: return nEl / 256 * 50;  /* IQ1_S */
        case 29: return nEl / 256 * 56;  /* IQ1_M */
        case 20: return nEl / 32 * 18;   /* IQ4_NL */
        case 18: return nEl / 256 * 104; /* IQ3_XXS */
        case 22: return nEl / 256 * 122; /* IQ3_S */
        case 23: return nEl / 256 * 164; /* IQ2_S fallback */
        case 24: return nEl / 32 * 20;   /* IQ4_XS */
        default: fprintf(stderr, "gg: gtype %u 不支持\n", gt); exit(1);
        }
    }
    void open(const std::vector<std::string>& paths) {
        for (int fi = 0; fi < (int)paths.size(); fi++) {
            int fd = ::open(paths[fi].c_str(), O_RDONLY);
            if (fd < 0) { fprintf(stderr, "gg open 失败 %s\n", paths[fi].c_str()); exit(1); }
            struct stat st;
            fstat(fd, &st);
            u8* m = (u8*)mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
            madvise(m, st.st_size, MADV_RANDOM);
            fds.push_back(fd);
            maps.push_back(m);
            const u8* p = m;
            p += 8;
            u64 n_ten = rd64(p); p += 8;
            u64 n_kv = rd64(p); p += 8;
            for (u64 k = 0; k < n_kv; k++) {
                rstr(p);
                u32 t = rd32(p); p += 4;
                p = skip(t, p);
            }
            /* GGUF 规范：张量偏移相对于「头全部（含张量表）结束后对齐」的数据区起点。
             * 旧代码在张量表之前就取了 hb（等于把基址提前了几十 KB），mini 对拍
             * 因转换器同错相消而没暴露；真权重（Strata 标准偏移）下 3–5% 块读到
             * 邻区字节 → NaN。2026-11-01 修：先解析完张量表再对齐。 */
            for (u64 k = 0; k < n_ten; k++) {
                std::string name = rstr(p);
                u32 nd = rd32(p); p += 4;
                GTen t;
                t.fi = fi;
                t.ne.resize(nd);
                for (u32 d = 0; d < nd; d++) t.ne[d] = rd64(p), p += 8;
                t.gt = rd32(p); p += 4;
                t.off = rd64(p); p += 8;
                u64 nEl = 1;
                for (auto d : t.ne) nEl *= d;
                t.nbytes = nblk_bytes(t.gt, nEl);
                ten[name] = t;
            }
            u64 hb = ((u64)(p - m) + 31) / 32 * 32; /* 张量表之后才是数据区 */
            for (auto& kv2 : ten) {
                if (kv2.second.fi == fi) kv2.second.off += hb;
            }
        }
    }
    static const u8* skip(u32 t, const u8* p) {
        if (t == 8) { u64 n = rd64(p); return p + 8 + n; }
        if (t == 9) {
            u32 et = rd32(p); p += 4;
            u64 n = rd64(p); p += 8;
            if (et == 8)
                for (u64 i = 0; i < n; i++) { u64 m = rd64(p); p += 8 + m; }
            else
                p += n * (u64)elt_sz(et);
            return p;
        }
        return p + (u64)elt_sz(t);
    }
};

/* IQ 反量化（照抄 llama.cpp dequantize_row_*） */
static float f16_to_f32(u16 h) {
    u32 s = (u32)(h & 0x8000) << 16, e = (h >> 10) & 31, m = h & 1023;
    u32 f = e ? (s | ((e - 15 + 127) << 23) | (m << 13)) : (m ? (s | (127 - 15) << 23) : s);
    if (e == 31) f = s | 0x7f800000;
    float out;
    memcpy(&out, &f, 4);
    return out;
}
static void deq_iq2_xxs(const u8* blk, float* y, i64 n) {
    const i64 nb = n / 256;
    for (i64 i = 0; i < nb; i++) {
        const u8* p = blk + i * 66;
        float d = f16_to_f32(*(const u16*)p);
        const u8* qs = p + 2;
        for (int ib32 = 0; ib32 < 8; ib32++) {
            u32 a32[2];
            memcpy(a32, qs + 4 * ib32, 8);
            const u8* a8 = (const u8*)a32;
            float db = d * (0.5f + (float)(a32[1] >> 28)) * 0.25f;
            for (int l = 0; l < 4; l++) {
                const u8* grid = (const u8*)(ds41_iq::iq2xxs_grid + a8[l]);
                u8 signs = ds41_iq::ksigns_iq2xs[(a32[1] >> (7 * l)) & 127];
                for (int j = 0; j < 8; j++)
                    *y++ = db * grid[j] * (signs & ds41_iq::kmask_iq2xs[j] ? -1.f : 1.f);
            }
        }
    }
}
static void deq_iq2_xs(const u8* blk, float* y, i64 n) {
    const i64 nb = n / 256;
    for (i64 i = 0; i < nb; i++) {
        const u8* p = blk + i * 74;
        float d = f16_to_f32(*(const u16*)p);
        const u8* qs = p + 2;
        const u8* sc = qs + 64;
        for (int ib32 = 0; ib32 < 8; ib32++) {
            float db[2] = {d * (0.5f + (float)(sc[ib32] & 0xf)) * 0.25f,
                           d * (0.5f + (float)(sc[ib32] >> 4)) * 0.25f};
            for (int l = 0; l < 4; l++) {
                u16 v;
                memcpy(&v, qs + 4 * ib32 + l, 2);
                const u8* grid = (const u8*)(ds41_iq::iq2xs_grid + (v & 511));
                u8 signs = ds41_iq::ksigns_iq2xs[v >> 9];
                for (int j = 0; j < 8; j++)
                    *y++ = db[l / 2] * grid[j] * (signs & ds41_iq::kmask_iq2xs[j] ? -1.f : 1.f);
            }
        }
    }
}
static void deq_iq1_m(const u8* blk, float* y, i64 n) {
    const i64 nb = n / 256;
    const float ID = 0.125f;
    for (i64 i = 0; i < nb; i++) {
        const u8* p = blk + i * 56;
        const u8* qs = p;
        const u8* qh = p + 32;
        const u16* sc = (const u16*)(p + 48);
        u16 su = (u16)((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));
        float d = f16_to_f32(su);
        for (int ib = 0; ib < 8; ib++) {
            float dl1 = d * (2.f * (float)((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 7) + 1.f);
            float dl2 = d * (2.f * (float)((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 7) + 1.f);
            u16 idx[4];
            idx[0] = (u16)(qs[0] | ((qh[0] << 8) & 0x700));
            idx[1] = (u16)(qs[1] | ((qh[0] << 4) & 0x700));
            idx[2] = (u16)(qs[2] | ((qh[1] << 8) & 0x700));
            idx[3] = (u16)(qs[3] | ((qh[1] << 4) & 0x700));
            float ddl[4] = {dl1, dl1, dl2, dl2};
            float dl[4] = {qh[0] & 0x08 ? -ID : ID, qh[0] & 0x80 ? -ID : ID,
                           qh[1] & 0x08 ? -ID : ID, qh[1] & 0x80 ? -ID : ID};
            for (int l = 0; l < 4; l++) {
                const i8* grid = (const i8*)(ds41_iq::iq1s_grid + idx[l]);
                for (int j = 0; j < 8; j++) *y++ = ddl[l] * (grid[j] + dl[l]);
            }
            qs += 4;
            qh += 2;
        }
    }
}
static void deq_iq4_nl(const u8* blk, float* y, i64 n) {
    const i64 nb = n / 32;
    for (i64 i = 0; i < nb; i++) {
        const u8* p = blk + i * 18;
        float d = f16_to_f32(*(const u16*)p);
        const u8* qs = p + 2;
        for (int j = 0; j < 16; j++) {
            y[j] = d * ds41_iq::kvalues_iq4nl[qs[j] & 0xf];
            y[j + 16] = d * ds41_iq::kvalues_iq4nl[qs[j] >> 4];
        }
        y += 32;
    }
}

/* ---------------- 模型 ---------------- */
struct Model {
    Cfg c;
    std::unordered_map<std::string, array> W;
    struct Comp {
        int src = -1, ratio = 1;
        array cache = zeros({1}, float32);
        array kv_state = zeros({1}, float32), score_state = zeros({1}, float32);
    };
    std::vector<array> wkv; /* 每层 [B, WIN, DH] ring */
    std::vector<Comp> comps;
    std::vector<int> comp_idx; /* 每层 → comps 下标（-1 无） */
    array token_map = zeros({1}, float32);
    std::vector<i32> tmap_h;
    GG gg;
    struct ExpKeyCmp { bool operator()(const std::pair<int, int>& a, const std::pair<int, int>& b) const {
        return a.first != b.first ? a.first < b.first : a.second < b.second; } };
    std::map<std::pair<int, int>, array, ExpKeyCmp> exp_cache;
    size_t exp_bytes = 0, exp_cap = (size_t)4 << 30;
    std::vector<std::pair<std::pair<int, int>, size_t>> exp_lru;
    array win_cos_d = zeros({1}, float32), win_sin_d = zeros({1}, float32);
    array win_plain_d = zeros({1}, float32), win_plain_sin_d = zeros({1}, float32);
    array cmp_cos_d = zeros({1}, float32), cmp_sin_d = zeros({1}, float32);
    int pos = 0;
    u8* eng_map = nullptr;
    size_t eng_map_sz = 0;
    u64 eng_rows = 0;
    int eng_fd = -1;
    std::vector<u32> hist;

    array Wq(const std::string& n) {
        auto it = W.find(n);
        if (it == W.end()) { fprintf(stderr, "[缺权重] %s\n", n.c_str()); exit(1); }
        return it->second;
    }
    /* 加载后校验：找含非有限值的张量 */
    void wcheck() {
        int bad = 0;
        for (auto& kv2 : W) {
            auto a = kv2.second;
            auto al = all(isfinite(a));
            al.eval();
            if (!al.item<bool>()) {
                fprintf(stderr, "[wchk] 非有限: %s\n", kv2.first.c_str());
                if (++bad > 20) break;
            }
        }
        fprintf(stderr, "[wchk] done bad=%d\n", bad);
    }
    /* 解量化到 f32（大权重是 q4 三件套；小权重直接 astype） */
    array Wd(const std::string& n) {
        if (!W.count(n)) { fprintf(stderr, "[缺权重] %s\n", n.c_str()); exit(1); }
        if (W.count(n + ".scales"))
            return dequantize(W.at(n), W.at(n + ".scales"), W.at(n + ".biases"), 64, 4);
        return astype(W.at(n), float32);
    }
    array lin(const array& x, const std::string& n) {
        auto it = W.find(n);
        if (it == W.end()) { fprintf(stderr, "缺权重 %s\n", n.c_str()); exit(1); }
        if (W.count(n + ".scales"))
            return quantized_matmul(x, it->second, W.at(n + ".scales"), W.at(n + ".biases"), true, 64, 4);
        return matmul(x, transpose(it->second));
    }
    static array rms(const array& x, const array& w, float eps) {
        return f32(x) * rsqrt(mean(square(f32(x)), -1, true) + fa(eps)) * f32(w);
    }

    /* host yarn 频率（官方 precompute_freqs_cis；apply_rotary 无幅度缩放） */
    static void yarn_freqs(int dim, int seqlen, int orig, float base, float factor, float bf, float bs,
                           std::vector<float>& cos, std::vector<float>& sin) {
        std::vector<double> fr(dim / 2);
        for (int i = 0; i < dim / 2; i++) fr[i] = std::pow((double)base, -2.0 * i / dim);
        if (orig > 0) {
            auto corr = [&](double rot) {
                return dim * std::log(orig / (rot * 2 * M_PI)) / (2 * std::log((double)base));
            };
            int lo = std::max((int)std::floor(corr(bf)), 0);
            int hi = std::min((int)std::ceil(corr(bs)), dim - 1);
            for (int i = 0; i < dim / 2; i++) {
                double ramp = (i - lo) / std::max(hi - lo, 1);
                ramp = std::max(0.0, std::min(1.0, ramp));
                double sm = 1 - ramp;
                fr[i] = fr[i] / factor * (1 - sm) + fr[i] * sm;
            }
        }
        cos.resize((size_t)seqlen * (dim / 2));
        sin.resize((size_t)seqlen * (dim / 2));
        for (int p = 0; p < seqlen; p++)
            for (int i = 0; i < dim / 2; i++) {
                double a = p * fr[i];
                cos[(size_t)p * (dim / 2) + i] = (float)std::cos(a);
                sin[(size_t)p * (dim / 2) + i] = (float)std::sin(a);
            }
    }
    static array freq_slice(const array& tbl, int pos0, int n) {
        return slice(tbl, Shape{pos0, 0}, Shape{pos0 + n, tbl.shape(1)});
    }
    static std::pair<Shape, Shape> last_slice(const Shape& sh, int b, int e) {
        Shape a(sh.size(), 0), bb(sh.size(), 0);
        for (size_t i = 0; i < sh.size(); i++) bb[i] = sh[i];
        a[sh.size() - 1] = b;
        bb[sh.size() - 1] = e;
        return {a, bb};
    }
    static Shape shp_cat(const Shape& a, const Shape& b) {
        Shape r = a;
        r.insert(r.end(), b.begin(), b.end());
        return r;
    }
    /* 相邻对旋转（官方 apply_rotary_emb；inv=共轭）。cs/sn [n, RD/2]，x 末维 = rd */
    static array rotary(const array& x, const array& cs, const array& sn, bool inv) {
        auto full = x.shape();
        int nd = x.ndim();
        int rd = full[nd - 1];
        Shape mid(full.begin(), full.end() - 1);
        auto pair = reshape(x, shp_cat(mid, {rd / 2, 2}));
        auto sl0 = last_slice(pair.shape(), 0, 1);
        auto x0 = reshape(slice(pair, sl0.first, sl0.second), shp_cat(mid, {rd / 2}));
        auto sl1 = last_slice(pair.shape(), 1, 2);
        auto x1 = reshape(slice(pair, sl1.first, sl1.second), shp_cat(mid, {rd / 2}));
        float s = inv ? -1.f : 1.f;
        /* cs/sn [n, rd/2]：位置在 x0 的第 0 维，广播成 {n, 1.., rd/2} */
        Shape ones(mid.size() - 1, 1);
        Shape csb = shp_cat({cs.shape(0)}, shp_cat(ones, {cs.shape(1)}));
        auto cB = broadcast_to(reshape(cs, csb), x0.shape());
        auto sB = broadcast_to(reshape(astype(sn, float32) * fa(s), csb), x0.shape());
        auto y0 = f32(x0) * cB - f32(x1) * sB;
        auto y1 = f32(x0) * sB + f32(x1) * cB;
        return astype(reshape(stack({y0, y1}, -1), full), x.dtype());
    }
    array rope_tail(const array& x, const array& cs, const array& sn, bool inv) {
        auto full = x.shape();
        int last = (int)full.back();
        auto slh = last_slice(full, 0, last - c.ROT);
        auto xh = slice(x, slh.first, slh.second);
        auto slt = last_slice(full, last - c.ROT, last);
        auto xt = slice(x, slt.first, slt.second);
        return concatenate({xh, rotary(xt, cs, sn, inv)}, -1);
    }

    bool load(const std::string& dir) {
        if (!load_cfg(c, dir)) { fprintf(stderr, "config 失败\n"); return false; }
        printf("[ds41] L=%d D=%d H=%d NE=%d TOPK=%d HC=%d srcs=%zu\n", c.L, c.D, c.H, c.NE, c.TOPK, c.HC,
               c.kv_src_raw.size());
        W = load_safetensors(dir + "/model.safetensors").first;
        printf("[ds41] dense %zu tensors\n", W.size());

        std::vector<float> t1, t2;
        yarn_freqs(c.ROT, MAXCTX, c.orig_ctx, c.base, c.yarn_f, c.beta_fast, c.beta_slow, t1, t2);
        win_cos_d = array(t1.data(), Shape{MAXCTX, c.ROT / 2}, float32);
        win_sin_d = array(t2.data(), Shape{MAXCTX, c.ROT / 2}, float32);
        yarn_freqs(c.ROT, MAXCTX, 0, c.base, 1, 0, 0, t1, t2);
        win_plain_d = array(t1.data(), Shape{MAXCTX, c.ROT / 2}, float32);
        win_plain_sin_d = array(t2.data(), Shape{MAXCTX, c.ROT / 2}, float32);
        yarn_freqs(c.ROT, MAXCTX, c.orig_ctx, c.comp_base, c.yarn_f, c.beta_fast, c.beta_slow, t1, t2);
        cmp_cos_d = array(t1.data(), Shape{MAXCTX, c.ROT / 2}, float32);
        cmp_sin_d = array(t2.data(), Shape{MAXCTX, c.ROT / 2}, float32);

        wkv.assign(c.L, zeros({1}, float32));
        comp_idx.assign(c.L, -1);
        for (int l = 0; l < c.L; l++) wkv[l] = zeros({MAXB, WIN, c.DH}, float32);
        for (int k = 0; k < (int)c.kv_src_raw.size(); k++) {
            Comp cp;
            cp.src = c.kv_src_raw[k];
            cp.ratio = c.ratios[cp.src];
            cp.cache = zeros({MAXB, MAXCTX / cp.ratio + 1, c.DH}, float32);
            cp.kv_state = zeros({MAXB, cp.ratio, c.DH}, float32);
            cp.score_state = full({MAXB, cp.ratio, c.DH}, -INFINITY, float32);
            comps.push_back(cp);
        }
        for (int l = 0; l < c.L; l++)
            for (int k = 0; k < (int)comps.size(); k++)
                if (comps[k].src == c.kv_of[l]) comp_idx[l] = k;

        token_map = W.at("engram_token_map");
        {
            auto tm = astype(token_map, int32);
            eval(tm);
            tmap_h.assign(tm.data<i32>(), tm.data<i32>() + tm.size());
        }
        {
            eng_fd = ::open(c.eng_file.c_str(), O_RDONLY);
            if (eng_fd < 0) { fprintf(stderr, "engram open fail\n"); return false; }
            struct stat st;
            fstat(eng_fd, &st);
            eng_map_sz = st.st_size;
            eng_map = (u8*)mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, eng_fd, 0);
            madvise(eng_map, st.st_size, MADV_RANDOM);
            GG g1;
            g1.open({c.eng_file});
            auto it = g1.ten.find("per_layer_token_embd.weight");
            eng_rows = it->second.ne[1];
            printf("[ds41] engram table %llu rows %.1fGB mmap\n", (unsigned long long)eng_rows,
                   eng_map_sz / 1e9);
        }
        gg.open(c.exp_shards);
        printf("[ds41] expert GGUF %zu shards %zu tensors\n", c.exp_shards.size(), gg.ten.size());
        return true;
    }

    void reset(void) {
        pos = 0;
        for (int l = 0; l < c.L; l++) wkv[l] = zeros({MAXB, WIN, c.DH}, float32);
        for (auto& cp : comps) {
            cp.cache = zeros({MAXB, MAXCTX / cp.ratio + 1, c.DH}, float32);
            cp.kv_state = zeros({MAXB, cp.ratio, c.DH}, float32);
            cp.score_state = full({MAXB, cp.ratio, c.DH}, -INFINITY, float32);
        }
    }

    /* ---- 专家取用（mmap → 反量化 → f16；LRU 按字节） ---- */
    static void deq_any(u32 gt, const u8* src, float* y, i64 n) {
        switch (gt) {
        case 16: deq_iq2_xxs(src, y, n); break;
        case 17: deq_iq2_xs(src, y, n); break;
        case 29: deq_iq1_m(src, y, n); break;
        default: fprintf(stderr, "deq_any gtype %u\n", gt); exit(1);
        }
    }
    array expert_get(int l, int e) {
        auto key = std::make_pair(l, e);
        auto it = exp_cache.find(key);
        if (it != exp_cache.end()) return it->second;
        char nm[96];
        std::vector<float> buf;
        array parts[3] = {zeros({1}, float32), zeros({1}, float32), zeros({1}, float32)};
        const char* names[3] = {"ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"};
        for (int t = 0; t < 3; t++) {
            snprintf(nm, sizeof nm, "blk.%d.%s.weight", l, names[t]);
            auto it2 = gg.ten.find(nm);
            if (it2 == gg.ten.end()) { fprintf(stderr, "缺 %s\n", nm); exit(1); }
            const GTen& T = it2->second;
            u64 K = T.ne[0], N = T.ne[1];
            u64 rowB = T.nbytes / (T.ne[1] * T.ne[2]);
            u64 nEl = N * K;
            buf.resize(nEl);
            deq_any(T.gt, gg.maps[T.fi] + T.off + e * N * rowB, buf.data(), nEl);
            parts[t] = astype(array(buf.data(), Shape{(int)N, (int)K}, float32), float16);
            if (t == 2) parts[t] = transpose(parts[t], {1, 0}); /* down: [5120,2304]→[2304,5120]（out 行） */
        }
        array all = concatenate({parts[0], parts[1], parts[2]}, 0);
        eval(all);
        size_t b = (size_t)all.nbytes();
        exp_bytes += b;
        exp_cache.emplace(key, all);
        exp_lru.push_back({key, b});
        while (exp_bytes > exp_cap && !exp_lru.empty()) {
            auto old = exp_lru.front();
            exp_lru.erase(exp_lru.begin());
            auto it3 = exp_cache.find(old.first);
            if (it3 != exp_cache.end()) {
                exp_bytes -= it3->second.nbytes();
                exp_cache.erase(it3);
            }
        }
        return all;
    }

    /* ---- Engram 行取（mmap IQ4_NL；ids [n_tok*24]） ---- */
    array eng_rows_get(const std::vector<i32>& ids) {
        int nt = (int)ids.size() / (c.hpn * (c.ngram - 1));
        int cols = c.hpn * (c.ngram - 1);
        std::vector<float> out((size_t)ids.size() * c.eng_hd);
        int rowB = c.eng_hd / 32 * 18;
        std::vector<float> row(c.eng_hd);
        for (int i = 0; i < nt; i++)
            for (int col = 0; col < cols; col++) {
                i32 r = ids[(size_t)i * cols + col];
                deq_iq4_nl(eng_map + (u64)r * rowB, row.data(), c.eng_hd);
                memcpy(&out[(size_t)i * cols * c.eng_hd + (size_t)col * c.eng_hd], row.data(), c.eng_hd * 4);
            }
        return array(out.data(), Shape{nt, cols * c.eng_hd}, float32);
    }

    /* ---- Engram 哈希（llama.cpp engram_hash 语义；host） ---- */
    void eng_hash(int e, int pos0, int s, std::vector<i32>& out) {
        int cols = c.hpn * (c.ngram - 1);
        out.assign((size_t)s * cols, -1);
        std::vector<u32> toks(c.ngram);
        for (int i = 0; i < s; i++) {
            int absp = pos0 + i;
            i32 cid = tmap_h[hist[absp]];
            if (cid < 0) cid = c.pad_id;
            toks[0] = (u32)cid;
            bool blocked = false;
            for (int o = 1; o < c.ngram; o++) {
                int pi = absp - o;
                if (pi < 0) {
                    toks[o] = (u32)c.pad_id;
                    blocked = true;
                } else {
                    i32 pv = tmap_h[hist[pi]];
                    if (pv < 0 || blocked) {
                        toks[o] = (u32)c.pad_id;
                        if (pv < 0) blocked = true;
                    } else
                        toks[o] = (u32)pv;
                }
            }
            u64 rolling = toks[0] * c.eng_mult_h[e * (c.ngram - 1) + 0];
            for (int o = 1; o < c.ngram; o++) rolling ^= toks[o] * c.eng_mult_h[e * (c.ngram - 1) + o];
            for (int h = 0; h < cols; h++) {
                int t = e * cols + h;
                u64 r = rolling % (u64)c.eng_prime_h[t] + (u64)c.eng_off_h[t];
                out[(size_t)i * cols + h] = (i32)r;
            }
        }
    }

    /* ---- Engram 前向（build_engram 语义：rms(x)·rms(key)·gate 加权余弦门） ---- */
    array engram(int ei, const array& x, int pos0, int s) {
        std::vector<i32> ids;
        eng_hash(ei, pos0, s, ids);
        auto rows = eng_rows_get(ids);
        std::string p = "blk." + std::to_string(c.eng_layers[ei]) + ".";
        auto kv = matmul(rows, transpose(Wd(p + "engram_kv.weight")));
        auto key = reshape(slice(kv, Shape{0, 0}, Shape{s, c.HC * c.D}), {s, c.HC, c.D});
        auto val = reshape(slice(kv, Shape{0, c.HC * c.D}, Shape{s, (c.HC + 1) * c.D}), {s, 1, c.D});
        auto eg = astype(Wq(p + "engram_gate.weight"), float32);
        auto xh = reshape(x, {s, c.HC, c.D});
        auto rx = f32(xh) * rsqrt(mean(square(f32(xh)), -1, true) + fa(c.NORM_EPS));
        auto rk = f32(key) * rsqrt(mean(square(f32(key)), -1, true) + fa(c.NORM_EPS));
        auto dot = sum(rx * rk * astype(reshape(eg, {1, c.HC, c.D}), float32), -1) * fa(1.f / std::sqrt((float)c.D));
        auto mag = sqrt(maximum(abs(dot), fa(1e-6f)));
        auto sgn = where(astype(dot, int32) >= ia(0), fa(1.f), fa(-1.f));
        auto gate = astype(sigmoid(mag * sgn), x.dtype());
        auto y = f32(xh) + reshape(gate, {s, c.HC, 1}) * broadcast_to(f32(val), {s, c.HC, c.D});
        return astype(reshape(y, x.shape()), x.dtype());
    }

    /* ---- HC 混合（mini sinkhorn_split） ---- */
    void hc_mix(const array& x, const std::string& pfx, int s, array& pre, array& post, array& comb) {
        int hc = c.HC;
        auto xf = reshape(f32(x), {s, hc * c.D});
        auto fn = astype(reshape(Wq(pfx + "_fn.weight"), Shape{(2 + hc) * hc, hc * c.D}), float32); /* [(2+hc)hc, hc*D]，GGUF 里 1 维存储 */
        auto rsq = rsqrt(mean(square(xf), -1, true) + fa(c.NORM_EPS));
        auto mixes = matmul(xf, transpose(fn)) * rsq; /* [s,24] */
        auto sc3 = reshape(f32(Wq(pfx + "_scale.weight")), {3}); /* [3] */
        auto bs = reshape(f32(Wq(pfx + "_base.weight")), {(2 + hc) * hc}); /* [24] */
        auto s0 = reshape(slice(sc3, Shape{0}, Shape{1}), {1, 1});
        auto s1 = reshape(slice(sc3, Shape{1}, Shape{2}), {1, 1});
        auto s2 = reshape(slice(sc3, Shape{2}, Shape{3}), {1, 1});
        auto mp = slice(mixes, Shape{0, 0}, Shape{s, hc});
        auto mq = slice(mixes, Shape{0, hc}, Shape{s, 2 * hc});
        auto mc = reshape(slice(mixes, Shape{0, 2 * hc}, Shape{s, (2 + hc) * hc}), {s, hc, hc});
        pre = sigmoid(mp * s0 + reshape(slice(bs, Shape{0}, Shape{hc}), {1, hc})) + fa(c.HC_EPS);
        if (getenv("DS41_SHAPE")) fprintf(stderr, "[hm1]", (int)pre.shape(0)), fflush(stderr);
        post = fa(2.f) * sigmoid(mq * s1 + reshape(slice(bs, Shape{hc}, Shape{2 * hc}), {1, hc}));
        comb = softmax(mc * s2 + reshape(slice(bs, Shape{2 * hc}, Shape{(2 + hc) * hc}), {hc, hc}), -1) + fa(c.HC_EPS);
        comb = comb / (sum(comb, -2, true) + fa(c.HC_EPS));
        for (int it = 1; it < c.SINKHORN; it++) {
            comb = comb / (sum(comb, -1, true) + fa(c.HC_EPS));
            comb = comb / (sum(comb, -2, true) + fa(c.HC_EPS));
        }
    }

    /* ---- 注意力（稠密快路）：q [1,H,s,DH]，kv [1,n,DH]，mask [1,1,s,n]，sinks 每头 ---- */
    array attn_run(const array& q, const array& kv, const array& sink, const array& mask) {
        int s = q.shape(2), n = kv.shape(1), h = q.shape(1);
        auto lg = matmul(q, astype(transpose(kv, {0, 2, 1}), q.dtype())) * fa(1.f / std::sqrt((float)c.DH));
        auto sk = broadcast_to(reshape(astype(sink, q.dtype()), {1, h, 1, 1}), {1, h, s, 1});
        lg = concatenate({lg, sk}, -1); /* [1,H,s,n+1] */
        auto m = concatenate({mask, ones({1, 1, s, 1}, float32)}, -1);
        if (getenv("DS41_SHAPE"))
            fprintf(stderr, "[shp] lg=%d,%d,%d,%d m=%d,%d,%d,%d\n", (int)lg.shape(0), (int)lg.shape(1),
                    (int)lg.shape(2), (int)lg.shape(3), (int)m.shape(0), (int)m.shape(1), (int)m.shape(2),
                    (int)m.shape(3));
        lg = where(astype(m, bool_), lg, fa(-INFINITY));
        auto p = softmax(f32(lg), -1); /* [1,H,s,n+1] */
        auto pv = slice(p, Shape{0, 0, 0, 0}, Shape{1, h, s, n});
        return matmul(pv, astype(kv, float32)); /* [1,H,s,DH] */
    }

    /* ---- 单层前向 ---- */
    std::pair<array, array> block(int l, const array& x0, array pre_prev, int pos0, int s) {
        std::string p = "blk." + std::to_string(l) + ".";
        int hc = c.HC, h = c.H, dh = c.DH;
        auto x = x0;

        if (!getenv("SGL_NOENG"))
            for (size_t ei = 0; ei < c.eng_layers.size(); ei++)
                if (c.eng_layers[ei] == l) x = engram((int)ei, x, pos0, s);

        array a_pre = zeros({1}, float32), a_post = zeros({1}, float32), a_comb = zeros({1}, float32);
        fprintf(stderr, "[L%d.hcm]", l);
        fflush(stderr);
        hc_mix(x, p + "hc_attn", s, a_pre, a_post, a_comb);
        fprintf(stderr, "[L%d.hcm_ok]", l);
        fflush(stderr);
        auto residual = x;
        auto pm = reshape(f32(pre_prev), {s, hc, 1});
        auto y = reshape(sum(reshape(f32(x), {s, hc, c.D}) * pm, 1), {s, c.D});
        y = rms(y, Wq(p + "attn_norm.weight"), c.NORM_EPS);
        if (getenv("DS41_TRACE")) {
            auto flat = abs(reshape(f32(y), {-1}));
            auto red = max(flat, -1);
            red.eval();
            fprintf(stderr, "[tr] L%d y maxabs=%.4g\n", l, red.data<float>()[0]);
        }

        /* q 路径 */
        auto qa = lin(y, p + "attn_q_a.weight");
        qa = rms(qa, Wq(p + "attn_q_a_norm.weight"), c.NORM_EPS);
        auto q = astype(reshape(lin(qa, p + "attn_q_b.weight"), {s, h, dh}), float32);        bool use_yarn = c.ratios[l] != 0;
        const array& wcos = use_yarn ? cmp_cos_d : win_plain_d;
        const array& wsin = use_yarn ? cmp_sin_d : win_plain_sin_d;
        q = rope_tail(q, freq_slice(wcos, pos0, s), freq_slice(wsin, pos0, s), false);
        fprintf(stderr, "[L%d.q]", l);
        fflush(stderr);
        auto qh4 = reshape(transpose(q, {1, 0, 2}), {1, h, s, dh});
        /* 窗口 kv */
        auto kvn = lin(y, p + "attn_kv.weight");
        if (getenv("DS41_TRACE")) {
            auto flat = abs(reshape(f32(kvn), {-1}));
            auto red = max(flat, -1);
            red.eval();
            fprintf(stderr, "[tr] L%d kvn_lin maxabs=%.4g\n", l, red.data<float>()[0]);
        }
        kvn = rms(kvn, Wq(p + "attn_kv_a_norm.weight"), c.NORM_EPS);
        kvn = reshape(astype(kvn, float32), {s, dh});
        kvn = astype(rope_tail(reshape(kvn, {s, 1, dh}), freq_slice(wcos, pos0, s), freq_slice(wsin, pos0, s),
                               false),
                     float32);
        auto kv2 = reshape(kvn, {s, dh});
        if (getenv("DS41_TRACE")) {
            auto flat = abs(reshape(f32(kv2), {-1}));
            auto red = max(flat, -1);
            red.eval();
            fprintf(stderr, "[tr] L%d kv2 maxabs=%.4g\n", l, red.data<float>()[0]);
        }
        if (pos0 == 0) {
            if (s <= WIN) {
                wkv[l] = slice_update(wkv[l], kv2, Shape{0, 0, 0}, Shape{MAXB, s, dh});
            } else {
                /* 官方 ring 种子：rows[cut..WIN) ← tokens[s-WIN, s-cut)；rows[0..cut) ← tokens[s-cut, s)
                   （行 j = 最新 pos≡j mod WIN，与 decode 写入一致） */
                int cut = s % WIN;
                wkv[l] = slice_update(wkv[l], slice(kv2, Shape{0, s - WIN}, Shape{s - cut, dh}), Shape{0, cut, 0},
                                      Shape{MAXB, WIN, dh});
                wkv[l] = slice_update(wkv[l], slice(kv2, Shape{0, s - cut}, Shape{s, dh}), Shape{0, 0, 0},
                                      Shape{MAXB, cut, dh});
            }
        } else {
            int row = pos0 % WIN;
            wkv[l] = slice_update(wkv[l], kv2, Shape{0, row, 0}, Shape{MAXB, row + 1, dh});
        }

        /* 压缩源：本层是源则写新行；crows = 本层消费的压缩行 */
        int n_comp = 0;
        array crows = zeros({1, 0, dh}, float32); /* 无压缩层时 0 行 */
        int ci = comp_idx[l];
        if (ci >= 0) {
            Comp& cp = comps[ci];
            int r = cp.ratio;
            if (cp.src == l) {
                auto lat = lin(y, p + "attn_compressor_kv.weight"); /* [s,512] */
                if (r > 1) {
                    auto g2 = lin(y, p + "attn_compressor_gate.weight");
                    auto lat3 = reshape(f32(lat), {s, 1, dh}), g23 = reshape(f32(g2), {s, 1, dh});
                    for (int i = 0; i < s; i++) {
                        int absp = pos0 + i;
                        int slot = absp % r;
                        cp.kv_state =
                            slice_update(cp.kv_state, slice(lat3, Shape{i, 0, 0}, Shape{i + 1, 1, dh}),
                                         Shape{0, slot, 0}, Shape{MAXB, slot + 1, dh});
                        cp.score_state =
                            slice_update(cp.score_state, slice(g23, Shape{i, 0, 0}, Shape{i + 1, 1, dh}),
                                         Shape{0, slot, 0}, Shape{MAXB, slot + 1, dh});
                        if ((absp + 1) % r == 0) {
                            auto sm = softmax(cp.score_state, 1);
                            auto rowv = sum(cp.kv_state * sm, 1, true); /* [MAXB,1,dh] */
                            rowv = rms(rowv, Wq(p + "attn_compressor_norm.weight"), c.NORM_EPS);
                            cp.cache = slice_update(cp.cache, rowv, Shape{0, absp / r, 0},
                                                    Shape{MAXB, absp / r + 1, dh});
                        }
                    }
                    n_comp = (pos0 + s) / r;
                } else {
                    /* ratio=1：每 token 一行，norm 后按真实位置 rope（官方 Compressor ratio==1 分支） */
                    auto latn = rms(f32(lat), Wq(p + "attn_compressor_norm.weight"), c.NORM_EPS);
                    auto lr = rope_tail(reshape(latn, {s, 1, dh}), freq_slice(cmp_cos_d, pos0, s),
                                        freq_slice(cmp_sin_d, pos0, s), false);
                    cp.cache = slice_update(cp.cache, reshape(lr, {1, s, dh}), Shape{0, pos0, 0},
                                            Shape{MAXB, pos0 + s, dh});
                    n_comp = pos0 + s;
                }
            } else {
                n_comp = (pos0 + s) / r;
            }
            if (n_comp > 0) {
                if (getenv("DS41_SHAPE"))
                    fprintf(stderr, "[cr] n_comp=%d cache=%d,%d,%d\n", n_comp, (int)cp.cache.shape(0),
                            (int)cp.cache.shape(1), (int)cp.cache.shape(2));
                crows = slice(cp.cache, Shape{0, 0, 0}, Shape{MAXB, n_comp, dh});
                if (r > 1) { /* ratio>1：组 j 的 rope 位置 = j*r（官方 freqs_cis[:s-s%r:r]） */
                    std::vector<int> st(n_comp);
                    for (int j = 0; j < n_comp; j++) st[j] = j * r;
                    auto cid = array(st.data(), Shape{n_comp}, int32);
                    crows = astype(rope_tail(reshape(crows, {n_comp, 1, dh}), take(cmp_cos_d, cid, 0),
                                             take(cmp_sin_d, cid, 0), false),
                                   float32);
                }
            }
        }

        if (getenv("DS41_SHAPE"))
            fprintf(stderr, "[kall] wkv=%d,%d,%d crows=%d,%d,%d\n", (int)wkv[l].shape(0), (int)wkv[l].shape(1),
                    (int)wkv[l].shape(2), (int)crows.shape(0), (int)crows.shape(1), (int)crows.shape(2));
        /* prefill：窗口 kv 用整段 chunk（官方 window_kv=kv）；decode：ring（行 j=最新 pos≡j mod WIN） */
        array kwin = zeros({1, 1, 1}, float32);
        int nw;
        if (pos0 == 0) {
            kwin = reshape(kv2, {1, s, dh});
            nw = s;
        } else {
            kwin = wkv[l];
            nw = WIN;
        }
        crows = reshape(crows, {1, n_comp, dh});
        auto kall = concatenate({kwin, crows}, 1); /* [1, nw+n_comp, DH] */
        int n = nw + n_comp;
        std::vector<float> mh((size_t)s * n, -INFINITY);
        int L = pos0 + s - 1;
        for (int i = 0; i < s; i++) {
            int absp = pos0 + i;
            for (int j = 0; j < nw; j++) {
                int ap = (pos0 == 0) ? j : (L >= WIN ? L - (L - j) % WIN : j);
                if (ap <= absp && ap >= absp - WIN + 1 && ap >= 0) mh[(size_t)i * n + j] = 0.f;
            }
            for (int j = 0; j < n_comp; j++) {
                int gend = (j + 1) * (ci >= 0 ? comps[ci].ratio : 1) - 1;
                if (gend <= absp) mh[(size_t)i * n + nw + j] = 0.f;
            }
        }
        auto mask = array(mh.data(), Shape{1, 1, s, n}, float32);
        fprintf(stderr, "[L%d.a]", l);
        fflush(stderr);
        auto sink = Wq(p + "attn_sinks.weight");
        auto o = attn_run(qh4, kall, sink, mask); /* [1,H,s,DH] */
        eval(o);
        if (getenv("DS41_TRACE")) {
            auto flat = abs(reshape(f32(o), {-1}));
            auto red = max(flat, -1);
            red.eval();
            fprintf(stderr, "[tr] L%d o maxabs=%.4g\n", l, red.data<float>()[0]);
        }
        fprintf(stderr, "[L%d.attn_ok]", l);
        fflush(stderr);
        o = reshape(transpose(reshape(o, {h, s, dh}), {1, 0, 2}), {s, h, dh});
        o = rope_tail(o, freq_slice(wcos, pos0, s), freq_slice(wsin, pos0, s), true);

        /* 输出分组投影：o [s,H,DH] → [s,G,hg*dh] → wo_a[G,ORANK,hg*dh] → [s,G*ORANK] → wo_b */
        int hg = h / c.G;
        auto og4 = reshape(o, {s, c.G, hg * dh});
        auto o_a = reshape(Wd(p + "attn_output_a.weight"), {c.G, c.ORANK, hg * dh});
        auto og = matmul(transpose(og4, {1, 0, 2}), transpose(o_a, {0, 2, 1})); /* [G,s,ORANK] */
        og = reshape(transpose(og, {1, 0, 2}), {s, c.G * c.ORANK});
        auto attn_out = lin(og, p + "attn_output_b.weight");
        if (getenv("DS41_TRACE")) {
            auto flat = abs(reshape(f32(attn_out), {-1}));
            auto red = max(flat, -1);
            red.eval();
            fprintf(stderr, "[tr] L%d attn_out maxabs=%.4g\n", l, red.data<float>()[0]);
            fflush(stderr);
        }

        /* hc_post(attn)：x = a_post·attn_out + Σ_comb·residual（mini hc_post） */
        auto ao = reshape(f32(attn_out), {s, 1, c.D});
        auto po = reshape(f32(a_post), {s, hc, 1});
        auto co = reshape(f32(a_comb), {s, hc, hc, 1});
        auto xr = reshape(f32(residual), {s, hc, 1, c.D});
        x = astype(po * ao + sum(co * xr, 2), x0.dtype());

        /* ffn */
        array f_pre = zeros({1}, float32), f_post = zeros({1}, float32), f_comb = zeros({1}, float32);
        hc_mix(x, p + "hc_ffn", s, f_pre, f_post, f_comb);
        if (getenv("DS41_HCDBG"))
            fprintf(stderr, "[hc%d] f_pre=%.4g f_post=%.4g f_comb=%.4g x_mid=%.4g\n", l, mx(f_pre), mx(f_post),
                    mx(f_comb), mx(x));
        residual = x;
        auto pm2 = reshape(f32(f_pre), {s, hc, 1});
        auto y2 = reshape(sum(reshape(f32(x), {s, hc, c.D}) * pm2, 1), {s, c.D});
        y2 = rms(y2, Wq(p + "ffn_norm.weight"), c.NORM_EPS);
        auto moe_out = moe(l, y2);
        if (getenv("DS41_TRACE")) {
            auto flat = abs(reshape(f32(moe_out), {-1}));
            auto red = max(flat, -1);
            red.eval();
            fprintf(stderr, "[tr] L%d moe maxabs=%.4g\n", l, red.data<float>()[0]);
            fflush(stderr);
        }
        fprintf(stderr, "[L%d.moe_ok]", l);
        fflush(stderr);
        auto fo = reshape(f32(moe_out), {s, 1, c.D});
        auto pf = reshape(f32(f_post), {s, hc, 1});
        auto cf = reshape(f32(f_comb), {s, hc, hc, 1});
        auto xr2 = reshape(f32(residual), {s, hc, 1, c.D});
        x = astype(pf * fo + sum(cf * xr2, 2), x0.dtype());
        return {x, f_pre};
    }

    static double mx(const array& a) {
        auto r = max(abs(reshape(f32(a), {-1})), -1);
        r.eval();
        return r.data<float>()[0];
    }
    /* ---- MoE：sqrtsoftplus 路由 + mmap 专家（LRU）+ 共享专家 ---- */
    array moe(int l, const array& y2) {
        std::string p = "blk." + std::to_string(l) + ".";
        int s = y2.shape(0);
        int dbg = getenv("DS41_MOEDBG") ? atoi(getenv("DS41_MOEDBG")) : -1;
        auto out = zeros({s, c.D}, float32);
        if (getenv("SGL_NOEXPERTS")) { /* 排除法：只算共享专家 */ }
        auto lg = matmul(f32(y2), transpose(astype(Wq(p + "ffn_gate_inp.weight"), float32)));
        if (getenv("DS41_SHAPE")) fprintf(stderr, "[m1]", (int)lg.shape(1)), fflush(stderr);
        auto bias = reshape(astype(Wq(p + "exp_probs_b.bias"), float32), {1, c.NE});
        auto biased = lg + bias;
        auto spv = maximum(lg, fa(0.f)) + log(fa(1.f) + exp(-abs(lg))); /* 稳定 softplus */
        auto sc = sqrt(spv); /* sqrtsoftplus */
        auto order = argsort(biased, -1);
        auto sel = slice(order, Shape{0, c.NE - c.TOPK}, Shape{s, c.NE});
        auto flat = reshape(astype(sel, int32), {-1}) + reshape(repeat(arange(0, s, 1, int32), c.TOPK, 0) * ia(c.NE), {-1});
        auto ws = reshape(take(reshape(sc, {-1}), flat, 0), {s, c.TOPK});
        ws = ws / (sum(ws, -1, true) + fa(1e-20f)) * fa(c.ROUTE_SCALE);
        eval(sel);
        eval(ws);
        const i32* sp = sel.data<i32>();
        const float* wp = ws.data<float>();
        std::vector<int> cnt(c.NE, 0);
        for (int i = 0; i < s * c.TOPK; i++) cnt[sp[i]]++;
        if (dbg == l) {
            fprintf(stderr, "[moe%d] lg=%.4g ws0=", l, mx(lg));
            auto w0 = reshape(ws, {-1}); w0.eval();
            for (int i = 0; i < c.TOPK; i++) fprintf(stderr, "%.4g(sp%d) ", w0.data<float>()[i], sp[i]);
            fprintf(stderr, "\n");
            auto W3 = expert_get(l, sp[0]);
            fprintf(stderr, "[moe%d] W3 gate=%.4g up=%.4g down=%.4g\n", l, mx(slice(W3, Shape{0, 0}, Shape{c.FFN, c.D})),
                    mx(slice(W3, Shape{c.FFN, 0}, Shape{2 * c.FFN, c.D})),
                    mx(slice(W3, Shape{2 * c.FFN, 0}, Shape{3 * c.FFN, c.D})));
        }
        for (int e = 0; e < c.NE; e++) {
            if (!cnt[e]) continue;
            std::vector<u32> rows;
            std::vector<float> wgt;
            for (int i = 0; i < s; i++)
                for (int t = 0; t < c.TOPK; t++)
                    if (sp[(size_t)i * c.TOPK + t] == e) {
                        rows.push_back(i);
                        wgt.push_back(wp[(size_t)i * c.TOPK + t]);
                    }
            auto ridx = array(rows.data(), Shape{(int)rows.size()}, int32);
            auto xe = take(f32(y2), ridx, 0);
            auto W3 = expert_get(l, e); /* [3N, D] f16：gate/up/down（down 已转置存 [in=FFN, out=D]） */
            auto g = matmul(xe, transpose(f32(slice(W3, Shape{0, 0}, Shape{c.FFN, c.D}))));
            auto u = matmul(xe, transpose(f32(slice(W3, Shape{c.FFN, 0}, Shape{2 * c.FFN, c.D}))));
            auto dW = f32(slice(W3, Shape{2 * c.FFN, 0}, Shape{3 * c.FFN, c.D}));
            /* 官方 Expert：up ±L，gate ≤L，act=silu(g)*up，权重在 w2 之前 */
            g = minimum(g, fa(c.SWIGLU_L));
            u = maximum(minimum(u, fa(c.SWIGLU_L)), fa(-c.SWIGLU_L));
            auto act = (g * sigmoid(g)) * u;
            auto wcol = array(wgt.data(), Shape{(int)rows.size(), 1}, float32);
            /* 官方 w2 前乘路由权重：ye = (silu(g)·u · w) @ down_stored，
             * down_stored 已是 [in=FFN, out=D]，不能再转置（旧代码把 dn 投到了
             * xe、ye 算成了 [rows,rows]，形状碰巧能拼出结果但数学全错 —— 2026-11-01 修） */
            auto ye = matmul(act * wcol, dW); /* [rows,FFN]@[FFN,D] = [rows,D] */
            if (dbg == l && e == sp[0])
                fprintf(stderr, "[moe%d] e%d g=%.4g u=%.4g act=%.4g ye=%.4g\n", l, e, mx(g), mx(u), mx(act), mx(ye));
            std::vector<float> onev((size_t)s * rows.size(), 0.f);
            for (size_t ri = 0; ri < rows.size(); ri++) onev[(size_t)rows[ri] * rows.size() + ri] = 1.f;
            auto oh = array(onev.data(), Shape{s, (int)rows.size()}, float32);
            out = out + matmul(oh, ye);
        }
        auto sh_g = lin(f32(y2), p + "ffn_gate_shexp.weight");
        if (dbg == l) {
            /* A/B：quantized_matmul 快路 vs dequantize+matmul 慢路，并把 y2 落盘给 python 裁决 */
            auto wname = p + "ffn_gate_shexp.weight";
            auto slow = matmul(f32(y2), transpose(Wd(wname)));
            fprintf(stderr, "[moe%d] shg fast=%.6g slow=%.6g\n", l, mx(sh_g), mx(slow));
            auto y2f = reshape(f32(y2), {-1});
            y2f.eval();
            FILE* fo = fopen("/tmp/y2_dump.bin", "wb");
            if (fo) {
                fwrite(y2f.data<float>(), 4, y2f.size(), fo);
                fclose(fo);
                fprintf(stderr, "[moe%d] y2 dumped %zu f32\n", l, y2f.size());
            }
            auto lgf = reshape(f32(lg), {-1});
            lgf.eval();
            fo = fopen("/tmp/lg_dump.bin", "wb");
            if (fo) {
                fwrite(lgf.data<float>(), 4, lgf.size(), fo);
                fclose(fo);
            }
        }
        auto sh_u = lin(f32(y2), p + "ffn_up_shexp.weight");
        sh_g = minimum(sh_g, fa(c.SWIGLU_L));
        sh_u = maximum(minimum(sh_u, fa(c.SWIGLU_L)), fa(-c.SWIGLU_L));
        auto sh = lin((sh_g * sigmoid(sh_g)) * sh_u, p + "ffn_down_shexp.weight");
        if (dbg == l) fprintf(stderr, "[moe%d] routed=%.4g sh_g=%.4g sh_u=%.4g sh=%.4g\n", l, mx(out), mx(sh_g), mx(sh_u), mx(sh));
        return out + f32(sh);
    }

    /* ---- 全模型前向：返回最后位置 logits [V] ---- */
    array forward(int pos0, int s) {
        int hc = c.HC;
        auto ids = array(hist.data() + pos0, Shape{1, s}, int32);
        auto w_emb = Wq("token_embd.weight");
        auto flat_ids = reshape(ids, {s});
        auto e_wq = take(w_emb, flat_ids, 0);
        auto e_s = take(Wq("token_embd.weight.scales"), flat_ids, 0);
        auto e_b = take(Wq("token_embd.weight.biases"), flat_ids, 0);
        auto h = dequantize(e_wq, e_s, e_b, 64, 4); /* [s, D] f32 */
        h = repeat(reshape(h, {s, 1, c.D}), hc, 1); /* [s, HC, D]（每拷贝同一嵌入，mini 语义） */
        if (getenv("DS41_TRACE")) {
            auto flat = abs(reshape(f32(h), {-1}));
            auto red = max(flat, -1);
            red.eval();
            fprintf(stderr, "[tr] embed maxabs=%.4g\n", red.data<float>()[0]);
        }
        array pre_prev = zeros({s, hc}, float32);
        {
            std::vector<float> one((size_t)s * hc, 0.f);
            for (int i = 0; i < s; i++) one[(size_t)i * hc] = 1.f;
            pre_prev = array(one.data(), Shape{s, hc}, float32);
        }
        int maxl = getenv("DS41_MAXL") ? atoi(getenv("DS41_MAXL")) : 10000;
        for (int l = 0; l < c.L; l++) {
            if (l >= maxl) break;
            fprintf(stderr, "[L%d\n", l);
            fflush(stderr);
            auto r = block(l, h, pre_prev, pos0, s);
            fprintf(stderr, "L%d]\n", l);
            fflush(stderr);
            h = r.first;
            pre_prev = r.second;
            if (getenv("DS41_TRACE")) {
                auto flat = abs(reshape(f32(h), {-1}));
                auto red = max(flat, -1);
                red.eval();
                fprintf(stderr, "[tr] L%d h maxabs=%.4g\n", l, red.data<float>()[0]);
                fflush(stderr);
            }
        }
        auto pm = reshape(f32(pre_prev), {s, hc, 1});
        auto hcc = reshape(sum(f32(h) * pm, 1), {s, c.D});
        hcc = rms(hcc, Wq("output_norm.weight"), c.NORM_EPS);
        auto lg = quantized_matmul(hcc, Wq("output.weight"), Wq("output.weight.scales"),
                                   Wq("output.weight.biases"), true, 64, 4);
        return reshape(slice(lg, Shape{s - 1, 0}, Shape{s, c.V}), {c.V}); /* 最后位置 */
    }

    void dump_top(const array& lg, const char* tag) {
        auto t = sort(lg, -1);
        auto idx = argsort(lg, -1);
        t.eval();
        idx.eval();
        const float* v = t.data<float>();
        const i32* ix = idx.data<i32>();
        int n = (int)lg.size();
        fprintf(stderr, "[%s] top:", tag);
        for (int i = 0; i < 5; i++) fprintf(stderr, " %d:%.4f", ix[n - 1 - i], v[n - 1 - i]);
        fprintf(stderr, "\n");
    }

    std::vector<u32> run(const std::vector<u32>& prompt, int n_gen) {
        reset();
        hist = prompt;
        std::vector<u32> gen;
        auto lg = forward(0, (int)prompt.size());
        if (getenv("DS41_TRACE")) dump_top(lg, "prefill");
        fprintf(stderr, "[run] prefill ok\n");
        fflush(stderr);
        auto nx = astype(argmax(lg, -1), int32);
        eval(nx);
        u32 t = (u32)item_i(nx, 0);
        gen.push_back(t);
        hist.push_back(t);
        pos = (int)prompt.size();
        for (int i = 1; i < n_gen; i++) {
            fprintf(stderr, "[run] step %d\n", i);
            fflush(stderr);
            auto lg2 = forward(pos, 1);
            auto nx2 = astype(argmax(lg2, -1), int32);
            if (getenv("DS41_TRACE")) dump_top(lg2, "step");
            eval(nx2);
            t = (u32)item_i(nx2, 0);
            gen.push_back(t);
            hist.push_back(t);
            pos += 1;
        }
        return gen;
    }
};

} // namespace fs41

/* 冒烟入口：./mlx_ds41_model <dense目录> [ids 以逗号分隔] [步数] */
int main(int argc, char** argv) {
    setbuf(stdout, nullptr);
    if (argc < 2) {
        fprintf(stderr, "usage: %s <mlx-dense dir> [ids like 9707,11] [steps]\n", argv[0]);
        return 2;
    }
    fs41::Model m;
    if (getenv("DS41_WCHECK")) m.wcheck();
    if (!m.load(argv[1])) return 1;
    std::vector<unsigned> prompt;
    if (argc > 2 && argv[2][0]) {
        std::string s = argv[2];
        size_t p = 0;
        while (p < s.size()) {
            size_t e = s.find(',', p);
            if (e == std::string::npos) e = s.size();
            prompt.push_back((unsigned)std::stoul(s.substr(p, e - p)));
            p = e + 1;
        }
    }
    if (prompt.empty()) prompt = {9707, 11};
    int n = argc > 3 ? std::stoi(argv[3]) : 16;
    printf("[ds41] prompt %zu tokens, gen %d steps\n", prompt.size(), n);
    auto t0 = std::chrono::steady_clock::now();
    std::vector<unsigned> gen;
    try {
        gen = m.run(prompt, n);
    } catch (std::exception& e) {
        fprintf(stderr, "EXC: %s\n", e.what());
        return 1;
    } catch (...) {
        fprintf(stderr, "EXC: unknown\n");
        return 1;
    }
    double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("[ds41] gen ids:");
    for (auto g : gen) printf(" %u", g);
    printf("\n[ds41] %.2fs total, %.3f s/step\n", el, el / (n + 1));
    return 0;
}
