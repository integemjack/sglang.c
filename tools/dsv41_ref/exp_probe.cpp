// d41_exp_probe.cpp — 用与 mlx_ds41_model.cpp 完全相同的代码路径反量化专家，查量级与有限性
// 构建: c++ -O2 -std=c++17 d41_exp_probe.cpp -o d41_exp_probe
#include "ds41_iq_tables.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <unordered_map>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

typedef uint8_t u8; typedef uint16_t u16; typedef uint32_t u32; typedef uint64_t u64;
typedef int8_t i8; typedef int64_t i64;

struct GTen { int fi = 0; u64 off = 0, raw = 0; u32 gt = 0; std::vector<u64> ne; u64 nbytes = 0; };
struct GG {
    std::vector<int> fds; std::vector<u8*> maps;
    std::unordered_map<std::string, GTen> ten;
    std::unordered_map<std::string, std::vector<GTen>> dup;
    static u64 rd64(const u8* p) { u64 v; memcpy(&v, p, 8); return v; }
    static u32 rd32(const u8* p) { u32 v; memcpy(&v, p, 4); return v; }
    static std::string rstr(const u8*& p) { u64 n = rd64(p); p += 8; std::string s((const char*)p, n); p += n; return s; }
    static u64 nblk_bytes(u32 gt, u64 nEl) {
        switch (gt) {
        case 16: return nEl / 256 * 66; case 17: return nEl / 256 * 74; case 29: return nEl / 256 * 56;
        case 13: return nEl / 256 * 176; case 0: return nEl * 4; case 1: case 30: return nEl * 2;
        case 8: return nEl / 32 * 34; case 12: return nEl / 256 * 144; case 14: return nEl / 256 * 210;
        case 19: return nEl / 256 * 50; case 20: return nEl / 32 * 18; case 23: return nEl / 256 * 136;
        case 22: return nEl / 256 * 122; case 24: return nEl;
        default: fprintf(stderr, "gtype %u\n", gt); exit(1);
        }
    }
    void open(const std::vector<std::string>& paths) {
        for (int fi = 0; fi < (int)paths.size(); fi++) {
            int fd = ::open(paths[fi].c_str(), O_RDONLY);
            if (fd < 0) { fprintf(stderr, "open fail %s\n", paths[fi].c_str()); exit(1); }
            struct stat st; fstat(fd, &st);
            u8* m = (u8*)mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
            madvise(m, st.st_size, MADV_RANDOM);
            maps.push_back(m);
            const u8* p = m; p += 8;
            u64 n_ten = rd64(p); p += 8;
            u64 n_kv = rd64(p); p += 8;
            for (u64 k = 0; k < n_kv; k++) { rstr(p); u32 t = rd32(p); p += 4; p = skip(t, p); }
            u64 hb = ((u64)(p - m) + 31) / 32 * 32;
            for (u64 k = 0; k < n_ten; k++) {
                std::string name = rstr(p);
                u32 nd = rd32(p); p += 4;
                GTen t; t.fi = fi; t.ne.resize(nd);
                for (u32 d = 0; d < nd; d++) t.ne[d] = rd64(p), p += 8;
                t.gt = rd32(p); p += 4;
                t.raw = rd64(p);
                t.off = hb + t.raw; p += 8;
                u64 nEl = 1; for (auto d : t.ne) nEl *= d;
                t.nbytes = nblk_bytes(t.gt, nEl);
                ten[name] = t;
            dup[name].push_back(t);
            }
        }
    }
    static const u8* skip(u32 t, const u8* p) {
        if (t == 8) { u64 n = rd64(p); return p + 8 + n; }
        if (t == 9) { u32 et = rd32(p); p += 4; u64 n = rd64(p); p += 8;
            if (et == 8) for (u64 i = 0; i < n; i++) { u64 m2 = rd64(p); p += 8 + m2; }
            else p += n * ((et == 2 || et == 3) ? 2 : (et == 7) ? 1 : (et >= 10 && et <= 12) ? 8 : 4);
            return p; }
        return p + ((t == 2 || t == 3) ? 2 : (t == 7) ? 1 : (t >= 10 && t <= 12) ? 8 : 4);
    }
};

static float f16_to_f32(u16 h) {
    u32 s = (u32)(h & 0x8000) << 16, e = (h >> 10) & 31, m = h & 1023;
    u32 f = e ? (s | ((e - 15 + 127) << 23) | (m << 13)) : (m ? (s | (127 - 15) << 23) : s);
    if (e == 31) f = s | 0x7f800000;
    float out; memcpy(&out, &f, 4); return out;
}
static void deq_iq2_xxs(const u8* blk, float* y, i64 n) {
    const i64 nb = n / 256;
    for (i64 i = 0; i < nb; i++) {
        const u8* p = blk + i * 66;
        float d = f16_to_f32(*(const u16*)p);
        const u8* qs = p + 2;
        for (int ib32 = 0; ib32 < 8; ib32++) {
            u32 a32[2]; memcpy(a32, qs + 4 * ib32, 8);
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
static void deq_iq1_s(const u8* blk, float* y, i64 n) {
    const i64 nb = n / 256;
    for (i64 i = 0; i < nb; i++) {
        const u8* p = blk + i * 50;
        float d = f16_to_f32(*(const u16*)p);
        const u8* qs = p + 2;
        const u16* qh = (const u16*)(p + 2 + 32);
        for (int ib = 0; ib < 8; ib++) {
            float dl = d * (2.f * (float)((qh[ib] >> 12) & 7) + 1.f);
            float delta = (qh[ib] & 0x8000) ? -0.125f : 0.125f;
            for (int l = 0; l < 4; l++) {
                const i8* grid = (const i8*)(ds41_iq::iq1s_grid + (qs[l] | (((qh[ib] >> 3 * l) & 7) << 8)));
                for (int j = 0; j < 8; j++) *y++ = dl * (grid[j] + delta);
            }
            qs += 4;
        }
    }
}
static void deq_any(u32 gt, const u8* src, float* y, i64 n) {
    if (gt == 29) deq_iq1_m(src, y, n);
    else if (gt == 16) deq_iq2_xxs(src, y, n);
    else if (gt == 19) deq_iq1_s(src, y, n);
    else { fprintf(stderr, "probe: gt %u unsupported\n", gt); exit(1); }
}
int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : ".";
    /* 同名张量跨分片重复情况会在 open 后打印 */
    std::vector<std::string> shards;
    for (int i = 1; i <= 8; i++) {
        char p[1024];
        snprintf(p, sizeof p, "%s/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-%05d-of-00006.gguf", dir.c_str(), i);
        int fd = ::open(p, O_RDONLY); if (fd < 0) break; ::close(fd); shards.push_back(p);
    }
    GG gg; gg.open(shards);
    /* 用两个候选基址解码对比：数据区起点 = kv 段末对齐（Strata 风格）vs 头全部对齐（标准） */
    {
        const char* names[2] = {"blk.0.ffn_gate_exps.weight", "blk.0.ffn_down_exps.weight"};
        for (int ni = 0; ni < 2; ni++) {
            auto it = gg.ten.find(names[ni]);
            if (it == gg.ten.end()) continue;
            GTen T = it->second;
            i64 N = T.ne[1], K = T.ne[0];
            i64 rowB = T.nbytes / (N * T.ne[2]);
            static const u64 BASES[2] = {128, 20544};
            for (int bi = 0; bi < 2; bi++) {
                long bad = 0, tot = 0;
                for (int e = 0; e < 6 && e < (int)T.ne[2]; e++) {
                    const u8* src = gg.maps[T.fi] + BASES[bi] + T.raw + (u64)e * N * rowB;
                    for (int r = 0; r < 40; r++) {
                        float y[256];
                        deq_any(T.gt, src + (u64)r * K / 256 * (T.gt == 29 ? 56 : T.gt == 19 ? 50 : 66), y, 256);
                        for (int j = 0; j < 256; j++) { tot++; if (!(y[j] == y[j] && y[j] < 1e30f && y[j] > -1e30f)) bad++; }
                    }
                }
                printf("%s base=%llu: bad=%ld/%ld (%.2f%%)\n", names[ni], (unsigned long long)BASES[bi], bad, tot, 100.0 * bad / tot);
            }
        }
    }
    int ndup = 0;
    for (auto& kv : gg.dup)
        if (kv.second.size() > 1) {
            printf("dup tensor %s: %zu copies", kv.first.c_str(), kv.second.size());
            for (auto& t : kv.second) printf(" fi=%d off=%llu", t.fi, (unsigned long long)t.off);
            printf("\n");
            ndup++;
            if (ndup >= 6) break;
        }
    printf("dup tensors >= %d\n", ndup);
    printf("shards=%zu tensors=%zu\n", shards.size(), gg.ten.size());
    const char* names[3] = {"ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"};
    for (int l : {0, 1, 20}) {
        for (int e : {0, 1, 100, 383}) {
            for (int t = 0; t < 3; t++) {
                char nm[96]; snprintf(nm, sizeof nm, "blk.%d.%s.weight", l, names[t]);
                auto it = gg.ten.find(nm);
                if (it == gg.ten.end()) { printf("missing %s\n", nm); return 1; }
                const GTen& T = it->second;
                u64 K = T.ne[0], N = T.ne[1], E = T.ne[2];
                u64 rowB = T.nbytes / (T.ne[1] * T.ne[2]);
                u64 nEl = N * K;
                std::vector<float> buf(nEl);
                deq_any(T.gt, gg.maps[T.fi] + T.off + e * N * rowB, buf.data(), nEl);
                if (l == 0 && e == 0 && t == 0) {
                    printf("first block raw: ");
                    for (int i = 0; i < 56; i++) printf("%02x", gg.maps[T.fi][T.off + e * N * rowB + i]);
                    printf("\nfirst 12 vals: ");
                    for (int i = 0; i < 12; i++) printf("%.5f ", buf[i]);
                    printf("\n");
                }
                double mx = 0, sum = 0; long bad = 0;
                for (u64 i = 0; i < nEl; i++) {
                    float v = buf[i];
                    if (!(v == v && v < 1e30 && v > -1e30)) bad++;
                    else { double a = fabs((double)v); if (a > mx) mx = a; sum += v; }
                }
                if (l == 0 && e == 0 && t == 0) {
                    /* 坏块分布：前 40 行每行的非有限计数 + 前 3 个坏块的行号/块号/原始字节 */
                    printf("fi=%d off=%llu bad-distribution rows 0..39:", T.fi, (unsigned long long)T.off);
                    for (int r = 0; r < 40; r++) {
                        long rb = 0;
                        for (int b = 0; b < 20; b++)
                            for (int j = 0; j < 256; j++)
                                if (!(buf[(u64)r * 5120 + (u64)b * 256 + j] == buf[(u64)r * 5120 + (u64)b * 256 + j] && buf[(u64)r * 5120 + (u64)b * 256 + j] < 1e30)) rb++;
                        printf(" %ld", rb);
                    }
                    printf("\n");
                    int shown = 0;
                    for (u64 b = 0; b < nEl / 256 && shown < 3; b++) {
                        bool isbad = false;
                        for (int j = 0; j < 256; j++) { float v = buf[b * 256 + j]; if (!(v == v && v < 1e30)) { isbad = true; break; } }
                        if (isbad) {
                            printf("bad block %llu (row %llu blk %llu) raw:", (unsigned long long)b, (unsigned long long)(b / 20), (unsigned long long)(b % 20));
                            const u8* bp = gg.maps[T.fi] + T.off + e * N * rowB + b * 56;
                            for (int i2 = 0; i2 < 56; i2++) printf("%02x", bp[i2]);
                            printf("\n");
                            shown++;
                        }
                    }
                }
                printf("L%-2d e%-3d %-14s gt=%u ne=%llu,%llu,%llu rowB=%llu | bad=%ld maxabs=%.5g mean=%.6g\n",
                       l, e, names[t], T.gt, (unsigned long long)K, (unsigned long long)N, (unsigned long long)E,
                       (unsigned long long)rowB, bad, mx, sum / (double)nEl);
            }
        }
    }
    return 0;
}
