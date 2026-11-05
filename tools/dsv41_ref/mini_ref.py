"""mini_ref.py — DeepSeek-V4.1 纯 torch CPU 镜像（dsv41-mini 金标准生成器）。

逐行转录 tools/dsv41_ref/{model,kernel,engram}.py 的语义，差异只有三点（都有注释）：
1. 无 tilelang：量化与 sparse_attn / hc_split_sinkhorn 用等价 torch 数学重写
2. 全程 fp32 激活（原版 bf16 流）——C++ 侧以 fp32 对拍，目标 ~1e-5
3. tokenizer=None：token_map 用随机表代替（真实模型的压缩词表映射在转换器里做）

mini 尺寸取「机制全保真、尺寸最小」：hc/压缩KV/索引器/Engram/MoE/共享专家/yarn/sinkhorn 全有。
运行：python mini_ref.py → 写 mini_ref.npz（权重 + 逐步 logits + 中间 hidden）。
"""
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

torch.manual_seed(42)
np.random.seed(42)

# ---------------------------------------------------------------- 量化原语（转录 kernel.py）

FP8_MAX = 448.0
FP4_MAX = 6.0


def log2_ceil(x: torch.Tensor) -> torch.Tensor:
    """kernel.fast_log2_ceil：对正数 ceil(log2(x))，位操作语义。x: fp32"""
    bits = x.view(torch.int32)
    exp = (bits >> 23) & 0xFF
    man = bits & ((1 << 23) - 1)
    return exp - 127 + (man != 0).to(torch.int32)


def round_scale(amax: torch.Tensor, qmax: float) -> torch.Tensor:
    """kernel.fast_round_scale：s = 2^ceil(log2(amax/qmax))"""
    return torch.exp2(log2_ceil(amax * (1.0 / qmax)).to(torch.float32))


def q_e4m3(x: torch.Tensor) -> torch.Tensor:
    """cast 到 fp8 e4m3 网格（RN）。"""
    return x.to(torch.float8_e4m3fn).to(torch.float32)


def q_e2m1(x: torch.Tensor) -> torch.Tensor:
    """cast 到 fp4 e2m1 网格 ±{0,.5,1,1.5,2,3,4,6}（RN，含亚常规）。"""
    sign = torch.sign(x)
    a = x.abs().clamp(max=FP4_MAX)
    # 亚常规（a < 2）：步长 0.5，网格 {0,.5,1,1.5}；round 为 banker's（ties-to-even）
    sub = torch.round(a / 0.5) * 0.5
    # 常规：网格 {2,3,4,6}；最近邻（严格平点概率 0，不影响对拍）
    cand = torch.stack([torch.full_like(a, 2.0), torch.full_like(a, 3.0), torch.full_like(a, 4.0), torch.full_like(a, 6.0)])
    d = (cand - a.unsqueeze(0)).abs()
    nrm2 = torch.tensor([2.0, 3.0, 4.0, 6.0])[(d).argmin(dim=0)]
    return sign * torch.where(a >= 2.0, nrm2, sub)


def act_quant_dequant(x: torch.Tensor, bs: int = 32) -> torch.Tensor:
    """kernel.act_quant(inplace=True)：fp8 + ue8m0 scale，返回假量化后的值。沿最后一维按 bs 分块。"""
    shp = x.shape
    xf = x.reshape(-1, shp[-1]).to(torch.float32)
    assert shp[-1] % bs == 0
    g = xf.view(*xf.shape[:-1], -1, bs)
    amax = g.abs().amax(-1, keepdim=True).clamp(min=1e-4)
    s = round_scale(amax, FP8_MAX)
    if os.environ.get("DS41_TRACE"):
        print(f"[py:qscale] sum={float(s.sum()):.6e} amax={float(s.abs().max()):.6e} neg={int((s < 0).sum())}")
    q = q_e4m3((g / s).clamp(-FP8_MAX, FP8_MAX))
    if os.environ.get("DS41_TRACE"):
        print(f"[py:qq] sum={float(q.sum()):.6e} first8={q.flatten()[:8].tolist()}")
    return (q * s).reshape(shp)


def fp4_act_quant_dequant(x: torch.Tensor, bs: int = 32, scale_kind: str = "e8m0") -> torch.Tensor:
    """kernel.fp4_act_quant(inplace=True)：fp4 e2m1 + e8m0 或 e4m3 scale。"""
    shp = x.shape
    xf = x.reshape(-1, shp[-1]).to(torch.float32)
    assert shp[-1] % bs == 0
    g = xf.view(*xf.shape[:-1], -1, bs)
    amax = g.abs().amax(-1, keepdim=True)
    if scale_kind == "e4m3":
        # 训练的压缩 KV：全零组 scale 也非零（floor 6·2^-9）
        s = q_e4m3((amax.clamp(min=6 * 2.0**-9) / FP4_MAX))
    else:
        s = round_scale(amax.clamp(min=6 * 2.0**-126), FP4_MAX)
    q = q_e2m1((g / s).clamp(-FP4_MAX, FP4_MAX))
    return (q * s).reshape(shp)


def fp4_weight_quant_dequant(w: torch.Tensor, bs: int = 32) -> torch.Tensor:
    """fp4 权重（per-row 1D 块 + e8m0，与 Linear fp4 的 scale [out, in/32] 对应）。"""
    return fp4_act_quant_dequant(w, bs, "e8m0")


# ---------------------------------------------------------------- 基础件（转录 model.py）


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        return (self.weight * (x * torch.rsqrt(var + self.eps))).to(x.dtype)


class Linear(nn.Module):
    """普通 fp32 线性（镜像注记：真模型为 fp8 权重块量化 matmul，mini 用 fp32 密集等价验证数学；
    激活侧的假量化点在模型里按原位置保留）。"""

    def __init__(self, din, dout):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dout, din) * 0.02)

    def forward(self, x):
        return F.linear(x.to(torch.float32), self.weight.to(torch.float32))


class Fp4ExpertLinear(nn.Module):
    """fp4 权重线性（专家用）：权重量化成 e2m1 网格（假量化），激活过 fp8 假量化，
    然后 fp32 matmul —— 等价于 fp4_gemm 的数学。"""

    def __init__(self, din, dout):
        super().__init__()
        w = torch.randn(dout, din) * 0.02
        self.register_buffer("wq", fp4_weight_quant_dequant(w))

    def forward(self, x):
        xq = act_quant_dequant(x, 32)
        return F.linear(xq, self.wq)


class ParallelEmbedding(nn.Module):
    def __init__(self, vocab, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab, dim) * 0.02)

    def forward(self, x):
        return F.embedding(x, self.weight)


def precompute_freqs(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow):
    """model.precompute_freqs_cis 的实数化：返回 cos/sin [seqlen, dim/2]（fp32）。"""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        def corrected_dim(rotations):
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    ang = torch.outer(torch.arange(seqlen, dtype=torch.float32), freqs)
    return ang.cos(), ang.sin()


def apply_rotary(x, cos, sin, inverse=False):
    """model.apply_rotary_emb 的实数化：相邻对 (x0,x1) 旋转；inverse 共轭（sin 取负）。
    x: [b,s,d] 或 [b,s,h,d]；cos/sin: [len, d/2]。"""
    s = -1.0 if inverse else 1.0
    x0 = x[..., 0::2]
    x1 = x[..., 1::2]
    L = x.shape[-3] if x.dim() == 4 else x.shape[-2]  # 4D: 序列维在 -3
    c = cos[-L:]
    sn = sin[-L:] * s
    if x.dim() == 4:  # [b,s,h,rd/2]：频率沿 s 广播，所有头共用
        c = c[None, :, None, :]
        sn = sn[None, :, None, :]
    else:  # [b,s,rd/2]
        c = c[None]
        sn = sn[None]
    y0 = x0 * c - x1 * sn
    y1 = x0 * sn + x1 * c
    return torch.stack([y0, y1], dim=-1).flatten(-2)


def get_window_topk_idxs(window_size, bsz, seqlen, start_pos):
    """model.get_window_topk_idxs 原样转录。"""
    if start_pos == 0:
        end = torch.arange(seqlen).unsqueeze(1)
        idxs = (end - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size))
        idxs = torch.where(idxs > end, -1, idxs)
    else:
        oldest = start_pos % window_size + 1
        idxs = torch.cat([torch.arange(oldest, window_size), torch.arange(oldest)])
        idxs = torch.where(idxs > start_pos, -1, idxs)
    return idxs.int().unsqueeze(0).expand(bsz, -1, -1).contiguous()


def sparse_attn_mirror(q, kv, attn_sink, topk_idxs, softmax_scale):
    """kernel.sparse_attn 的数学镜像：每 (b,m,h)：logit_j = q_h·kv_{idx_j}·scale（-1 槽 = -inf）；
    softmax 相对 scores_max（kernel 的 -1e30 有限下界约定），sink 作为一个 logit=sink_h、value=0 的额外槽。"""
    b, m, h, d = q.shape
    topk = topk_idxs.shape[-1]
    idx = topk_idxs.long().clamp(min=0)  # [b,m,topk]
    kvg = kv[:, None, None].expand(b, m, h, kv.shape[1], d)  # [b,m,h,n,d]：kv 行全头共享
    kvsel = torch.gather(kvg, 3, idx[:, :, None, :, None].expand(b, m, h, topk, d))
    logits = torch.einsum("bmhd,bmhtd->bmht", q.to(torch.float32), kvsel.to(torch.float32)) * softmax_scale
    valid = topk_idxs >= 0
    logits = torch.where(valid.unsqueeze(2), logits, torch.full_like(logits, float("-inf")))
    mx = logits.amax(-1, keepdim=True)
    has_valid = valid.any(-1, keepdim=True).unsqueeze(2)
    mx = torch.where(has_valid, mx, torch.full_like(mx, -1e30))  # kernel 的有限下界约定
    p = torch.exp(logits - mx)
    denom = p.sum(-1, keepdim=True) + torch.exp(attn_sink.view(1, 1, h, 1).to(torch.float32) - mx)
    o = torch.einsum("bmht,bmhtd->bmhd", p, kvsel.to(torch.float32)) / denom
    return o.to(q.dtype)


# ---------------------------------------------------------------- 模型

CFG = dict(
    vocab=256, dim=64, n_layers=2, n_heads=8, head_dim=32, rope_head_dim=8,
    q_lora_rank=32, o_groups=2, o_lora_rank=16, window_size=16,
    compress_ratios=(2, 0), kv_source_layers=(0,), index_source_layers=(0,),
    compress_rope_theta=40000.0, original_seq_len=32, rope_theta=10000.0,
    rope_factor=8.0, beta_fast=4, beta_slow=1,
    n_routed=8, n_activated=2, moe_inter=32, swiglu_limit=10.0,
    gate_temp=1.0, route_scale=1.5, score_func="sqrtsoftplus",
    index_n_heads=4, index_head_dim=32, index_topk=8,
    hc_mult=2, hc_sinkhorn_iters=5, hc_eps=1e-6, norm_eps=1e-20,
    engram_layer_ids=(1,), engram_num_embeddings=(1024,), engram_max_ngram=2,
    engram_n_heads=2, engram_head_dim=32,
    max_seq_len=64, n_dec_steps=4,
)


class Compressor(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ratio = c["compress_ratios"][0]
        self.hd = c["head_dim"]
        self.norm = RMSNorm(self.hd, c["norm_eps"])
        self.wkv = Linear(c["dim"], self.hd)  # fp32（ratio>1）
        self.wgate = Linear(c["dim"], self.hd)
        self.kv_state = torch.zeros(4, c["compress_ratios"][0], self.hd)
        self.score_state = torch.full((4, c["compress_ratios"][0], self.hd), -torch.inf)

    def forward(self, x, start_pos):
        b, s, _ = x.shape
        r = self.ratio
        if r == 1:
            return self.norm(self.wkv(x))
        kv, score = self.wkv(x.float()), self.wgate(x.float())
        if start_pos == 0:
            rem = s % r
            cut = s - rem
            if rem:
                kv, self.kv_state[:b, :rem] = kv[:, :cut], kv[:, cut:]
                score, self.score_state[:b, :rem] = score[:, :cut], score[:, cut:]
            kv = (kv.view(b, -1, r, self.hd) * score.view(b, -1, r, self.hd).softmax(2)).sum(2)
        else:
            ok = (start_pos + 1) % r == 0
            slot = start_pos % r
            self.kv_state[:b, slot] = kv.squeeze(1)
            self.score_state[:b, slot] = score.squeeze(1)
            if not ok:
                return None
            kv = (self.kv_state[:b] * self.score_state[:b].softmax(1)).sum(1, keepdim=True)
        return self.norm(kv.to(x.dtype))


class Indexer(nn.Module):
    def __init__(self, c, layer_id):
        super().__init__()
        self.c = c
        self.ratio = c["compress_ratios"][layer_id]
        self.h = c["index_n_heads"]
        self.hd = c["index_head_dim"]
        self.topk = c["index_topk"]
        self.scale = self.hd**-0.5
        self.wq_b = Linear(c["q_lora_rank"], self.h * self.hd)
        self.weights_proj = Linear(c["dim"], self.h)
        self.wk = Linear(c["head_dim"], self.hd)
        self.k_norm = RMSNorm(self.hd, c["norm_eps"])
        self.k_cache = torch.zeros(4, c["max_seq_len"] // self.ratio, self.hd)

    def forward(self, x, qr, latent, start_pos, fcos, fsin, offset):
        b, s, _ = x.shape
        r, rd, end_pos = self.ratio, self.c["rope_head_dim"], start_pos + s
        if latent is not None:
            # latent 代表组首 token：位置 j·r
            f_cos = fcos[0 : s - s % r : r] if start_pos == 0 else fcos[start_pos + 1 - r : start_pos + 2 - r]
            f_sin = fsin[0 : s - s % r : r] if start_pos == 0 else fsin[start_pos + 1 - r : start_pos + 2 - r]
            k = self.k_norm(self.wk(latent))
            k[..., -rd:] = apply_rotary(k[..., -rd:], f_cos, f_sin)
            k = fp4_act_quant_dequant(k, 32, "e8m0")
            self.k_cache[:b, start_pos // r : start_pos // r + k.shape[1]] = k
        q = self.wq_b(qr).view(b, s, self.h, self.hd)
        q[..., -rd:] = apply_rotary(q[..., -rd:], fcos[start_pos:end_pos], fsin[start_pos:end_pos])
        q = fp4_act_quant_dequant(q, 32, "e8m0")
        index_k = self.k_cache[:b, : end_pos // r]
        w = self.weights_proj(x) * (self.scale * self.h**-0.5)
        sc = torch.einsum("bshd,btd->bsht", q, index_k)
        sc = (sc.relu() * w.unsqueeze(-1)).sum(2)  # [b,s,T]
        # 可见性先 mask（不可达 = -inf），再 topk，再按位置序还原并加 offset
        if start_pos == 0:
            clens = (torch.arange(1, s + 1) // r).unsqueeze(-1)  # [s,1]
            sc = sc.masked_fill(torch.arange(s // r)[None, :] >= clens, float("-inf"))  # [s,T]
        if os.environ.get("DS41_TRACE") and s > 6:
            print(f"[py:scrow3]={sc[0, 3].tolist()}")
        topk = min(self.topk, end_pos // r)
        idxs = sc.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
        clens_d = end_pos // r if start_pos else clens
        return torch.where(idxs < clens_d, idxs + offset, torch.tensor(-1)).int()


class Attention(nn.Module):
    def __init__(self, layer_id, c, freqs):
        super().__init__()
        self.c, self.layer_id, self.freqs = c, layer_id, freqs
        d, hd = c["dim"], c["head_dim"]
        self.rd = c["rope_head_dim"]
        self.ratio = c["compress_ratios"][layer_id]
        self.win = c["window_size"]
        self.scale = hd**-0.5
        self.is_kv_source = layer_id in c["kv_source_layers"]
        self.is_index_source = layer_id in c["index_source_layers"]
        self.attn_sink = nn.Parameter(torch.zeros(c["n_heads"]))
        self.wq_a = Linear(d, c["q_lora_rank"])
        self.q_norm = RMSNorm(c["q_lora_rank"], c["norm_eps"])
        self.wq_b = Linear(c["q_lora_rank"], c["n_heads"] * hd)
        self.wkv = Linear(d, hd)
        self.kv_norm = RMSNorm(hd, c["norm_eps"])
        self.wo_a = nn.Parameter(torch.randn(c["o_groups"], c["o_lora_rank"], hd * c["n_heads"] // c["o_groups"]) * 0.02)
        self.wo_b = Linear(c["o_groups"] * c["o_lora_rank"], d)
        self.compressor = Compressor(c) if self.is_kv_source else None
        self.indexer = Indexer(c, layer_id) if self.is_index_source else None
        self.win_cache = torch.zeros(4, self.win, hd)
        self.compress_cache = (
            torch.zeros(4, c["max_seq_len"] // self.ratio, hd) if self.is_kv_source else None
        )
        # 窗口层 base theta 无 yarn；压缩层用压缩 theta + yarn
        if self.ratio:
            self.cf = precompute_freqs(self.rd, c["max_seq_len"], c["original_seq_len"],
                                       c["compress_rope_theta"], c["rope_factor"], c["beta_fast"], c["beta_slow"])
        else:
            self.cf = precompute_freqs(self.rd, c["max_seq_len"], 0, c["rope_theta"], 1, 32, 1)

    def _window_kv(self, x, start_pos, s):
        b = x.shape[0]
        kv = self.kv_norm(self.wkv(x))
        kv[..., -self.rd :] = apply_rotary(kv[..., -self.rd :], self.freqs[0][start_pos : start_pos + s], self.freqs[1][start_pos : start_pos + s])
        _stat(kv, "py:kv_rot")
        kv = act_quant_dequant(kv, 32)  # fp8 假量化（RoPE 后整向量，含尾部）
        _stat(kv, "py:kv_q")
        if start_pos == 0:
            if s <= self.win:
                self.win_cache[:b, :s] = kv
            else:
                cut = s % self.win
                self.win_cache[:b, cut:] = kv[:, -self.win + cut :]
                self.win_cache[:b, :cut] = kv[:, -cut:]
            wkv = kv
        else:
            self.win_cache[:b, start_pos % self.win] = kv.squeeze(1)
            wkv = self.win_cache[:b]
        return wkv, get_window_topk_idxs(self.win, b, s, start_pos)

    def _compress(self, x, qr, start_pos, s, offset):
        b = x.shape[0]
        r = self.ratio
        clen = (start_pos + s) // r
        latent = None
        if self.is_kv_source:
            latent = self.compressor(x, start_pos)
        idxs = None
        if self.is_index_source:
            idxs = self.indexer(x, qr, latent, start_pos, self.freqs[0], self.freqs[1], offset)
        if latent is not None:
            if start_pos == 0:
                f0, f1 = self.cf[0][0 : s - s % r : r], self.cf[1][0 : s - s % r : r]
            else:
                f0, f1 = self.cf[0][start_pos + 1 - r : start_pos + 2 - r], self.cf[1][start_pos + 1 - r : start_pos + 2 - r]
            latent[..., -self.rd :] = apply_rotary(latent[..., -self.rd :], f0, f1)
            latent = fp4_act_quant_dequant(latent, 16, "e4m3")  # 组 16 + e4m3
            self.compress_cache[:b, start_pos // r : start_pos // r + latent.shape[1]] = latent
        return self.compress_cache[:b, :clen] if self.is_kv_source else None, idxs

    def forward(self, x, start_pos, shared):
        b, s, _ = x.shape
        fc0 = self.freqs[0][start_pos : start_pos + s]
        fc1 = self.freqs[1][start_pos : start_pos + s]
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).view(b, s, self.c["n_heads"], self.c["head_dim"])
        q[..., -self.rd :] = apply_rotary(q[..., -self.rd :], fc0, fc1)

        kv, widx = self._window_kv(x, start_pos, s)
        _stat(q, "py:q")
        _stat(kv, "py:kv")
        if os.environ.get("DS41_TRACE") and start_pos == 11 and self.layer_id == 0:
            np.save("/tmp/d_q_py.npy", q.detach().numpy())
            np.save("/tmp/d_kv_py.npy", kv.detach().numpy())
        if self.ratio:
            ckv, cidx = self._compress(x, qr, start_pos, s, kv.shape[1])
            kv = torch.cat([kv, ckv], 1)
            idxs = torch.cat([widx, cidx], -1)
        else:
            idxs = widx
        if os.environ.get("DS41_TRACE") and start_pos == 11 and self.layer_id == 0:
            np.save("/tmp/d_kvf_py.npy", kv.detach().numpy())
            np.save("/tmp/d_idx_py.npy", idxs.detach().numpy())
        o = sparse_attn_mirror(q, kv, self.attn_sink, idxs, self.scale)
        if os.environ.get("DS41_TRACE") and start_pos == 11 and self.layer_id == 0:
            np.save("/tmp/d_o_py.npy", o.detach().numpy())
        o[..., -self.rd :] = apply_rotary(o[..., -self.rd :], fc0, fc1, inverse=True)
        g = self.c["o_groups"]
        o = o.view(b, s, g, -1)
        o = act_quant_dequant(o, 32)  # o 投影前的 fp8 假量化
        og = torch.einsum("bsgd,grd->bsgr", o.to(torch.float32), self.wo_a.to(torch.float32))
        return self.wo_b(og.flatten(2))


class Gate(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.topk = c["n_activated"]
        self.w = nn.Parameter(torch.randn(c["n_routed"], c["dim"]) * 0.02)
        self.bias = nn.Parameter(torch.zeros(c["n_routed"]))

    def forward(self, x):
        sc = F.linear(x.float(), self.w.float()) / self.c["gate_temp"]
        f = self.c["score_func"]
        if f == "softmax":
            sc = sc.softmax(-1)
        elif f == "sigmoid":
            sc = sc.sigmoid()
        else:
            sc = F.softplus(sc).sqrt()
        idx = (sc + self.bias).topk(self.topk, -1)[1]
        w = sc.gather(1, idx)
        if self.topk > 1:
            w = w / (w.sum(-1, keepdim=True) + 1e-20)  # 1e-20，不是 norm_eps
        return w * self.c["route_scale"], idx


class Expert(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.w1 = Fp4ExpertLinear(c["dim"], c["moe_inter"])
        self.w2 = Fp4ExpertLinear(c["moe_inter"], c["dim"])
        self.w3 = Fp4ExpertLinear(c["dim"], c["moe_inter"])
        self.lim = c["swiglu_limit"]

    def forward(self, x, weights=None):
        gate = self.w1(x).float()
        up = self.w3(x).float()
        if self.lim > 0:
            up = up.clamp(-self.lim, self.lim)
            gate = gate.clamp(max=self.lim)
        y = F.silu(gate) * up
        if weights is not None:
            y = weights * y
        return self.w2(y.to(x.dtype))


class MoE(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.gate = Gate(c)
        self.experts = nn.ModuleList([Expert(c) for _ in range(c["n_routed"])])
        self.shared = Expert(c)

    def forward(self, x):
        shp = x.shape
        x = x.view(-1, self.c["dim"])
        w, idx = self.gate(x)
        y = torch.zeros_like(x)
        counts = torch.bincount(idx.flatten(), minlength=self.c["n_routed"])
        for i in range(self.c["n_routed"]):
            if counts[i] == 0:
                continue
            sel = torch.where(idx == i)
            y[sel[0]] += self.experts[i](x[sel[0]], w[sel[0], sel[1], None])
        y += self.shared(x)
        return y.view(shp)


class Engram(nn.Module):
    def __init__(self, c, layer_hash_index, primes):
        super().__init__()
        self.idx = layer_hash_index
        dim, hc = c["dim"], c["hc_mult"]
        self.hc = hc
        rows, ehd = c["engram_num_embeddings"][layer_hash_index], c["engram_head_dim"]
        self.n_hash_cols = (c["engram_max_ngram"] - 1) * c["engram_n_heads"]
        # 表按 ParallelEngramEmbedding 的存储量化（fp8 行 + 32 块 scale），镜像直接存反量化值
        tab = torch.randn(rows, ehd) * 0.05
        self.register_buffer("table", act_quant_dequant(tab, 32))
        self.wkv = Linear(self.n_hash_cols * ehd, dim * (hc + 1))
        self.q_weight = nn.Parameter(torch.ones(hc, dim))
        self.k_weight = nn.Parameter(torch.ones(hc, dim))
        self.primes = primes  # [ngram_sizes][heads]
        rng = np.random.default_rng(10007 * (layer_hash_index + 1))
        self.mult = torch.from_numpy(rng.integers(0, 1 << 20, size=(primes.shape[0] + 1,)).astype(np.int64) * 2 + 1)
        self.eps = c["norm_eps"]
        self.pad_id = 7  # 随机 token_map 里的 pad 位置
        self.cache = torch.zeros(4, c["max_seq_len"], dtype=torch.long)  # NgramHashState 的缓存

    def lookup(self, ids, start_pos):
        """NgramHashState.forward 的镜像：压缩 id → 滚动 XOR 哈希 → 每桶取模。
        mini 的 token_map 恒等（vocab 256 即压缩空间）；multipliers 固定种子。"""
        b, s = ids.shape
        ng = self.primes.shape[0]
        self.cache[:b, start_pos : start_pos + s] = ids
        mult = self.mult
        pos = torch.arange(start_pos, start_pos + s)
        tokens = []
        blocked = torch.zeros(b, s, dtype=torch.bool)
        for shift in range(ng + 1):
            src = self.cache[:b].gather(1, (pos - shift).clamp(min=0).expand(b, s))
            blocked = blocked | (pos < shift).unsqueeze(0) | (src == -1)
            tokens.append(torch.where(blocked, torch.tensor(self.pad_id), src))
        tokens = torch.stack(tokens, -1)  # [b,s,ng+1]
        prods = tokens * mult  # [b,s,ng+1]
        rolling = prods[..., 0]  # [b,s]
        cols = []
        for i in range(1, ng + 1):
            rolling = torch.bitwise_xor(rolling, prods[..., i])
            for hh in range(self.primes.shape[1]):
                cols.append((rolling % self.primes[i - 1][hh]).to(torch.long))
        out = torch.stack(cols, -1)  # [b,s,n_hash_cols]
        if os.environ.get("DS41_TRACE") and s > 6:
            print(f"[py:rows]=", out[0, 3].tolist(), out[1, 7].tolist())
        return out

    def forward(self, x, ids, start_pos):
        b, s, hc, d = x.shape
        rows = self.lookup(ids, start_pos)  # [b,s,n_hash_cols]
        vals = self.table[rows]  # [b,s,cols,ehd]
        kv = self.wkv(vals.flatten(-2))
        key, value = kv.split([hc * d, d], -1)
        key = key.float().view(b, s, hc, d)
        wgt = self.q_weight.float() * self.k_weight.float()
        h = x.float()
        rstd = torch.rsqrt(h.square().mean(-1) + self.eps) * torch.rsqrt(key.square().mean(-1) + self.eps)
        dot = (h * wgt * key).sum(-1) * rstd * d**-0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)


def sinkhorn_split(mixes, scale, base, hc, iters, eps):
    """kernel.hc_split_sinkhorn 的数学镜像。mixes: [n, (2+hc)·hc]"""
    pre = torch.sigmoid(mixes[:, :hc] * scale[0] + base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[:, hc : 2 * hc] * scale[1] + base[hc : 2 * hc])
    comb = mixes[:, 2 * hc :].reshape(-1, hc, hc) * scale[2] + base[2 * hc :].view(hc, hc)
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


class Block(nn.Module):
    def __init__(self, layer_id, c):
        super().__init__()
        self.c = c
        self.layer_id = layer_id
        self.attn = Attention(layer_id, c, FREQS)
        self.ffn = MoE(c)
        self.attn_norm = RMSNorm(c["dim"], c["norm_eps"])
        self.ffn_norm = RMSNorm(c["dim"], c["norm_eps"])
        hc = c["hc_mult"]
        self.hc_attn_fn = nn.Parameter(torch.randn((2 + hc) * hc, hc * c["dim"]) * 0.02)
        self.hc_ffn_fn = nn.Parameter(torch.randn((2 + hc) * hc, hc * c["dim"]) * 0.02)
        self.hc_attn_base = nn.Parameter(torch.zeros((2 + hc) * hc))
        self.hc_ffn_base = nn.Parameter(torch.zeros((2 + hc) * hc))
        self.hc_attn_scale = nn.Parameter(torch.ones(3))
        self.hc_ffn_scale = nn.Parameter(torch.ones(3))

    def hc_mixes(self, x, fn, scale, base):
        b, s, hc, d = x.shape
        xf = x.flatten(2).float()
        rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.c["norm_eps"])
        mixes = (F.linear(xf, fn.float()) * rsqrt).view(b * s, -1)
        pre, post, comb = sinkhorn_split(
            mixes, scale, base, hc, self.c["hc_sinkhorn_iters"], self.c["hc_eps"]
        )
        if os.environ.get("DS41_TRACE") and self.c is CFG and s > 6 and fn is self.hc_attn_fn:
            _stat(post.view(b, s, hc), "py:L0.a_post")
            _stat(comb.view(b, s, hc, hc), "py:L0.a_comb")
        return (pre.view(b, s, hc), post.view(b, s, hc), comb.view(b, s, hc, hc))

    def forward(self, x, start_pos, pre_mix):
        residual = x
        a_pre, a_post, a_comb = self.hc_mixes(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        y = (pre_mix.unsqueeze(-1) * x.float()).sum(2).to(x.dtype)  # hc_pre
        y = self.attn_norm(y)
        y = self.attn(y, start_pos, None)
        x = a_post.unsqueeze(-1) * y.unsqueeze(-2) + (a_comb.unsqueeze(-1) * residual.unsqueeze(-2)).sum(2)  # hc_post
        x = x.to(residual.dtype)

        residual = x
        f_pre, f_post, f_comb = self.hc_mixes(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        if os.environ.get("DS41_TRACE") and x.shape[1] > 6 and self.layer_id == 0:
            np.save("/tmp/fpost_py.npy", f_post.detach().numpy())
            np.save("/tmp/fcomb_py.npy", f_comb.detach().numpy())
            np.save("/tmp/res2_py.npy", residual.detach().numpy())
        y = (a_pre.unsqueeze(-1) * x.float()).sum(2).to(x.dtype)
        y = self.ffn_norm(y)
        y = self.ffn(y)
        x = f_post.unsqueeze(-1) * y.unsqueeze(-2) + (f_comb.unsqueeze(-1) * residual.unsqueeze(-2)).sum(2)
        if os.environ.get("DS41_TRACE") and x.shape[1] > 6 and self.layer_id == 0:
            np.save("/tmp/comb_py.npy", x.detach().numpy())
        x = x.to(residual.dtype)
        return x, f_pre


FREQS = None
_OGN = 0


class Transformer(nn.Module):
    def __init__(self, c):
        super().__init__()
        global FREQS
        FREQS = precompute_freqs(c["rope_head_dim"], c["max_seq_len"], c["original_seq_len"],
                                 c["rope_theta"], c["rope_factor"], c["beta_fast"], c["beta_slow"])
        self.c = c
        self.embed = ParallelEmbedding(c["vocab"], c["dim"])
        # mini：只有 layer0 压缩（kv+index source）；layer1 放 Engram（参考实现：Engram 在 Block 之前）
        self.engram = Engram(c, 0, torch.tensor([[101], [103]])) if 1 in c["engram_layer_ids"] else None
        self.blocks = nn.ModuleList([Block(i, c) for i in range(c["n_layers"])])
        self.norm = RMSNorm(c["dim"], c["norm_eps"])
        self.head_w = nn.Parameter(torch.randn(c["vocab"], c["dim"]) * 0.02)

    def forward(self, ids, start_pos):
        b, s = ids.shape
        c = self.c
        h = self.embed(ids).unsqueeze(2).repeat(1, 1, c["hc_mult"], 1)
        pre_mix = torch.zeros(b, s, c["hc_mult"])
        pre_mix[:, :, 0] = 1.0  # make_identity_pre_mix：one-hot
        for i, blk in enumerate(self.blocks):
            if self.engram is not None and i in c["engram_layer_ids"]:
                h = self.engram(h, ids, start_pos)
            h, pre_mix = blk(h, start_pos, pre_mix)
        h = (pre_mix.unsqueeze(-1) * h.float()).sum(2)  # 最后的 hc_pre
        return F.linear(self.norm(h).to(torch.float32), self.head_w.to(torch.float32))


def _stat(t, tag):
    print(f"[{tag}] sum={float(t.float().sum()):.6e} amax={float(t.float().abs().max()):.6e}")


@torch.inference_mode()
def main():
    c = CFG
    model = Transformer(c)
    ids = torch.randint(0, c["vocab"], (2, 12))
    # prefill
    for name, mod in [
        ("embed", model.embed),
        ("L0.attn", model.blocks[0].attn),
        ("L0.ffn", model.blocks[0].ffn),
        ("engram", model.engram),
        ("L1.attn", model.blocks[1].attn),
        ("L1.ffn", model.blocks[1].ffn),
    ]:
        if mod is not None:
            mod.register_forward_hook(lambda m, i, o, n=name: _stat(o, f"py:{n}"))
    for name, mod in [("L0.ffn_in", model.blocks[0].ffn), ("L1.ffn_in", model.blocks[1].ffn)]:
        mod.register_forward_pre_hook(lambda m, a, n=name: _stat(a[0], f"py:{n}"))
    model.blocks[0].attn.register_forward_hook(
        lambda m, i, o: np.save("/tmp/at0_py.npy", o.detach().numpy()) if i[0].shape[1] > 6 else None
    )
    model.blocks[0].ffn.register_forward_hook(
        lambda m, i, o: np.save("/tmp/ff0_py.npy", o.detach().numpy()) if i[0].shape[1] > 6 else None
    )
    model.blocks[0].register_forward_hook(
        lambda m, i, o: np.save("/tmp/blk0_py.npy", o[0].detach().numpy()) if o[0].shape[1] > 6 else None
    )
    if model.engram is not None:
        model.engram.register_forward_hook(
            lambda m, i, o: np.save("/tmp/eg_py.npy", o.detach().numpy()) if i[0].shape[1] > 6 else None
        )
    logits = model(ids, 0)
    outs = {"ids": ids.numpy(), "logits_pre": logits[:, -1].numpy()}
    # decode
    toks = torch.cat([ids, torch.zeros(2, c["n_dec_steps"], dtype=torch.long)], 1)
    for i in range(12, 12 + c["n_dec_steps"]):
        logits = model(toks[:, i - 1 : i], i - 1)
        nxt = logits[:, -1].argmax(-1)  # temperature 0
        toks[:, i] = nxt
        outs[f"logits_{i}"] = logits[:, -1].numpy()
        outs[f"tok_{i}"] = nxt.numpy()
    # 权重
    sd = model.state_dict()
    for k, v in sd.items():
        outs["w_" + k] = v.detach().numpy()
    outs["engram_mult_0"] = model.engram.mult.numpy()
    outs["engram_primes"] = model.engram.primes.numpy()
    outs["cfg"] = np.frombuffer(json.dumps(c).encode(), dtype=np.uint8)
    np.savez("mini_ref.npz", **outs)
    print("logits_pre[0,:6] =", logits[:, -1][0, :6].tolist())
    print("toks:", toks[:, 12:].tolist())
    print("saved", len(outs), "arrays")


if __name__ == "__main__":
    main()
