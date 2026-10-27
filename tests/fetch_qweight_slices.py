"""从真实的 AWQ / GPTQ 检查点里只取几个张量（HTTP Range），生成"真实落盘布局"夹具。

为什么不整包下载：磁盘只剩几 GB，而我们只需要证明"字段布局与真实文件一致"。
每个张量的字节都从 HuggingFace 上按 Range 取，只有几 MB。

参考反量化：
  AWQ  直接照抄 AutoAWQ 官方实现（awq/utils/packing_utils.py 的
       unpack_awq / reverse_awq_order / dequantize_gemm），逐行未改，
       出处见下方 official_awq_dequant 的注释（Apache-2.0）。
  GPTQ 按 GPTQModel nn_modules/qlinear/__init__.py::_dequantize_from_codes
       的语义（scales[g_idx] * (q - zeros[g_idx])），与 C 侧零共享。

用法（仓库根目录）：
    python sglang-c/tests/fetch_qweight_slices.py            # 默认落 models/qweight-real
产出：
    models/qweight-real/awq/model.safetensors   + config.json + refmeta/refbin
    models/qweight-real/gptq/model.safetensors  + config.json + refmeta/refbin
"""
import json
import os
import struct
import sys
import urllib.request

import numpy as np
import torch

# 真实检查点（Qwen 官方发布，未经二次加工）
AWQ_REPO = "Qwen/Qwen2.5-0.5B-Instruct-AWQ"
GPTQ_REPO = "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4"
# 取这几个张量：覆盖 (a) K!=N 的 gate_proj、(b) N 很小的 k_proj、(c) K=4864 的 down_proj
WANTED = ["model.layers.0.mlp.gate_proj", "model.layers.0.self_attn.k_proj",
          "model.layers.0.mlp.down_proj"]


def http(url, rng=None):
    req = urllib.request.Request(url, headers={"Range": rng} if rng else {})
    return urllib.request.urlopen(req, timeout=120).read()


def get_config(repo):
    url = "https://huggingface.co/%s/resolve/main/config.json" % repo
    return json.loads(http(url).decode())


def read_header(repo, fn="model.safetensors"):
    base = "https://huggingface.co/%s/resolve/main/" % repo
    n = struct.unpack("<Q", http(base + fn, "bytes=0-7"))[0]
    hdr = json.loads(http(base + fn, "bytes=8-%d" % (7 + n)).decode())
    return base + fn, n, hdr


def fetch_tensor(base, hlen, meta):
    s, t = meta["data_offsets"]
    off = 8 + hlen + s
    raw = http(base, "bytes=%d-%d" % (off, 8 + hlen + t - 1))
    if len(raw) != t - s:
        raise SystemExit("range fetch short: %d != %d" % (len(raw), t - s))
    dt = meta["dtype"]
    npdt = {"F16": np.float16, "I32": np.int32, "F32": np.float32,
            "BF16": np.uint16}[dt]
    arr = np.frombuffer(raw, dtype=npdt).reshape(meta["shape"])
    if dt == "BF16":
        arr = (arr.astype(np.uint32) << 16).view(np.float32)
    return arr


# --------------------------------------------------------------- 官方参考实现


def official_awq_dequant(qweight, qzeros, scales, bits, group_size):
    """照抄 AutoAWQ（casper-hansen/AutoAWQ, Apache-2.0）
    awq/utils/packing_utils.py::unpack_awq + reverse_awq_order + dequantize_gemm。
    逐行未改，保持与上游完全一致。"""
    AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]

    def unpack_awq(qweight, qzeros, bits):
        shifts = torch.arange(0, 32, bits, device=qzeros.device)
        iweights = torch.bitwise_right_shift(qweight[:, :, None], shifts[None, None, :]).to(
            torch.int8
        )
        iweights = iweights.view(iweights.shape[0], -1)
        izeros = torch.bitwise_right_shift(qzeros[:, :, None], shifts[None, None, :]).to(
            torch.int8
        )
        izeros = izeros.view(izeros.shape[0], -1)
        return iweights, izeros

    def reverse_awq_order(iweights, izeros, bits):
        reverse_order_tensor = torch.arange(iweights.shape[-1], dtype=torch.int32,
                                            device=izeros.device)
        reverse_order_tensor = reverse_order_tensor.view(-1, 32 // bits)
        reverse_order_tensor = reverse_order_tensor[:, AWQ_REVERSE_ORDER]
        reverse_order_tensor = reverse_order_tensor.view(-1)
        izeros = izeros[:, reverse_order_tensor]
        iweights = iweights[:, reverse_order_tensor]
        return iweights, izeros

    iweight, izeros = unpack_awq(qweight, qzeros, bits)
    iweight, izeros = reverse_awq_order(iweight, izeros, bits)
    iweight = torch.bitwise_and(iweight, (2 ** bits) - 1)
    izeros = torch.bitwise_and(izeros, (2 ** bits) - 1)
    scales = scales.repeat_interleave(group_size, dim=0)
    izeros = izeros.repeat_interleave(group_size, dim=0)
    iweight = (iweight - izeros) * scales
    return iweight  # [K, N]


def gptq_dequant(qw, qz, sc, g_idx, bits, group_size):
    """GPTQModel _dequantize_from_codes 的等价实现（numpy）"""
    pack = 32 // bits
    K = qw.shape[0] * pack
    N = qw.shape[1]
    G = group_size if group_size > 0 else K
    zc = np.zeros((qz.shape[0], N), dtype=np.int32)
    for g in range(qz.shape[0]):
        for n in range(N):
            zc[g, n] = (int(qz[g, n // pack]) >> (bits * (n % pack))) & (2 ** bits - 1)
    w = np.zeros((N, K), dtype=np.float32)
    sc32 = sc.astype(np.float32)
    for k in range(K):
        g = int(g_idx[k]) if g_idx is not None else k // G
        q = (qw[k // pack, :] >> (bits * (k % pack))) & (2 ** bits - 1)
        w[:, k] = (q - zc[g, :]) * sc32[g, :]
    return w


def save_st(path, tensors):
    from safetensors.numpy import save_file
    save_file({k: np.ascontiguousarray(v) for k, v in tensors.items()}, path)


def export(repo, out, method):
    cfg = get_config(repo)
    qc = cfg.get("quantization_config", {})
    bits = qc.get("bits", 4)
    group = qc.get("group_size", 128)
    base, hlen, hdr = read_header(repo)
    tensors = {}
    refs = []
    for want in WANTED:
        got = {}
        for suf in (".qweight", ".qzeros", ".scales", ".g_idx"):
            k = want + suf
            if k in hdr:
                got[suf] = fetch_tensor(base, hlen, hdr[k])
        if ".qweight" not in got:
            print("  skip", want, "(not in checkpoint)")
            continue
        short = want.split(".")[-2] + "_" + want.split(".")[-1]
        for suf, arr in got.items():
            tensors[want + suf] = arr
        if method == "awq":
            ref = official_awq_dequant(torch.from_numpy(got[".qweight"]),
                                       torch.from_numpy(got[".qzeros"]),
                                       torch.from_numpy(got[".scales"].astype(np.float32)),
                                       bits, group).numpy()
            ref = ref.T.copy()  # 上游是 [K, N]，本项目用 HF 的 [N, K]
        else:
            ref = gptq_dequant(got[".qweight"], got[".qzeros"], got[".scales"],
                               got.get(".g_idx"), bits, group)
        refs.append((want + ".weight", ref))
        print("  %-45s %s qweight=%s scales=%s" % (want, method, got[".qweight"].shape,
                                                  got[".scales"].shape))
    os.makedirs(out, exist_ok=True)
    save_st(os.path.join(out, "model.safetensors"), tensors)
    c = {"hidden_size": cfg["hidden_size"], "quantization_config": qc}
    json.dump(c, open(os.path.join(out, "config.json"), "w"), indent=1)

    # 参考反量化结果（HF 的 [N,K] 行主序），供 C 测试逐元素比
    meta, blob = [], b""
    for name, ref in refs:
        arr = np.ascontiguousarray(ref, dtype=np.float32)
        meta.append("%s\t%d\t%d,%d\t%d" % (name, 2, arr.shape[0], arr.shape[1], len(blob)))
        blob += arr.tobytes()
    open(os.path.join(out, "refmeta"), "w").write("\n".join(meta) + "\n")
    open(os.path.join(out, "refbin"), "wb").write(blob)
    print("  -> %s (%d 张量, 参考 %d 字节)" % (out, len(tensors), len(blob)))


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "models/qweight-real"
    print("AWQ  %s" % AWQ_REPO)
    export(AWQ_REPO, os.path.join(root, "awq"), "awq")
    print("GPTQ %s" % GPTQ_REPO)
    export(GPTQ_REPO, os.path.join(root, "gptq"), "gptq")


if __name__ == "__main__":
    main()