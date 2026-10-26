"""独立参考实现：numpy GPT-2 forward，对照 sglang-c 的 C 实现。
与 C 代码零共享，仅共用同一份权重文件和 tiktoken 编码。"""
import numpy as np
import tiktoken
from safetensors.numpy import load_file

W = load_file("models/gpt2/model.safetensors")
enc = tiktoken.get_encoding("gpt2")

def ln(x, g, b, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = x.var(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * g + b

def gelu(x):  # gelu_new (tanh)
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

def forward(ids):
    x = W["wte.weight"][ids] + W["wpe.weight"][np.arange(len(ids))]
    for l in range(12):
        h = ln(x, W[f"h.{l}.ln_1.weight"], W[f"h.{l}.ln_1.bias"])
        qkv = h @ W[f"h.{l}.attn.c_attn.weight"] + W[f"h.{l}.attn.c_attn.bias"]
        q, k, v = np.split(qkv, 3, axis=-1)
        out = np.zeros_like(q)
        for t in range(len(ids)):
            for hd in range(12):
                qh = q[t, hd * 64:(hd + 1) * 64]
                ks = k[:t + 1, hd * 64:(hd + 1) * 64]
                vs = v[:t + 1, hd * 64:(hd + 1) * 64]
                sc = ks @ qh / np.sqrt(64.0)
                sc = np.exp(sc - sc.max())
                out[t, hd * 64:(hd + 1) * 64] = sc @ vs / sc.sum()
        x = x + out @ W[f"h.{l}.attn.c_proj.weight"] + W[f"h.{l}.attn.c_proj.bias"]
        h = ln(x, W[f"h.{l}.ln_2.weight"], W[f"h.{l}.ln_2.bias"])
        h = gelu(h @ W[f"h.{l}.mlp.c_fc.weight"] + W[f"h.{l}.mlp.c_fc.bias"])
        x = x + h @ W[f"h.{l}.mlp.c_proj.weight"] + W[f"h.{l}.mlp.c_proj.bias"]
    x = ln(x, W["ln_f.weight"], W["ln_f.bias"])
    return x[-1] @ W["wte.weight"].T

prompt = "The meaning of life is"
ids = enc.encode(prompt)
logits = forward(ids)
top5 = np.argsort(logits)[::-1][:5]
print("prompt ids:", ids)
print("numpy greedy top5:", [(int(i), enc.decode([int(i)]), float(logits[i])) for i in top5])
print("numpy greedy next token:", int(top5[0]), repr(enc.decode([int(top5[0])])))

# 连续 greedy 生成 12 个 token
seq = ids[:]
for _ in range(12):
    seq.append(int(np.argmax(forward(seq))))
print("numpy greedy continuation:", repr(enc.decode(seq[len(ids):])))
