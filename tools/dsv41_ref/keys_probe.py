import mlx.core as mx

W = mx.load("./models/dsv41/mlx-dense/model.safetensors")
ks = set(W.keys())
n_main = len([k for k in ks if not k.endswith((".scales", ".biases"))])
print("total keys:", len(ks), "main tensors:", n_main)
need = []
for l in [0, 1, 39]:
    for base in [
        f"blk.{l}.attn_norm.weight",
        f"blk.{l}.attn_q_a.weight",
        f"blk.{l}.attn_q_a_norm.weight",
        f"blk.{l}.attn_q_b.weight",
        f"blk.{l}.attn_kv.weight",
        f"blk.{l}.attn_output_a.weight",
        f"blk.{l}.attn_output_b.weight",
        f"blk.{l}.hc_attn_fn.weight",
        f"blk.{l}.hc_attn_base.weight",
        f"blk.{l}.attn_sinks.weight",
        f"blk.{l}.attn_comp_wkv.weight",
        f"blk.{l}.attn_comp_wgate.weight",
        f"blk.{l}.compressor_kv.weight",
        f"blk.{l}.wq_a.weight",
        f"blk.{l}.wq_b.weight",
        f"blk.{l}.wo_a.weight",
        f"blk.{l}.wo_b.weight",
        f"blk.{l}.ffn_norm.weight",
        f"blk.{l}.ffn_gate_inp.weight",
        f"blk.{l}.ffn_gate_shexp.weight",
        f"blk.{l}.ffn_up_shexp.weight",
        f"blk.{l}.ffn_down_shexp.weight",
        f"blk.{l}.exp_probs_b.bias",
        f"blk.{l}.hc_ffn_fn.weight",
        f"blk.{l}.hc_ffn_base.weight",
    ]:
        if base not in ks:
            need.append(base)
for g in ["token_embd.weight", "output_norm.weight", "output.weight"]:
    if g not in ks:
        need.append(g)
print("missing critical:", need if need else "none")
# q4 张量是否都带 scales/biases
for k in sorted(ks):
    if k.endswith(".scales"):
        continue
    if k + ".scales" in ks and k + ".biases" not in ks:
        print("missing biases for", k)
