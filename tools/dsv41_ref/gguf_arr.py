import numpy as np
from gguf import GGUFReader

DT = {0: np.uint8, 1: np.int8, 2: np.uint16, 3: np.int16, 4: np.uint32, 5: np.int32,
      6: np.float32, 7: np.uint8, 10: np.uint64, 11: np.int64, 12: np.float64}

base = "/Volumes/9100 PRO/models/dsv41/"
r = GGUFReader(base + "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00006.gguf")

keys = ["deepseek41.attention.compress_ratios", "deepseek41.attention.kv_source_layers",
        "deepseek41.attention.indexer.source_layers", "deepseek41.attention.indexer.candidate_source_layer",
        "deepseek41.ple.layers", "deepseek41.ple.ngram_size", "deepseek41.ple.heads_per_ngram",
        "deepseek41.ple.layer_multipliers", "deepseek41.ple.head_offsets", "deepseek41.ple.head_vocab_sizes",
        "deepseek41.engram.pad_token_id", "deepseek41.engram.token_map", "deepseek41.hash_layer_count",
        "deepseek41.expert_gating_func", "general.file_type", "deepseek41.attention.indexer.candidate_top_k_blocks"]
for k in keys:
    f = r.fields[k]
    if f.types[0] == 8:
        vals = bytes(b"".join(p.tobytes() for p in f.parts[1:])).decode("utf-8", "replace")
    elif f.types[0] == 9:  # ARRAY：types[1] 是元素类型
        raw = b"".join(p.tobytes() for p in f.parts[1:])
        vals = np.frombuffer(raw, dtype=DT[f.types[1]]).tolist()
    else:
        vals = f.parts[-1].tolist()
    if isinstance(vals, list) and len(vals) > 48:
        vals = f"[{len(vals)} 项] 前16={vals[:16]} 后8={vals[-8:]}"
    print(f"{k} = {vals}")
