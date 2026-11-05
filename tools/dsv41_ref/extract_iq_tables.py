"""从 ggml-common.h 提取 IQ 量化查表（GGML_TABLE_BEGIN/END 宏形式）-> sglang-c/src/ds41_iq_tables.h"""
import re

src = open("ggml-common.h", encoding="utf-8", errors="replace").read()

def grab(name):
    m = re.search(r"GGML_TABLE_BEGIN\(\s*\w+\s*,\s*" + name + r"\s*,\s*([A-Za-z0-9_]+)\s*\)", src)
    if not m:
        raise RuntimeError(f"找不到表 {name}")
    i = m.end()
    j = src.find("GGML_TABLE_END()", i)
    body = src[i:j].strip()
    body = re.sub(r"\n\s*\n", "\n", body)
    n = m.group(1)
    return body, n

out = ["// 自动生成：从 llama.cpp ggml-common.h 提取的 IQ 量化查表（勿手改）",
       "#pragma once",
       "#include <cstdint>",
       "#define NGRID_IQ1S 2048",
       "namespace ds41_iq {"]
for name, typ in [("kmask_iq2xs", "uint8_t"), ("ksigns_iq2xs", "uint8_t"),
                  ("iq2xxs_grid", "uint64_t"), ("iq2xs_grid", "uint64_t"), ("kvalues_iq4nl", "int8_t"),
                  ("iq1s_grid", "uint64_t")]:
    body, n = grab(name)
    out.append(f"static const {typ} {name}[{n}] = {{")
    out.append(body)
    out.append("};")
    out.append("")
out.append("} // namespace ds41_iq")
data = "\n".join(out)
open("sglang-c/src/ds41_iq_tables.h", "w", encoding="utf-8").write(data)
print(f"写 sglang-c/src/ds41_iq_tables.h，{len(data)} 字符")
# 校验条目数
for name in ("kmask_iq2xs", "ksigns_iq2xs", "iq2xxs_grid", "kvalues_iq4nl", "iq1s_grid"):
    seg = data[data.find(name):]
    cnt = seg[:seg.find(";")].count(",") + 1
    print(f"  {name}: ~{cnt} 项")
