"""端点覆盖测试：起服务后逐个端点实测（CPU 与 GPU 共用）。

用法：python tests/ext_endpoints_test.py [port]
  默认 30111（CPU）；GPU 服务用 python tests/ext_endpoints_test.py 30112
"""
import json
import os
import sys
import urllib.error
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("SGL_TEST_PORT", 30111))
BASE = "http://127.0.0.1:%d" % PORT
fails = []


def call(path, body=None, method=None, raw=False, timeout=120):
    url = BASE + path
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            txt = r.read().decode("utf-8", "replace")
            return r.status, (txt if raw else json.loads(txt) if txt else None)
    except urllib.error.HTTPError as e:
        txt = e.read().decode("utf-8", "replace")
        return e.code, txt
    except Exception as e:  # noqa: BLE001
        return -1, str(e)


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else "  <- " + str(detail)[:300]))
    if not cond:
        fails.append(name)


# 基线
st, r = call("/health")
check("GET /health", st == 200 and r.get("ok") is True, r)

# 分词
st, r = call("/tokenize", {"text": "Hello world"})
check("POST /tokenize", st == 200 and r.get("count", 0) > 0 and len(r.get("tokens", [])) == r["count"], r)
toks = r.get("tokens") if isinstance(r, dict) else None
st, r2 = call("/v1/tokenize", {"prompt": "Hello world"})
check("POST /v1/tokenize", st == 200 and r2.get("tokens") == toks, r2)

st, r3 = call("/detokenize", {"tokens": toks})
check("POST /detokenize", st == 200 and "Hello" in (r3.get("text") or ""), r3)
st, r4 = call("/v1/detokenize", {"input_ids": toks})
check("POST /v1/detokenize", st == 200 and r4.get("text") == r3.get("text"), r4)

st, r5 = call("/encode", {"text": "Hello world"})
check("POST /encode", st == 200 and r5.get("length") == len(r5.get("input_ids", [])), r5)

# count_tokens
st, r6 = call("/v1/messages/count_tokens", {"messages": [{"role": "user", "content": "Hello world"}]})
check("POST /v1/messages/count_tokens", st == 200 and r6.get("input_tokens", 0) > 0, r6)

# Responses API
st, r7 = call("/v1/responses", {"input": "Hello", "max_output_tokens": 8})
ok = (st == 200 and r7.get("object") == "response" and r7.get("status") == "completed"
      and isinstance(r7.get("output"), list) and r7["output"][0]["content"][0]["type"] == "output_text")
check("POST /v1/responses", ok, r7)

# Anthropic Messages（非流式）
st, r8 = call("/v1/messages", {"model": "x", "max_tokens": 8,
                               "messages": [{"role": "user", "content": "Hello"}]})
ok = (st == 200 and r8.get("type") == "message" and r8.get("role") == "assistant"
      and isinstance(r8.get("content"), list) and r8["content"][0]["type"] == "text"
      and r8.get("usage", {}).get("output_tokens", 0) > 0)
check("POST /v1/messages", ok, r8)

# Anthropic Messages（SSE 流式）
st, txt = call("/v1/messages", {"model": "x", "max_tokens": 6, "stream": True,
                                "messages": [{"role": "user", "content": "Hello"}]}, raw=True)
ok = st == 200 and "event: message_start" in txt and "content_block_delta" in txt and "message_stop" in txt
check("POST /v1/messages (stream SSE)", ok, txt[:400])

# Ollama
st, r9 = call("/api/tags")
check("GET /api/tags", st == 200 and len(r9.get("models", [])) >= 1, r9)
st, r10 = call("/api/show", {"model": "x"})
check("POST /api/show", st == 200 and "modelfile" in r10, r10)
st, r11 = call("/api/generate", {"prompt": "Hello", "stream": False})
check("POST /api/generate", st == 200 and r11.get("done") is True and "response" in r11, r11)
st, r12 = call("/api/chat", {"messages": [{"role": "user", "content": "Hello"}]})
check("POST /api/chat", st == 200 and r12.get("message", {}).get("role") == "assistant", r12)

# SageMaker
st, r13 = call("/invocations", {"inputs": "Hello"})
check("POST /invocations", st == 200 and "generated_text" in r13, r13)

# 生命周期
st, r14 = call("/pause_generation", {})
check("POST /pause_generation", st == 200 and r14.get("paused") is True, r14)
st, r15 = call("/slow_down", {"delay_ms": 1})
check("POST /slow_down", st == 200 and r15.get("delay_ms") == 1, r15)
st, r16 = call("/continue_generation", {})
check("POST /continue_generation", st == 200 and r16.get("paused") is False, r16)

st, r17 = call("/start_profile", {})
check("POST /start_profile", st == 200 and r17.get("profiling") is True, r17)
st, r18 = call("/stop_profile", {})
check("POST /stop_profile", st == 200 and r18.get("profiling") is False, r18)

st, r19 = call("/release_memory_occupation", {})
check("POST /release_memory_occupation", st == 200 and r19.get("success") is True, r19)
st, r20 = call("/resume_memory_occupation", {})
check("POST /resume_memory_occupation", st == 200 and r20.get("success") is True, r20)

# 会话
st, r21 = call("/open_session", {"session": "s1"})
check("POST /open_session", st == 200 and r21.get("session_id") == "s1", r21)
st, r22 = call("/close_session", {"session": "s1"})
check("POST /close_session", st == 200 and r22.get("success") is True, r22)

# 运行时 LoRA（父目录有 test-lora 夹具）
st, r23 = call("/load_lora_adapter", {"lora_name": "t1", "lora_path": "../../models/test-lora"})
lora_ok = st == 200 and r23.get("success") is True
if st == 501:
    # 设备后端在 from_llama() 时把 LoRA 合并进权重，运行时增删做不了——501 + 原因是正确行为
    print("SKIP POST /load_lora_adapter <- 501: " + str(r23)[:120])
else:
    check("POST /load_lora_adapter", lora_ok, r23)
if lora_ok:
    st, r24 = call("/generate", {"text": "Hello", "lora": "t1", "sampling_params": {"max_new_tokens": 4}})
    check("lora routing (/generate with lora=t1)", st == 200 and "completion" in r24, r24)
    st, r25 = call("/unload_lora_adapter", {"lora_name": "t1"})
    check("POST /unload_lora_adapter", st == 200 and r25.get("success") is True, r25)

# /model_info
st, r26 = call("/model_info")
check("GET /model_info", st == 200 and r26.get("model_name") == "tiny-llama-hf" and isinstance(r26.get("config"), dict), r26)

# 解析工具
st, r27 = call("/separate_reasoning", {"text": "<think\uff1ehmm<|end_of_thinking|>answer"})
check("POST /separate_reasoning", st == 200 and r27.get("content") == "answer" and r27.get("reasoning_content") == "hmm", r27)
st, r28 = call("/parse_function_call", {"text": "<tool_call>{\"name\":\"f\",\"arguments\":{}}</tool_call>"})
check("POST /parse_function_call", st == 200 and len(r28.get("tool_calls", [])) == 1, r28)

# 明确 501 的（不是假装成功）
for path in ("/v1/classify", "/hicache/storage-backend", "/update_weights_from_tensor",
             "/v1/audio/transcriptions", "/freeze_gc", "/set_trace_level"):
    st, r = call(path, {})
    check("501 " + path, st == 501, (st, r))

# 优先级字段被接受（不报错）
st, r29 = call("/generate", {"text": "hi", "priority": 7, "sampling_params": {"max_new_tokens": 2}})
check("priority accepted", st == 200 and "completion" in r29, r29)

print("\n%d checks failed" % len(fails))
sys.exit(1 if fails else 0)
