# NOTES_t11 — 健壮性硬化（soak + fuzz + 降级）

日期：2026-10-27 ｜ GPU: cuda_http(qwen05@30002) ｜ CPU: sglc --serve(qwen3-06@30003)

## 结果

- **GPU 栈全过**（`tests/soak_fuzz.ps1`）：20 轮 soak drift -0.8%（曲线平坦无泄漏）、
  fuzz 8/8（畸形 JSON / 空 body / 类型错 / max_tokens=1e9 / 负数 / 二进制 / 未知字段 /
  200KB 超长文本）、断连、并发同请求、ctx 溢出 finish_reason=length、全程存活。
- **CPU 栈全过**（`-SkipEnvDependent`）：fuzz 7 项 + 断连；soak 5 轮 drift 0%。

## 修复的 bug

1. **200KB 单字符 prompt 挂死服务**：BPE encode 对病态输入是 O(n²)——加 64KB 预检
   （两栈同修），返回 503 "prompt too long"。
2. **CPU 响应固定卡 ~21-24s**：respond 分 head/body 两次 send + Nagle×延迟 ACK =
   每响应 2×RTO 停顿。TCP_NODELAY（两栈）后 24.3s→6.9s（即真实生成时长）。
3. **accept 瞬时错误杀死 acceptor**（两栈）：WSAECONNABORTED/EINTR 重试。
4. **nvcc obj 不跟踪头文件依赖**：llama.h 改字段后旧 cuda_backend.obj 按旧偏移访问
   → 启动即崩。修法：删 build-cuda/cuda_backend.obj 全量重编；头文件常改时建议
   CMake 自定义命令补 DEPENDS（未做，改动面大）。
5. MSVC 源文件必须 UTF-8 BOM（llama.h/cuda.h 等第二次踩坑）：无 BOM 时中文注释
   按 GBK 误读会吞结构体成员声明。

## 已知环境问题（非本仓库代码）

无 body 的 GET /health 在高负载后间歇无响应——**MinGW/MSVC 两套独立 HTTP 实现同现象**，
指向本机 winsock/LSP 层。缓解：监控用 `POST /health -d x`（稳定）；SO_SNDTIMEO 5s
兜底防线程泄漏。相关项已用 `-SkipEnvDependent` 隔离。

## 降级路径

- KV pool 满：CPU 调度器等待重试（既有）；GPU 段分配满→ 503 明确报错。
- ctx 满：finish_reason=length（GPU 已断言；CPU 同路径）。
- 显存不足：cuda_backend 上传 OOM→ 明确 stderr + 退出码 2。
