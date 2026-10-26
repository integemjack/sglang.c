# NOTES_t5 — GPU HTTP 对等 + 稳定性修复

日期：2026-10-27 ｜ 机器：RTX 2060 6GB，CUDA 13.3，qwen2.5-0.5B

## 验收结果（tests/stream_test.ps1 + build-cuda/conc_test.ps1）

- 流式 == 非流式：PASS（chat 同 prompt 同参数，SSE 拼接 58 == plain 58 chars）
- 并发 12（> 8 段）排队：PASS（12/12 完成，不再 503）
- finish_reason：PASS（max_tokens 达到 → "length"；eos → "stop"）
- conc_test 8 并发多轮：全过

## 本步修复

1. **排队**：submit 超出 ACTIVE_CONC 不再 503，挂 FIFO waitq，worker 每轮 drain 到空 slot；
   上限 128 防积压。finish_reason 如实上报（max_tokens/seg 满 → length）。
2. **SSE 快请求丢增量**：请求在 wait_done 进入前就 DONE → 只发 finish chunk。
   DONE 分支补发全部未流式增量。
3. **CUDA graph 与缓冲 realloc 的 use-after-free**（重大）：captured graph 的 memcpy
   节点绑定 pinned host 地址（h_tok/h_pq...）、kernel 节点绑定设备指针（x/h/q/...）。
   `sgl_cuda_reserve` / `sgl_cuda_ensure_io` 增长时会释放这些地址——旧 graph replay
   读悬空指针 → sticky CUDA error。修复：两处 realloc 前 `invalidate_graphs()`
   全部作废（lazy re-capture）。
4. **http 服务默认 SGL_CUDA_GRAPH=0**：动态 B（1..8 交错变化）+ 多次 buffer 增长下
   graph replay 仍间歇报非法访问（invalidate 修复后仍有，根因未完全定位）。
   生成路径（plain decode）验证 100% 稳定；cuda_test 单场景 graph 收益验收不受影响
   （B=1 +174%、B=4 +87% 的数字保留）。显式设 SGL_CUDA_GRAPH=1 可开。
5. **accept 循环被瞬时错误杀死**：WSAECONNABORTED（并发洪峰下 aborted handshake）
   曾直接 break 出 accept 循环 → 服务「活着但不再接客」。改为 continue。
6. **send 分段间歇失败**：respond 的 head/body 两次 send 之间间歇 WOULDBLOCK/卡住
   （抓到过 sent 90+(-1)），客户端拿到声明 Content-Length 却等不到 body。
   修复：合并为单次缓冲写 + send_all 重试 + SO_SNDTIMEO/TCP_NODELAY/SO_RCVTIMEO。

## 已知问题（未根治）

**GET /health 间歇超时**（重负载后）：conn_thread 的 recv 正常收到完整请求、
read_request/路由正常，respond 的 send 间歇性永久不返回（线程 EventPairLow 等待，
SO_SNDTIMEO 也不触发；curl --retry 无效；POST /generate 同一时刻 100% 正常；
进程不崩，重启后恢复）。排除项：accept 循环、read_request 解析、GPU/CUDA 错误、
句柄/线程泄漏（稳定 202 handles）、CPU 自旋。疑似 Winsock×多线程×CUDA 运行时
在本机的交互问题，复现路径与探针输出已记录在上文。运维建议：监控用
POST /generate 小请求 + 客户端重试。

## 铁律遵守情况

只改 apps/cuda_http.c、src/cuda_backend.cu（invalidate_graphs + reserve/ensure_io
两处调用）、tests/stream_test.ps1；全部修改经 conc_test×2 + stream_test 全套验证。
