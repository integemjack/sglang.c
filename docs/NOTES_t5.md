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

## G4 复现与定位（2026-10-27）

**结论：该现象在纯 CPU 版（sglc.exe --serve，MinGW 构建，不链接任何 CUDA 运行时）上能复现，
因此「疑似 CUDA 运行时交互」的假设不成立——根因是 HTTP 服务的线程/锁设计，与显卡无关。**

### 代码路径差异（无 body 请求 vs 有 body 请求）

- 线程模型：两版都是「每连接一线程 + 一个 worker 驱动调度循环」。
  - CPU：src/http.c `acceptor_fn`(630) → `pthread_create(conn_fn)`(653)；`worker_fn`(233)。
  - GPU：apps/cuda_http.c `main` 的 accept 循环(860) → `CreateThread(conn_thread)`(873)；`worker`(199)。
- 阻塞性：连接 socket 全程**阻塞**（两版都没有 ioctlsocket(FIONBIO)/O_NONBLOCK）。
- 超时：CPU `conn_fn` 设 SO_RCVTIMEO=30000、SO_SNDTIMEO=30000（http.c:615-616）；
  GPU `conn_thread` 设 SO_RCVTIMEO=30000、SO_SNDTIMEO=5000（cuda_http.c:752-754）。
- GET /health（无 body）路径：`read_request` 收到 `\r\n\r\n` 即返回（CPU http.c:144-204，
  clen=0 时「读满 body」循环(189)不进入；GPU cuda_http.c:357-392 同理），随后 `route` 直接
  `respond`（CPU http.c:310-312；GPU cuda_http.c:610-613），**不碰调度器锁、不等 worker**。
  有 body 的 POST /generate 才会 `submit` + `wait_done`（等 worker 的 cv_done/cv_step）。
- `respond`→`send_all`：CPU 一次性写、`send` 返回 <=0 就放弃、不重试（http.c:106-112），
  阻塞 send 受 SO_SNDTIMEO=30s 兜底；GPU 有 WOULDBLOCK/NOBUFS 重试循环、上限 ~2s（cuda_http.c:410-425）。
  **两版的 send 都不可能「永久不返回」**：CPU 最坏 30s 后 send 失败返回，GPU 重试循环最坏 ~2s 返回 -1。
- `EventPairLow`：是 winpthreads 条件变量/信号量的底层事件名。CPU 的 worker/acceptor/conn 用
  pthread_cond（http.c:239,260）；GPU 用 Win32 CONDITION_VARIABLE（cuda_http.c:586）。
  栈里出现 EventPairLow 说明线程阻塞在「条件变量等待」，而非 send 系统调用本身。

### 复现结果（探针 tests/g4_probe.py，目标端口 127.0.0.1:30000，模型 models/gpt2）

- 基线（无负载）：raw 200/200、requests 50/50、curl 50/50 全 ok，med≈16ms。**无异常。**
- **复现成功一次**：`load 300 8 128`（8 个并发 POST /generate 各 128 token + 串行 GET /health）
  → 300 次里 **20 次 timeout_read（6.7%）**：TCP connect≈15ms 正常，但 10s 内收不到响应首字节；
  ok 的长尾 p95=3.2s、max=9.5s。与验收记录签名一致（连接建立、请求已发、响应迟迟不来）。
  该轮 `sglc.exe` 为纯 CPU 构建，进程内无 CUDA 运行时。
- 复现**不可靠**：随后 5 轮（baseline / conc 并发 / long×2 / load / sust 30s 饱和预热）在
  机器 CPU 100% 时也 **0 超时**（max firstbyte 26.5ms）。唯一成功的那轮，机器上另有**两个
  curl 大文件下载**在抢占 CPU+磁盘+网络（已确认是其它任务），形成系统级资源风暴。
- 卡住瞬间取证：`Get-NetTCPConnection -OwningProcess <pid>` 显示无 SYN 堆积、无 CloseWait 累积、
  established 连接正常轮转；`SendQueueBytes` 本机不可用（恒空）。即**没有「send 阻塞、发送队列满」的
  TCP 层证据**，指向「服务端线程尚未走到 send」。

### 定位

排除项（有证据）：CUDA 运行时交互（CPU 版复现）、send() 永久阻塞（send_all 有 30s/2s 上界）、
TCP 发送队列堆积（快照无堆积）、accept 循环崩溃（accept 失败已 continue）、句柄/线程泄漏（负载下
线程数 26→54 可回落）。

**结构性根因（CPU 版，http.c）：一把粗粒度 `srv->lock` 同时被两处热路径争用——**
1. worker 每跑一步 `sgl_sched_step` 都 `pthread_mutex_lock(&srv->lock)`(236) 并在步末解锁(248)；
   `sgl_sched_step` 每次只做一步 decode-batch/prefill 即返回（scheduler.c:217-258），单步不长，
   但在有活时 worker 几乎不间断地 lock→step→unlock→lock…（238 的 cond_wait 仅在 nactive==0 时进入）。
2. acceptor 每接受一个新连接，都要 `pthread_mutex_lock(&srv->lock)`(647) 去自增 `active_conns`，
   然后才 `pthread_create(conn_fn)`(653)。

高并发/系统资源紧张时，acceptor 反复在 (647) 抢锁输给 worker，新连接虽已被内核完成握手
（客户端 connect() 秒回），却长时间得不到 accept→登记→建线程→recv→respond，表现为
「连接已建立但首字节迟迟不来」；一旦 worker 步间空窗被 acceptor 抢到，请求随即被处理。
这与「间歇、重负载后、POST /generate 同时刻正常（它本就在跑）、重启恢复（清空积压）」全部吻合。
GPU 版 accept 循环不取 E.lock（cuda_http.c:860-875），故其卡死若同签名，应另有触发路径；
但「是否 CUDA」这一关键问题已由 CPU 复现否定。

### 建议修法（未动手，供落地）

1. **拆分锁**：给 `active_conns` 单独一把 `conns_lock`（或 Win 下 `InterlockedIncrement`），
   让 acceptor 不再排在 worker 的计算锁后面。最小改动：
   ```c
   // struct sgl_http_server 增加： pthread_mutex_t conns_lock;
   // sgl_http_serve 里 pthread_mutex_init(&srv->conns_lock, NULL);
   // acceptor_fn:647-649 把 srv->lock 换成 srv->conns_lock：
   //   pthread_mutex_lock(&srv->conns_lock); srv->active_conns++; pthread_mutex_unlock(&srv->conns_lock);
   // conn_fn:623-625 的 --active_conns 同样改用 conns_lock（cv_conns_zero 的等待逻辑相应挪到 conns_lock）。
   ```
2. CPU `send_all`（http.c:106-112）对齐 GPU：对 WSAEWOULDBLOCK/WSAENOBUFS 短重试、别在首个
   可恢复错误上静默截断响应。
3. 运维旁证仍成立：监控用带 body 的小请求（走不同路径、不触发 accept 登记争用）+ 客户端重试。

探针脚本：tests/g4_probe.py（baseline / load / long / conc / sust 五种模式）。
