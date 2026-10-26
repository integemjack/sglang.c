# -*- coding: utf-8 -*-
# G4 探针：CPU 版 sglc --serve 上复现「GET /health 间歇超时」。
# 背景负载 = 并发 POST /generate（驱动 worker/调度器），
# 同时反复 GET /health，分别用 raw socket / requests / curl / Invoke-WebRequest 计时。
# 只读诊断，不改服务。用法见 __main__。
# G17：--port/--host 可传，同一探针既能打 CPU 版 sglc --serve（30000），
#      也能打 GPU 版 cuda_http（历史症状就是 GPU 版报的，之前从没打过它）：
#        python g4_probe.py load 600 12 256 --port 30024
import json, socket, sys, time, threading, subprocess, statistics, os

HOST, PORT = "127.0.0.1", 30000
PY = sys.executable

def parse_net_args(argv):
    """取出 --port/--host（位置任意），其余按位置参数返回（G17）。"""
    global HOST, PORT
    rest = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--port", "-p") and i + 1 < len(argv):
            PORT = int(argv[i + 1]); i += 2; continue
        if a == "--host" and i + 1 < len(argv):
            HOST = argv[i + 1]; i += 2; continue
        rest.append(a); i += 1
    return rest

def http_get_health_raw(timeout_s):
    """raw socket GET /health，返回 (connect_ms, firstbyte_ms, complete_ms, status)。
    status: ok / timeout_connect / timeout_read / reset / empty / err:<msg>"""
    t0 = time.perf_counter()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout_s)
    try:
        s.connect((HOST, PORT))
    except socket.timeout:
        s.close(); return None, None, None, "timeout_connect"
    except OSError as e:
        s.close(); return None, None, None, "err:connect:%s" % e
    t_conn = (time.perf_counter() - t0) * 1000
    try:
        s.sendall(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
        # 首字节
        buf = b""
        t_first = None
        while True:
            k = s.recv(4096)
            if not k:
                break
            if t_first is None:
                t_first = (time.perf_counter() - t0) * 1000
            buf += k
            if b"\r\n\r\n" in buf and (b"}" in buf):
                break
        t_done = (time.perf_counter() - t0) * 1000
        if t_first is None:
            return t_conn, None, t_done, "empty"
        if b"200" in buf and b'"ok"' in buf:
            return t_conn, t_first, t_done, "ok"
        return t_conn, t_first, t_done, "reset" if not buf else "badresp"
    except socket.timeout:
        return t_conn, None, None, "timeout_read"
    except ConnectionResetError:
        return t_conn, None, None, "reset"
    except OSError as e:
        return t_conn, None, None, "err:io:%s" % e
    finally:
        try: s.close()
        except OSError: pass

def post_generate(max_new, timeout_s=120):
    body = json.dumps({"text": "The quick brown fox", "sampling_params": {"max_new_tokens": max_new, "temperature": 0}}).encode()
    req = (b"POST /generate HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
           b"Content-Length: %d\r\nConnection: close\r\n\r\n" % len(body)) + body
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout_s)
    try:
        s.connect((HOST, PORT)); s.sendall(req)
        buf = b""
        while True:
            k = s.recv(65536)
            if not k: break
            buf += k
        return b"200" in buf
    except OSError:
        return False
    finally:
        try: s.close()
        except OSError: pass

stop_flag = threading.Event()
load_stats = {"ok": 0, "fail": 0}
def load_worker(max_new):
    while not stop_flag.is_set():
        if post_generate(max_new): load_stats["ok"] += 1
        else: load_stats["fail"] += 1

def run_probe(n, timeout_s, label):
    res = []
    for i in range(n):
        tc, tf, td, st = http_get_health_raw(timeout_s)
        res.append((st, tc, tf, td))
    ok = sum(1 for r in res if r[0] == "ok")
    to = sum(1 for r in res if r[0] == "timeout_read")
    toc = sum(1 for r in res if r[0] == "timeout_connect")
    rst = sum(1 for r in res if r[0] == "reset")
    bad = sum(1 for r in res if r[0] not in ("ok","timeout_read","timeout_connect","reset"))
    comp = [r[3] for r in res if r[0] == "ok" and r[3] is not None]
    fb   = [r[2] for r in res if r[0] == "ok" and r[2] is not None]
    print("== %s ==" % label)
    print("  n=%d ok=%d timeout_read=%d timeout_connect=%d reset=%d other=%d" % (n, ok, to, toc, rst, bad))
    if comp:
        print("  complete_ms: min=%.1f med=%.1f p95=%.1f max=%.1f" % (
            min(comp), statistics.median(comp), sorted(comp)[int(len(comp)*0.95)-1], max(comp)))
    if fb:
        print("  firstbyte_ms: min=%.1f med=%.1f max=%.1f" % (min(fb), statistics.median(fb), max(fb)))
    # 打印非 ok 的样本
    anomalies = [r for r in res if r[0] != "ok"]
    for a in anomalies[:10]:
        print("   ANOMALY status=%s conn=%s first=%s done=%s" % (a[0],
            ("%.1f"%a[1]) if a[1] else "-", ("%.1f"%a[2]) if a[2] else "-", ("%.1f"%a[3]) if a[3] else "-"))
    return res

def probe_requests(n, timeout_s, label):
    import requests
    lat = []; to = 0; err = 0
    for i in range(n):
        t0 = time.perf_counter()
        try:
            r = requests.get("http://%s:%d/health" % (HOST, PORT), timeout=timeout_s)
            lat.append((time.perf_counter()-t0)*1000)
        except requests.exceptions.Timeout:
            to += 1
        except Exception:
            err += 1
    print("== %s (requests) == n=%d ok=%d timeout=%d err=%d" % (label, n, len(lat), to, err))
    if lat:
        print("  ms: min=%.1f med=%.1f max=%.1f" % (min(lat), statistics.median(lat), max(lat)))

def probe_curl(n, timeout_s, label):
    lat = []; to = 0; err = 0
    for i in range(n):
        t0 = time.perf_counter()
        p = subprocess.run(["curl.exe", "-s", "-o", "NUL", "-w", "%{http_code}",
                            "--max-time", str(timeout_s), "http://%s:%d/health" % (HOST, PORT)],
                           capture_output=True, text=True)
        dt = (time.perf_counter()-t0)*1000
        if p.stdout.strip() == "200": lat.append(dt)
        elif p.returncode == 28: to += 1
        else: err += 1
    print("== %s (curl) == n=%d ok=%d timeout=%d err=%d" % (label, n, len(lat), to, err))
    if lat:
        print("  ms: min=%.1f med=%.1f max=%.1f" % (min(lat), statistics.median(lat), max(lat)))

if __name__ == "__main__":
    argv = parse_net_args(sys.argv[1:])
    mode = argv[0] if len(argv) > 0 else "baseline"
    print("target: %s:%d mode=%s" % (HOST, PORT, mode))
    if mode == "baseline":
        # 无负载：raw + requests + curl 各测一轮，确认基线正常
        run_probe(200, 10.0, "baseline raw GET /health (no load)")
        probe_requests(50, 10.0, "baseline")
        probe_curl(50, 10.0, "baseline")
    elif mode == "load":
        n = int(argv[1]) if len(argv) > 1 else 300
        nl = int(argv[2]) if len(argv) > 2 else 8
        mn = int(argv[3]) if len(argv) > 3 else 128
        threads = [threading.Thread(target=load_worker, args=(mn,), daemon=True) for _ in range(nl)]
        for t in threads: t.start()
        time.sleep(2.0)  # 让负载起来
        try:
            run_probe(n, 10.0, "loaded raw GET /health (%d load threads, max_new=%d)" % (nl, mn))
            probe_requests(60, 10.0, "loaded")
            probe_curl(60, 10.0, "loaded")
        finally:
            stop_flag.set()
            time.sleep(0.5)
            print("  load POST /generate: ok=%d fail=%d" % (load_stats["ok"], load_stats["fail"]))
    elif mode == "long":
        # 长超时：区分「慢（最终完成）」还是「永久卡死」
        n = int(argv[1]) if len(argv) > 1 else 40
        nl = int(argv[2]) if len(argv) > 2 else 8
        mn = int(argv[3]) if len(argv) > 3 else 128
        to = float(argv[4]) if len(argv) > 4 else 60.0
        threads = [threading.Thread(target=load_worker, args=(mn,), daemon=True) for _ in range(nl)]
        for t in threads: t.start()
        time.sleep(8.0)  # 预热：让负载线程完成若干轮，worker 进入稳定循环
        print("  warmup done, load so far: ok=%d fail=%d" % (load_stats["ok"], load_stats["fail"]))
        slow = []
        try:
            for i in range(n):
                tc, tf, td, st = http_get_health_raw(to)
                line = "  [%2d] status=%-14s conn=%s first=%s done=%s" % (i, st,
                    ("%7.1f"%tc) if tc else "      -", ("%7.1f"%tf) if tf else "      -",
                    ("%8.1f"%td) if td else "       -")
                print(line)
                if st == "ok" and td and td > 1000: slow.append(td)
                if st == "timeout_read": slow.append(">%.0fs" % to)
            print("  slow/timeout samples (>1s or never):", slow)
        finally:
            stop_flag.set(); time.sleep(0.5)
            print("  load POST /generate: ok=%d fail=%d" % (load_stats["ok"], load_stats["fail"]))
    elif mode == "sust":
        # 先让负载真正饱和（长预热），再串行探测 health：确定性触发
        n = int(argv[1]) if len(argv) > 1 else 100
        nl = int(argv[2]) if len(argv) > 2 else 8
        mn = int(argv[3]) if len(argv) > 3 else 128
        to = float(argv[4]) if len(argv) > 4 else 10.0
        warm = float(argv[5]) if len(argv) > 5 else 30.0
        threads = [threading.Thread(target=load_worker, args=(mn,), daemon=True) for _ in range(nl)]
        for t in threads: t.start()
        print("  warming up %.0fs to saturate worker..." % warm)
        time.sleep(warm)
        print("  warmup: load ok=%d fail=%d" % (load_stats["ok"], load_stats["fail"]))
        try:
            run_probe(n, to, "sustained-load raw GET /health (%d load, max_new=%d, to=%.0fs)" % (nl,mn,to))
        finally:
            stop_flag.set(); time.sleep(0.5)
            print("  load POST /generate: ok=%d fail=%d" % (load_stats["ok"], load_stats["fail"]))
    elif mode == "conc":
        # 并发 health：看 connect() 延迟是否随 backlog 上升（区分 acceptor 饿死 vs send 阻塞）
        dur = float(argv[1]) if len(argv) > 1 else 30.0
        nl = int(argv[2]) if len(argv) > 2 else 12
        mn = int(argv[3]) if len(argv) > 3 else 64
        hc = int(argv[4]) if len(argv) > 4 else 16
        lt = [threading.Thread(target=load_worker, args=(mn,), daemon=True) for _ in range(nl)]
        for t in lt: t.start()
        time.sleep(6.0)
        print("  warmup: load ok=%d fail=%d" % (load_stats["ok"], load_stats["fail"]))
        stop_probe = threading.Event()
        conns=[]; firsts=[]; oks=[0]; tos=[0]; rst=[0]
        def hp():
            while not stop_probe.is_set():
                tc, tf, td, st = http_get_health_raw(20.0)
                if tc is not None: conns.append(tc)
                if st=="ok":
                    oks[0]+=1
                    if tf is not None: firsts.append(tf)
                elif st=="timeout_read": tos[0]+=1
                elif st=="reset": rst[0]+=1
        pr=[threading.Thread(target=hp, daemon=True) for _ in range(hc)]
        t_start=time.time()
        for t in pr: t.start()
        time.sleep(dur)
        stop_probe.set(); time.sleep(1.0)
        stop_flag.set()
        def q(a,p):
            a=sorted(a); return a[min(len(a)-1,int(len(a)*p))] if a else 0
        print("== concurrent health under load (%.0fs, %d probers, %d load) ==" % (dur,hc,nl))
        print("  ok=%d timeout=%d reset=%d" % (oks[0],tos[0],rst[0]))
        print("  connect_ms  med=%.1f p95=%.1f max=%.1f  (n=%d)" % (q(conns,.5),q(conns,.95),max(conns) if conns else 0,len(conns)))
        print("  firstbyte_ms med=%.1f p95=%.1f max=%.1f  (n=%d)" % (q(firsts,.5),q(firsts,.95),max(firsts) if firsts else 0,len(firsts)))
        print("  load POST /generate: ok=%d fail=%d" % (load_stats["ok"], load_stats["fail"]))
