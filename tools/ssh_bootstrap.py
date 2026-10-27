"""SSH 帮助脚本：用密码连一次，装本机公钥，之后免密。

用法：
  python ssh_bootstrap.py                 # 装公钥 + 打印远端环境
密码从环境变量 SEETACLOUD_SSH_PASSWORD 读（Friday 在执行时换成真值）。
"""
import os
import sys

import paramiko

HOST = "connect.bjb2.seetacloud.com"
PORT = 46450
USER = "root"
PW = os.environ.get("SEETACLOUD_SSH_PASSWORD", "")
PUB = os.path.join(os.path.expanduser("~"), ".ssh", "id_ed25519.pub")

ENV_PROBE = r"""
echo "== host =="; hostname; uname -a
echo "== cpu/mem =="; nproc; free -g | head -2
echo "== gpu =="; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "== cuda =="; which nvcc && nvcc --version | tail -2
echo "== toolchain =="; which gcc && gcc --version | head -1; which cmake && cmake --version | head -1
which tmux; which rsync; which git && git --version
echo "== disk =="; df -h / /root | tail -3
echo "== python =="; which python3; python3 -V
echo "== os =="; cat /etc/os-release | head -2
"""


def main():
    if not PW:
        raise SystemExit("缺 SEETACLOUD_SSH_PASSWORD")
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PW, timeout=30,
              allow_agent=False, look_for_keys=False)
    print("密码登录成功")

    def run(cmd):
        _, out, err = c.exec_command(cmd, timeout=120)
        o = out.read().decode("utf-8", "replace")
        e = err.read().decode("utf-8", "replace")
        return o + (("\n[stderr] " + e) if e.strip() else "")

    # 装公钥
    key = open(PUB).read().strip()
    run("mkdir -p ~/.ssh && chmod 700 ~/.ssh")
    run("grep -qF '%s' ~/.ssh/authorized_keys 2>/dev/null || echo '%s' >> ~/.ssh/authorized_keys"
        % (key, key))
    run("chmod 600 ~/.ssh/authorized_keys")
    print("公钥已装：", run("grep -c '' ~/.ssh/authorized_keys").strip(), "条")
    print(run(ENV_PROBE))
    c.close()


if __name__ == "__main__":
    main()