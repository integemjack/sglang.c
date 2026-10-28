#!/bin/bash
# 把本地 sglang-c 的工作树（git 跟踪的文件 + 指定未跟踪文件）同步到远端并编译。
# 本地用法（PowerShell 里通过 ssh 调远端那个版本没用，这个脚本在**本地**跑，
# 在 git bash 或 WSL 下执行；Windows 下我通常直接 tar+scp，见 tools/rsync.sh 的说明）。
set -eu
cd "$(dirname "$0")/.."
tar -czf /tmp/sync.tgz --exclude=build --exclude=build-msvc \
    --exclude=.git --exclude=models --exclude='*.obj' .
scp -P "${SGL_SSH_PORT:-46450}" -o BatchMode=yes /tmp/sync.tgz \
    root@connect.bjb2.seetacloud.com:/root/sync.tgz
ssh -p "${SGL_SSH_PORT:-46450}" -o BatchMode=yes root@connect.bjb2.seetacloud.com \
    "cd /root/sglang-c && tar xzf /root/sync.tgz && rm /root/sync.tgz && \
     find . -name '*.sh' -exec sed -i 's/\r$//' {} \; && \
     bash tools/rbuild.sh ${1:-build}"
rm -f /tmp/sync.tgz