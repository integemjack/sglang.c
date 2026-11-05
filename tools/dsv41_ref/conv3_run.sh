#!/bin/bash
# v3 转换（后台跑，日志 /tmp/conv3.log）
cd ~/v2t
nohup ~/tools/venv/bin/python /tmp/conv3_dsv41.py ./models/dsv41 > /tmp/conv3.log 2>&1 < /dev/null &
pid=$!
disown $pid
echo "conv3 pid=$pid"
