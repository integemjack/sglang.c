#!/bin/bash
# 远端构建/测试助手（在 AutoDL 实例上跑）。本地用 ssh 调它，省得每次拼长命令行。
#   ./rbuild.sh            配置 + 编译（CUDA, sm_120）+ ctest
#   ./rbuild.sh build      只编译
#   ./rbuild.sh test       只跑 ctest
#   ./rbuild.sh run <bin>  [args...]   跑 build/bin 下的某个程序
set -u
cd /root/sglang-c || exit 1
export PATH=/usr/local/cuda/bin:$PATH
JOBS=${JOBS:-32}

case "${1:-all}" in
  all)
    cmake -S . -B build -DSGL_CUDA=ON -DSGL_CUDA_ARCH=120 -DCMAKE_BUILD_TYPE=Release > /tmp/cm.log 2>&1 || { tail -20 /tmp/cm.log; exit 1; }
    make -C build -j "$JOBS" > /tmp/build.log 2>&1 || { grep -E 'error:|undefined reference' /tmp/build.log | head -20; exit 1; }
    echo "build ok"
    ctest --test-dir build --output-on-failure 2>&1 | tail -12
    ;;
  build)
    make -C build -j "$JOBS" > /tmp/build.log 2>&1 || { grep -E 'error:|undefined reference' /tmp/build.log | head -20; exit 1; }
    echo "build ok"
    ;;
  test)
    ctest --test-dir build --output-on-failure 2>&1 | tail -15
    ;;
  run)
    shift
    ./build/bin/"$@"
    ;;
esac