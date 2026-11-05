#!/bin/bash
set -e
MLXROOT=~/tools/venv/lib/python3.9/site-packages/mlx
cd ~/v2t
c++ -O2 -std=c++17 -I$MLXROOT/include -I. mlx_ds41_model.cpp -o mlx_ds41_model_dbg \
    -L$MLXROOT/lib -lmlx -Wl,-rpath,$MLXROOT/lib 2>&1 | grep -E "error" | head -10 || true
ls -l mlx_ds41_model_dbg && echo BUILD_OK
DS41_MAXL=3 DS41_MOEDBG=1 SGL_NOENG=1 \
  ./mlx_ds41_model_dbg ./models/dsv41/mlx-dense "9707,11,873,198,44216,31873" 1 \
  2>&1 | grep -aE "shg fast|moe1|hc1|gen ids"
