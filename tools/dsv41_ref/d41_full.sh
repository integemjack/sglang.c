#!/bin/bash
# ds41 full run（新权重 v3），40 层 × 8 步，无 MAXL
cd ~/v2t
SGL_NOENG=1 ./mlx_ds41_model_dbg ./models/dsv41/mlx-dense "9707,11,873,198,44216,31873" 8 > /tmp/d41_v3_full.log 2>&1
echo "exit=$?" >> /tmp/d41_v3_full.log
