#!/bin/bash
# 把 /root/staging 下的模型目录合并进 /root/sglang-c/models（已存在的目录按文件覆盖）。
# 用法：./merge_models.sh [staging_dir]
set -eu
STAGE="${1:-/root/staging}"
DEST=/root/sglang-c/models
mkdir -p "$DEST"
for f in "$STAGE"/*; do
  [ -e "$f" ] || continue
  name=$(basename "$f")
  if [ -d "$DEST/$name" ]; then
    cp -rf "$f"/. "$DEST/$name"/
  else
    mv "$f" "$DEST"/
  fi
done
rmdir "$STAGE" 2>/dev/null || true
du -sh "$DEST"
ls "$DEST"