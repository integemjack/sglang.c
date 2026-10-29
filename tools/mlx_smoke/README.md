# tools/mlx_smoke — MLX 后端可行性冒烟（W2 骨架验证）

一个**独立**的小程序，不属于 `sgl` 库、不碰仓库任何现有代码，只用来回答一个问题：
「在这台 Apple Silicon 上，能不能用 pip 装的 MLX 直接编出一个跑在 Metal GPU 上的 C++ 程序？」

已在 iMac（Apple M1 / 16 GB / macOS 27.0.1 / MLX 0.29.3）上跑通，输出 `SMOKE_PASS`。
背景与后续计划见 [`docs/PLAN_mlx_backend.md`](../../docs/PLAN_mlx_backend.md)。

## 它验证了什么

| 验证项 | 结果（2026-10-29 实测） |
|---|---|
| `find_package(MLX CONFIG)` 能吃到 pip wheel | ✅ `Found MLX: .../site-packages/mlx/lib/libmlx.dylib`，`MLX_BUILD_METAL=ON` |
| C++20 + AppleClang 21 能编 MLX 头 | ✅ |
| `@rpath/libmlx.dylib` + `mlx.metallib` 能加载 | ✅ `otool -L` 显示 `@rpath/libmlx.dylib`，`LC_RPATH` 指向 wheel 的 `lib/` |
| Metal GPU 真在算（不是回落 CPU） | ✅ `metal_avail=1`，`device_name=Apple M1`，`architecture=applegpu_g13g` |
| `mx::fast::rms_norm` 精度 | ✅ maxerr **1.19e-07** vs 主机参考（[4,64] fp32） |
| `mx::fast::rope` 的**数组 offset 重载** | ✅ 与逐行标量 offset **完全相等（maxerr = 0）** —— batch decode 的逐行位置可直接用 |
| `mx::fast::scaled_dot_product_attention(causal)` | ✅ maxerr **2.98e-08** vs 主机朴素实现（B3 H2 S6 D8） |
| fp16 路径 | ✅ `astype(float16)` 后 rms_norm 跑通，无 NaN/Inf |
| 内存预算 API | ✅ `max_recommended_working_set_size = 12713115648`（11.84 GB）、`max_buffer_length = 8.88 GB` |

## 怎么跑

```bash
# 前置：pip install mlx（wheel 自带 C++ 库/头/CMake config，不需要从源码编 MLX）
PY=/path/to/venv/bin/python3          # 装了 mlx 的那个解释器
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$PY"
cmake --build build
./build/sgl_mlx_smoke                 # 最后一行应为 SMOKE_PASS，退出码 0
```

也可以直接指定 MLX 根目录，跳过 python 探测：

```bash
cmake -B build -G Ninja -DMLX_ROOT="$($PY -m mlx --cmake-dir)"
```

## 两个已踩到的坑（写进 CMakeLists 的注释里了）

1. **`MLX_LIBRARIES` 是目标名 `mlx`，不是路径**。要拿 rpath 目录必须
   `get_target_property(... mlx IMPORTED_LOCATION)` 再 `get_filename_component(... DIRECTORY)`；
   直接对 `MLX_LIBRARIES` 取 dirname 会得到空串，链接能过但运行时找不到 dylib。
2. **`libmlx.dylib` 的 install name 是 `@rpath/libmlx.dylib`**，所以必须设 `BUILD_RPATH`/
   `INSTALL_RPATH` 指向 wheel 的 `lib/`；`mlx.metallib`（105 MB 预编译 Metal kernel）
   要与 dylib 同目录 —— wheel 里本来就是，**打包时别只拷 dylib**。

## 从 Windows 往 macOS 传源码的坑（与 MLX 无关，但会咬人）

不要用 PowerShell 的管道传文件（`Get-Content -Raw | ssh host "cat > f"`）：
PS 5.1 会把 UTF-8 中文按 ANSI 解码再按 UTF-8 编码，**双重编码后的字节可能吞掉紧随的换行**，
表现为「CMake 报 `Flow control statements are not properly nested`」——因为 `if(` 那行被并进了
上一行的中文注释里。用 `scp` 传，传完 `md5` 对一下。
远端路径含空格（如 `/Volumes/9100 PRO`）时，Windows 的 `scp` 转义不好使，
先 `ln -s "/Volumes/9100 PRO" ~/v2t` 再用无空格的相对路径。
