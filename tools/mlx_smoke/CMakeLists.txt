cmake_minimum_required(VERSION 3.27)
project(sgl_mlx_smoke LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 定位 MLX：优先 -DMLX_ROOT=...，否则问 python（pip 的 mlx wheel 自带 C++ 包与 CMake config）
if(NOT DEFINED MLX_ROOT OR MLX_ROOT STREQUAL "")
  find_package(Python 3.9 COMPONENTS Interpreter REQUIRED)
  execute_process(COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
                  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE MLX_ROOT)
endif()
message(STATUS "MLX_ROOT = ${MLX_ROOT}")
list(APPEND CMAKE_PREFIX_PATH "${MLX_ROOT}")
find_package(MLX CONFIG REQUIRED)
message(STATUS "MLX_FOUND=${MLX_FOUND} INCLUDE=${MLX_INCLUDE_DIRS} LIBS=${MLX_LIBRARIES}")

# libmlx.dylib 的 install name 是 @rpath/libmlx.dylib → 必须把 wheel 的 lib 目录写进 rpath，
# 且 mlx.metallib 要与 dylib 同目录（wheel 里本来就是）。
# 注意：MLX_LIBRARIES 只是目标名 "mlx"（不是路径），要从 imported target 上取真实文件位置。
get_target_property(SGL_MLX_LIB mlx IMPORTED_LOCATION)
if(NOT SGL_MLX_LIB)
  get_target_property(SGL_MLX_LIB mlx IMPORTED_LOCATION_RELEASE)
endif()
get_filename_component(SGL_MLX_LIBDIR "${SGL_MLX_LIB}" DIRECTORY)
message(STATUS "MLX libdir (rpath) = ${SGL_MLX_LIBDIR}")

# 1) smoke：算子级可行性（rms_norm / rope / sdpa / fp16 / 内存预算）
add_executable(sgl_mlx_smoke smoke.cpp)
target_link_libraries(sgl_mlx_smoke PRIVATE mlx)
set_target_properties(sgl_mlx_smoke PROPERTIES
  BUILD_RPATH "${SGL_MLX_LIBDIR}" INSTALL_RPATH "${SGL_MLX_LIBDIR}")

# 2) probe2：写 mlx_backend.cpp 前钉死 7 个 API 语义（rope 轴/freqs/scale/数组 offset、
#    array 构造是否拷贝、slice_update、softmax 轴、4D matmul、take）
add_executable(sgl_mlx_probe2 probe2.cpp)
target_link_libraries(sgl_mlx_probe2 PRIVATE mlx)
set_target_properties(sgl_mlx_probe2 PROPERTIES
  BUILD_RPATH "${SGL_MLX_LIBDIR}" INSTALL_RPATH "${SGL_MLX_LIBDIR}")
