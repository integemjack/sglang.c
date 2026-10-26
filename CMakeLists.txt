cmake_minimum_required(VERSION 3.20)
project(sglang_c VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ---- 全局编译选项（仅 GNU/Clang；MSVC/nvcc 不接受这些旗标）----
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang" AND NOT MSVC)
  add_compile_options(-Wall -Wextra -Wno-unused-parameter)
endif()
if(MSVC)
  # 源码是 UTF-8（含中文注释）；MSVC 默认按系统代码页(936) 误读会吃掉注释/字符串
  add_compile_options(/utf-8)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

# ---- 库 ----（先建目标，再挂选项）
file(GLOB_RECURSE SGL_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c)
if(MSVC)
  # http.c 依赖 winpthreads（MinGW 专属）；MSVC 构建排除，sglc --serve 同步禁用
  list(REMOVE_ITEM SGL_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/http.c)
endif()
add_library(sgl STATIC ${SGL_SOURCES})
target_include_directories(sgl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# OpenMP：不用 find_package 的导入目标（它会写死 libgomp.dll.a 绝对路径，
# 绕过 -static），改由 gcc 驱动链接，-static 下自动选静态 libgomp。
# 仅 GNU 编译器启用：Apple Clang 无 -fopenmp（需 brew libomp，CI 不引入）；
# Linux Clang 同理可选，统一走单线程 fallback。
if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
  target_compile_options(sgl PUBLIC -fopenmp)
  target_link_options(sgl INTERFACE -fopenmp)
  message(STATUS "OpenMP: enabled (compiler-driven)")
else()
  message(STATUS "OpenMP not available with ${CMAKE_C_COMPILER_ID}, single-threaded fallback")
endif()

if(WIN32)
  target_link_libraries(sgl PUBLIC ws2_32) # http.c 的 winsock
  target_link_options(sgl INTERFACE -static) # 生成不依赖 MinGW 运行库 DLL 的独立 exe
else()
  # POSIX：math 函数在独立 libm（Windows 在主 CRT）
  target_link_libraries(sgl PUBLIC m)
endif()

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  # NOTE: WinLibs GCC 16.2 (ucrt) 在本机 (i7-10700F) 上启用 AVX2 (-march=native/-mavx2)
  # 会间歇性 0xC0000005（最小复现：纯栈代码按值传 80B 结构体 + printf，无堆内存）。
  # 默认 SSE2 与 -march=x86-64-v2 实测 25/25 稳定。等工具链修复后再启用 AVX2。
  # （t12.2 之后：AVX2 已由 ops.c 的手写 intrinsics + target attribute 按 DOT 单点启用，
  #  不依赖全局 -mavx2，运行时 __builtin_cpu_supports 检测、QEMU 自动回退 SSE2。）
  # x86-64-v2 仅限 x86_64；arm64 (Apple Silicon / linux-arm64 CI) 不加架构旗标。
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64")
    target_compile_options(sgl PUBLIC -O2 -funroll-loops)
  else()
    target_compile_options(sgl PUBLIC -O2 -march=x86-64-v2 -funroll-loops)
  endif()
  if(WIN32)
    target_compile_definitions(sgl PUBLIC __USE_MINGW_ANSI_STDIO=1) # %zu 等在 msvcrt printf 不支持
  endif()
endif()

# ---- t12.1 OpenBLAS 运行时分派（可选）----
# ops.c 在运行时 LoadLibrary libopenblas.dll（SGL_BLAS=0 可强制内置路径）；
# SGL_OPENBLAS=ON 只把 third_party 预编译 dll 拷到可执行文件目录，方便被找到。
option(SGL_OPENBLAS "Copy bundled OpenBLAS (third_party/openblas) next to binaries" OFF)
if(SGL_OPENBLAS)
  set(SGL_OPENBLAS_DLL ${CMAKE_CURRENT_SOURCE_DIR}/third_party/openblas/bin/libopenblas.dll)
  if(EXISTS ${SGL_OPENBLAS_DLL})
    # ALL target：每次构建都检查（copy_if_different），third_party 换包后自动同步
    add_custom_target(openblas_dll ALL
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SGL_OPENBLAS_DLL}
              ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    message(STATUS "OpenBLAS: bundled dll will be copied (runtime dispatch, SGL_BLAS=0 to disable)")
  else()
    message(WARNING "SGL_OPENBLAS=ON but ${SGL_OPENBLAS_DLL} missing (fetch OpenBLAS-0.3.29_x64_64.zip into third_party/openblas)")
  endif()
endif()

# ---- CLI ----
add_executable(sglc apps/sglc.c)
target_link_libraries(sglc PRIVATE sgl)

# ---- 测试 ----
enable_testing()
# 让 ctest 从任意工作目录都能找到真实权重（避免静默 SKIP）：
# 优先仓库外的 ../models（开发机布局），CI 单仓 checkout 时用仓内 models/。
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../models/gpt2)
  set(SGL_MODELS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../models)
else()
  set(SGL_MODELS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/models)
endif()
file(GLOB SGL_TESTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_*.c)
if(MSVC)
  # test_http 直接编 http.c（winpthreads）；MSVC 构建排除
  list(REMOVE_ITEM SGL_TESTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_http.c)
elseif(NOT WIN32)
  # POSIX 无 winsock
  list(REMOVE_ITEM SGL_TESTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_http.c)
endif()
foreach(t ${SGL_TESTS})
  get_filename_component(name ${t} NAME_WE)
  add_executable(${name} ${t})
  target_link_libraries(${name} PRIVATE sgl)
  add_test(NAME ${name} COMMAND ${name})
  set_tests_properties(${name} PROPERTIES ENVIRONMENT "SGL_TEST_MODEL=${SGL_MODELS_DIR}/gpt2;SGL_TEST_LLAMA=${SGL_MODELS_DIR}/tiny-llama;SGL_TEST_LLAMA_REF=${CMAKE_CURRENT_SOURCE_DIR}/tests/ref_tinyllama.json;SGL_TEST_TINYLLAMA=${SGL_MODELS_DIR}/tinyllama;SGL_TEST_QWEN=${SGL_MODELS_DIR}/qwen05;SGL_TEST_LORA=${SGL_MODELS_DIR}/test-lora;SGL_TEST_LORA_REF=${CMAKE_CURRENT_SOURCE_DIR}/tests/ref_lora.json;SGL_TEST_LORA_S1=${SGL_MODELS_DIR}/test-lora-s1;SGL_TEST_LORA_S2=${SGL_MODELS_DIR}/test-lora-s2;SGL_TEST_LORA_REF_S1=${CMAKE_CURRENT_SOURCE_DIR}/tests/ref_lora_s1.json;SGL_TEST_LORA_REF_S2=${CMAKE_CURRENT_SOURCE_DIR}/tests/ref_lora_s2.json")
endforeach()

# ---- CUDA 后端（SGL_CUDA=ON，需 MSVC + CUDA 工具链；与 MinGW 构建共存于独立目录）----
option(SGL_CUDA "Build CUDA backend (needs VS + nvcc)" OFF)
if(SGL_CUDA)
  # 注：MSBuild 的 CUDA 集成在此环境不稳（宿主环境差异导致 nvcc 阶段失败），
  # 改用自定义命令直接驱动 nvcc（参数已验证），产物 obj 链入静态库。
  # 工具链路径参数化（CI runner 版本不同）：CUDA 取 CUDA_PATH env 或 glob 最新 v*；
  # MSVC 宿主 glob VS2022 的 BuildTools/Community/Professional 任一 SKU。
  set(CUDA_TOOLKIT_DIR "$ENV{CUDA_PATH}")
  if(NOT CUDA_TOOLKIT_DIR)
    file(GLOB CUDA_TOOLKIT_DIRS "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*")
    if(CUDA_TOOLKIT_DIRS)
      list(SORT CUDA_TOOLKIT_DIRS)
      list(REVERSE CUDA_TOOLKIT_DIRS)
      list(GET CUDA_TOOLKIT_DIRS 0 CUDA_TOOLKIT_DIR)
    endif()
  endif()
  find_program(NVCC_EXE nvcc HINTS "${CUDA_TOOLKIT_DIR}/bin")
  if(NOT NVCC_EXE)
    message(FATAL_ERROR "nvcc not found (set CUDA_PATH or install CUDA toolkit)")
  endif()
  file(GLOB MSVC_HOST_BINS
       "C:/Program Files (x86)/Microsoft Visual Studio/*/BuildTools/VC/Tools/MSVC/*/bin/Hostx64/x64"
       "C:/Program Files/Microsoft Visual Studio/*/BuildTools/VC/Tools/MSVC/*/bin/Hostx64/x64"
       "C:/Program Files/Microsoft Visual Studio/*/Community/VC/Tools/MSVC/*/bin/Hostx64/x64"
       "C:/Program Files/Microsoft Visual Studio/*/Professional/VC/Tools/MSVC/*/bin/Hostx64/x64"
       "C:/Program Files/Microsoft Visual Studio/*/Enterprise/VC/Tools/MSVC/*/bin/Hostx64/x64")
  if(MSVC_HOST_BINS)
    list(SORT MSVC_HOST_BINS)
    list(REVERSE MSVC_HOST_BINS)
    list(GET MSVC_HOST_BINS 0 MSVC_CCBIN)
  else()
    message(FATAL_ERROR "MSVC host for nvcc (-ccbin) not found")
  endif()
  message(STATUS "CUDA toolkit: ${CUDA_TOOLKIT_DIR}")
  message(STATUS "nvcc host: ${MSVC_CCBIN}")
  set(SGL_CUDA_OBJ ${CMAKE_CURRENT_BINARY_DIR}/cuda_backend.obj)
  add_custom_command(
    OUTPUT ${SGL_CUDA_OBJ}
    COMMAND ${NVCC_EXE} -c ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda_backend.cu
            -I${CMAKE_CURRENT_SOURCE_DIR}/include
            -ccbin "${MSVC_CCBIN}"
            --generate-code=arch=compute_75,code=sm_75 -O2 -o ${SGL_CUDA_OBJ}
    DEPENDS src/cuda_backend.cu
    COMMENT "nvcc cuda_backend.cu (sm_75)"
    VERBATIM)
  set(SGL_CORE_SRC
      src/alloc.c src/log.c src/json.c src/tensor.c src/safetensors.c
      src/bpe.c src/ops.c src/quant.c src/llama.c src/sampling.c)
  add_library(sgl_cuda STATIC ${SGL_CORE_SRC} ${SGL_CUDA_OBJ})
  target_include_directories(sgl_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  if(MSVC)
    # /utf-8 只给 C 源（nvcc 不认该旗标；.cu 已纯 ASCII，头文件带 BOM）
    target_compile_options(sgl_cuda PUBLIC /O2 "$<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>")
    target_compile_definitions(sgl_cuda PUBLIC _CRT_SECURE_NO_WARNINGS)
  endif()
  add_executable(cuda_test tests/cuda_test.c)
  target_link_libraries(cuda_test PRIVATE sgl_cuda)
  add_executable(cuda_cli apps/cuda_cli.c)
  target_link_libraries(cuda_cli PRIVATE sgl_cuda)
  add_executable(long_vcpu tests/long_vcpu.c)  # t2 acceptance: 8K prefill GPU vs CPU
  target_link_libraries(long_vcpu PRIVATE sgl_cuda)
  add_executable(cuda_http apps/cuda_http.c)
  target_link_libraries(cuda_http PRIVATE sgl_cuda ws2_32)
  if(MSVC)
    foreach(t cuda_test cuda_cli long_vcpu cuda_http)
      target_link_libraries(${t} PRIVATE
          "${CUDA_TOOLKIT_DIR}/lib/x64/cudart_static.lib"
          "${CUDA_TOOLKIT_DIR}/lib/x64/cublas.lib")
      target_compile_options(${t} PRIVATE /O2 /utf-8)
      target_compile_definitions(${t} PRIVATE _CRT_SECURE_NO_WARNINGS)
    endforeach()
  endif()
  message(STATUS "CUDA backend: enabled (sm_75)")
endif()
