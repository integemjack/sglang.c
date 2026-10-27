cmake_minimum_required(VERSION 3.20)
project(sglang_c VERSION 0.4.0 LANGUAGES C)

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
add_library(sgl STATIC ${SGL_SOURCES})
# third_party/stb：图片解码（G8，single-header，MIT）
target_include_directories(sgl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
                                  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb)

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

# ---- 源码 UTF-8 BOM 自检（MSVC/nvcc 按 GBK 读无 BOM 的中文注释会吞结构体声明，t11 踩坑两次）----
file(GLOB _bomcheck_src ${CMAKE_CURRENT_SOURCE_DIR}/src/*.[ch] ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
     ${CMAKE_CURRENT_SOURCE_DIR}/include/sgl/*.h ${CMAKE_CURRENT_SOURCE_DIR}/apps/*.c)
foreach(_f ${_bomcheck_src})
  file(READ ${_f} _magic LIMIT 3 HEX)
  if(NOT _magic STREQUAL "efbbbf")
    message(WARNING "missing UTF-8 BOM (MSVC build will misparse CJK comments): ${_f}")
  endif()
endforeach()

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
  set_tests_properties(${name} PROPERTIES ENVIRONMENT "SGL_TEST_MODEL=${SGL_MODELS_DIR}/gpt2;SGL_TEST_LLAMA=${SGL_MODELS_DIR}/tiny-llama;SGL_TEST_LLAMA_REF=${CMAKE_CURRENT_SOURCE_DIR}/tests/ref_tinyllama.json;SGL_TEST_TINYLLAMA=${SGL_MODELS_DIR}/tinyllama;SGL_TEST_QWEN=${SGL_MODELS_DIR}/qwen05;SGL_TEST_LORA=${SGL_MODELS_DIR}/test-lora;SGL_TEST_LORA_REF=${CMAKE_CURRENT_SOURCE_DIR}/tests/ref_lora.json;SGL_TEST_LORA_S1=${SGL_MODELS_DIR}/test-lora-s1;SGL_TEST_LORA_S2=${SGL_MODELS_DIR}/test-lora-s2;SGL_TEST_LORA_REF_S1=${CMAKE_CURRENT_SOURCE_DIR}/tests/ref_lora_s1.json;SGL_TEST_LORA_REF_S2=${CMAKE_CURRENT_SOURCE_DIR}/tests/ref_lora_s2.json;SGL_TEST_LLAVA=${SGL_MODELS_DIR}/llava-rand;SGL_TEST_CLIP_REAL=${SGL_MODELS_DIR}/clip-vit-base-patch32")
endforeach()

# ---- t10 llava 多模态：给 glob 生成的 test_llava 指定模型目录（存在才设）----
if(EXISTS ${SGL_MODELS_DIR}/llava-rand/clip_model.safetensors AND EXISTS ${SGL_MODELS_DIR}/llava-rand/_ref_proj.npy)
  set_tests_properties(test_llava PROPERTIES DEPENDS test_llama
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..)
endif()

# ---- G8 真实 CLIP 权重分层对照（下载 + 跑 tests/clip_real_ref.py 后才注册，防 fresh-clone 红）----
if(EXISTS ${SGL_MODELS_DIR}/clip-vit-base-patch32/model.safetensors AND
   EXISTS ${SGL_MODELS_DIR}/clip-vit-base-patch32/ref/pv.npy)
  set_tests_properties(test_clip_real PROPERTIES DEPENDS test_llava
    ENVIRONMENT "SGL_TEST_CLIP_REAL=${SGL_MODELS_DIR}/clip-vit-base-patch32")
endif()

# ---- t9/G9 模型族对照（models/ 下存在对应（生成或下载的）目录时才注册，防 fresh-clone 红）----
# G9 新增 gemma2（dynamic NTK rope 变体）/ qwen3next（layer_types 混合 linear_attention）
foreach(fam mistral gemma gemma2 qwen3 qwen3next)
  if(EXISTS ${SGL_MODELS_DIR}/${fam}-rand/model.safetensors OR EXISTS ${SGL_MODELS_DIR}/qwen3-06/model.safetensors)
    if(fam STREQUAL "qwen3")
      set(_d ${SGL_MODELS_DIR}/qwen3-06)
      set(_r ${CMAKE_CURRENT_SOURCE_DIR}/tests/_ref_qwen3.json)
    else()
      set(_d ${SGL_MODELS_DIR}/${fam}-rand)
      set(_r ${CMAKE_CURRENT_SOURCE_DIR}/tests/_ref_${fam}.json)
    endif()
    if(EXISTS ${_d}/model.safetensors AND EXISTS ${_r})
      add_test(NAME test_family_${fam} COMMAND test_llama ${_d} ${_r})
      set_tests_properties(test_family_${fam} PROPERTIES DEPENDS test_llama)
    endif()
  endif()
endforeach()

# ---- G13 递归状态按序列归属（radix 前缀共享下两条序列不能互踩）----
# test_lin_seq.c 已被上面的 glob 注册为 test_lin_seq，这里只补模型路径环境变量
if(EXISTS ${SGL_MODELS_DIR}/qwen3next-rand/model.safetensors)
  set_tests_properties(test_lin_seq PROPERTIES DEPENDS test_llama
    ENVIRONMENT "SGL_TEST_QNEXT=${SGL_MODELS_DIR}/qwen3next-rand")
else()
  set_tests_properties(test_lin_seq PROPERTIES ENVIRONMENT "SGL_TEST_QNEXT=")
endif()

# ---- G12 真实权重分段对照（CLIP-L/14-336 + llava mm_projector，参考由 llava_real_ref.py 生成）----
# test_llava_real.c 已被上面的 glob 注册，这里只补模型路径（无参考时留空 -> 自跳过）
if(EXISTS ${SGL_MODELS_DIR}/clip-vit-large-patch14-336/_ref_vis.npy AND
   EXISTS ${SGL_MODELS_DIR}/llava-real/_ref_real_proj.npy)
  set_tests_properties(test_llava_real PROPERTIES DEPENDS test_llava
    ENVIRONMENT "SGL_TEST_LLAVA_REAL_CLIP=${SGL_MODELS_DIR}/clip-vit-large-patch14-336;SGL_TEST_LLAVA_REAL_PROJ=${SGL_MODELS_DIR}/llava-real")
else()
  set_tests_properties(test_llava_real PROPERTIES ENVIRONMENT
    "SGL_TEST_LLAVA_REAL_CLIP=;SGL_TEST_LLAVA_REAL_PROJ=")
endif()

# ---- G12 占位符位置插入 visual token（需 g12-mlp：LM + 2层 MLP projector + <image>）----
if(EXISTS ${SGL_MODELS_DIR}/g12-mlp/projector.safetensors AND
   EXISTS ${SGL_MODELS_DIR}/g12-mlp/clip_model.safetensors)
  set_tests_properties(test_g12_place PROPERTIES DEPENDS test_llama
    ENVIRONMENT "SGL_TEST_G12=${SGL_MODELS_DIR}/g12-mlp")
else()
  set_tests_properties(test_g12_place PROPERTIES ENVIRONMENT "SGL_TEST_G12=")
endif()

# ---- CUDA 后端（SGL_CUDA=ON，需 CUDA 工具链；Windows = MSVC host，Linux = GCC host）----
# 两个平台共用同一份 nvcc 命令行骨架，差异只收在下面 if(MSVC)/else 两个分支里：
#   Windows：-ccbin 指向 VS 的 Hostx64/x64，产物 .obj，链 Toolkit 的 lib/x64/*.lib，cuda_http 链 ws2_32
#   Linux  ：nvcc 自己找 g++（PATH 里必须有），产物 .o（-Xcompiler -fPIC），链 lib64 的静态 cudart + cublas
option(SGL_CUDA "Build CUDA backend (needs CUDA toolkit; MSVC host compiler on Windows)" OFF)
if(SGL_CUDA)
  # 注：MSBuild 的 CUDA 集成在此环境不稳（宿主环境差异导致 nvcc 阶段失败），
  # 改用自定义命令直接驱动 nvcc（参数已验证），产物 obj/o 链入静态库。
  # toolkit 定位三选：CUDA_PATH / CUDA_HOME env -> 平台默认目录（Windows glob 最新 v*；
  # Linux/macOS /usr/local/cuda）-> nvcc 在 PATH 里时从它的路径反推。
  set(CUDA_TOOLKIT_DIR "")
  if(DEFINED ENV{CUDA_PATH} AND EXISTS "$ENV{CUDA_PATH}")
    set(CUDA_TOOLKIT_DIR "$ENV{CUDA_PATH}")
  elseif(DEFINED ENV{CUDA_HOME} AND EXISTS "$ENV{CUDA_HOME}")
    set(CUDA_TOOLKIT_DIR "$ENV{CUDA_HOME}")
  elseif(WIN32)
    file(GLOB CUDA_TOOLKIT_DIRS "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*")
    if(CUDA_TOOLKIT_DIRS)
      list(SORT CUDA_TOOLKIT_DIRS)
      list(REVERSE CUDA_TOOLKIT_DIRS)
      list(GET CUDA_TOOLKIT_DIRS 0 CUDA_TOOLKIT_DIR)
    endif()
  elseif(EXISTS "/usr/local/cuda")
    set(CUDA_TOOLKIT_DIR "/usr/local/cuda")
  endif()
  # find_program 找不到 HINTS 里的会继续搜 PATH（nvcc 在 PATH 时也能用）
  find_program(NVCC_EXE nvcc HINTS "${CUDA_TOOLKIT_DIR}/bin")
  if(NOT NVCC_EXE)
    message(FATAL_ERROR "nvcc not found (set CUDA_PATH or install CUDA toolkit)")
  endif()
  if(NOT CUDA_TOOLKIT_DIR)
    # 从 nvcc 路径反推 toolkit 根（<root>/bin/nvcc）
    get_filename_component(_nvcc_bin "${NVCC_EXE}" DIRECTORY)
    get_filename_component(CUDA_TOOLKIT_DIR "${_nvcc_bin}" DIRECTORY)
  endif()
  # 一个或多个 SM 架构（分号分隔）：GPU 包默认单架 75；Windows on Arm 等需要
  # 「新卡为主、保持 x64 对齐」的场景可以传 75;120，每个架构各生成一段 cubin。
  set(SGL_CUDA_ARCH "75" CACHE STRING "CUDA SM architecture(s), e.g. 75 or 75;120")
  message(STATUS "CUDA toolkit: ${CUDA_TOOLKIT_DIR}")
  message(STATUS "nvcc: ${NVCC_EXE} (arch sm_${SGL_CUDA_ARCH})")
  if(MSVC)
    # nvcc 的 host 编译器目录：直接用当前 C 编译器所在目录，自动适配
    # Hostx64/x64（x64 宿主）与 Hostarm64/arm64（Windows on Arm —— CUDA 13.4 起
    # 才有 arm64 版 nvcc）。只有拿不到 CMAKE_C_COMPILER 目录时才退回 glob 搜。
    get_filename_component(_cl_dir "${CMAKE_C_COMPILER}" DIRECTORY)
    if(EXISTS "${_cl_dir}/cl.exe" OR EXISTS "${_cl_dir}/cl")
      set(MSVC_CCBIN "${_cl_dir}")
    else()
      file(GLOB MSVC_HOST_BINS
           "C:/Program Files (x86)/Microsoft Visual Studio/*/BuildTools/VC/Tools/MSVC/*/bin/Host*/x64"
           "C:/Program Files/Microsoft Visual Studio/*/BuildTools/VC/Tools/MSVC/*/bin/Host*/x64"
           "C:/Program Files/Microsoft Visual Studio/*/Community/VC/Tools/MSVC/*/bin/Host*/x64"
           "C:/Program Files/Microsoft Visual Studio/*/Professional/VC/Tools/MSVC/*/bin/Host*/x64"
           "C:/Program Files/Microsoft Visual Studio/*/Enterprise/VC/Tools/MSVC/*/bin/Host*/x64"
           "C:/Program Files (x86)/Microsoft Visual Studio/*/BuildTools/VC/Tools/MSVC/*/bin/Host*/arm64"
           "C:/Program Files/Microsoft Visual Studio/*/BuildTools/VC/Tools/MSVC/*/bin/Host*/arm64"
           "C:/Program Files/Microsoft Visual Studio/*/Community/VC/Tools/MSVC/*/bin/Host*/arm64"
           "C:/Program Files/Microsoft Visual Studio/*/Professional/VC/Tools/MSVC/*/bin/Host*/arm64"
           "C:/Program Files/Microsoft Visual Studio/*/Enterprise/VC/Tools/MSVC/*/bin/Host*/arm64")
      if(MSVC_HOST_BINS)
        list(SORT MSVC_HOST_BINS)
        list(REVERSE MSVC_HOST_BINS)
        list(GET MSVC_HOST_BINS 0 MSVC_CCBIN)
      else()
        message(FATAL_ERROR "MSVC host for nvcc (-ccbin) not found")
      endif()
    endif()
    message(STATUS "nvcc host: ${MSVC_CCBIN}")
    set(SGL_CUDA_OBJ ${CMAKE_CURRENT_BINARY_DIR}/cuda_backend.obj)
    set(SGL_NVCC_HOST_FLAGS -ccbin "${MSVC_CCBIN}")
    set(SGL_NVCC_ARCH_FLAGS "")
    foreach(_a ${SGL_CUDA_ARCH})
      list(APPEND SGL_NVCC_ARCH_FLAGS
           --generate-code=arch=compute_${_a},code=sm_${_a})
    endforeach()
    set(SGL_NVCC_EXTRA_FLAGS "")
  else()
    # 非 MSVC：host 编译器由 nvcc 自己在 PATH 里找 g++/clang++；产物 .o 要 -fPIC
    #（Ubuntu 的 gcc 默认 -fPIE，PIC 目标同样能链进 PIE 可执行文件）。
    # 额外嵌一份 PTX：sm_75 的 cubin 不能直接跑在 sm_80+ 上，驱动可 JIT 这份 PTX。
    set(SGL_CUDA_OBJ ${CMAKE_CURRENT_BINARY_DIR}/cuda_backend.o)
    set(SGL_NVCC_HOST_FLAGS "")
    set(SGL_NVCC_ARCH_FLAGS "")
    foreach(_a ${SGL_CUDA_ARCH})
      # 一个 --generate-code 里要带两个 code 必须用方括号（否则 nvcc 报
      # "redefinition of keyword 'code'"）。cubin 优先，PTX 供 JIT 兼容新卡。
      list(APPEND SGL_NVCC_ARCH_FLAGS
           "--generate-code=arch=compute_${_a},code=[sm_${_a},compute_${_a}]")
    endforeach()
    set(SGL_NVCC_EXTRA_FLAGS -Xcompiler -fPIC)
  endif()
  add_custom_command(
    OUTPUT ${SGL_CUDA_OBJ}
    COMMAND ${NVCC_EXE} -c ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda_backend.cu
            -I${CMAKE_CURRENT_SOURCE_DIR}/include
            ${SGL_NVCC_HOST_FLAGS}
            ${SGL_NVCC_ARCH_FLAGS}
            ${SGL_NVCC_EXTRA_FLAGS}
            -O2 -o ${SGL_CUDA_OBJ}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda_backend.cu
    COMMENT "nvcc cuda_backend.cu (sm_${SGL_CUDA_ARCH})"
    VERBATIM)
  # sgl_cuda = 与 CPU 库 sgl 同一份源（含 http.c / scheduler.c / radix.c）+ nvcc 后端对象。
  # 这样 sglc.exe 可以只链 sgl_cuda 一个库，就把 CPU 引擎和 GPU 引擎都编进同一个可执行文件
  # （--device auto：有显卡跑 GPU，没显卡回落 CPU，不再需要 cuda_http.exe 这个额外程序）。
  set(SGL_CORE_SRC ${SGL_SOURCES})
  if(MSVC)
    # dot_avx2_msvc.c 只服务 MSVC（需要 /arch:AVX2 + <intrin.h>）：GCC/Clang 的 AVX2 内积
    # 在 ops.c 用函数级 target("avx2,fma") 实现，两处都编会撞符号（sgl_dot_f32_avx2 重复定义）
    list(APPEND SGL_CORE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/dot_avx2_msvc.c) # G18: MSVC 的 AVX2 内积（单独 /arch:AVX2 编译 + CPUID 分派）
    list(REMOVE_DUPLICATES SGL_CORE_SRC)
  endif()
  add_library(sgl_cuda STATIC ${SGL_CORE_SRC} ${SGL_CUDA_OBJ})
  # dot_avx2_msvc.c 单独开 /arch:AVX2（MSVC 没有函数级 target 属性）
  if(MSVC)
    set_source_files_properties(src/dot_avx2_msvc.c PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
  endif()
  target_include_directories(sgl_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb) # G8: image.c 在 SGL_CORE_SRC 里，stb 头要能找到
  if(MSVC)
    # /utf-8 只给 C 源（nvcc 不认该旗标；.cu 已纯 ASCII，头文件带 BOM）
    target_compile_options(sgl_cuda PUBLIC /O2 "$<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>")
    target_compile_definitions(sgl_cuda PUBLIC _CRT_SECURE_NO_WARNINGS)
  else()
    # 与 CPU 库 sgl 保持同一套选项（-Wall/-Wextra 来自上面的全局 add_compile_options）
    target_compile_options(sgl_cuda PUBLIC -O2)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
      if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64")
        target_compile_options(sgl_cuda PUBLIC -funroll-loops)
      else()
        target_compile_options(sgl_cuda PUBLIC -march=x86-64-v2 -funroll-loops)
      endif()
    endif()
    if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
      # 与 sgl 一致：OpenMP 走编译器驱动（否则 ops.c 的 #pragma omp 在 sgl_cuda 里被忽略，
      # 且 -Wall 会报 -Wunknown-pragmas）
      target_compile_options(sgl_cuda PUBLIC -fopenmp)
      target_link_options(sgl_cuda INTERFACE -fopenmp)
    endif()
  endif()
  add_executable(cuda_test tests/cuda_test.c)
  target_link_libraries(cuda_test PRIVATE sgl_cuda)
  add_executable(cuda_cli apps/cuda_cli.c)
  target_link_libraries(cuda_cli PRIVATE sgl_cuda)
  add_executable(long_vcpu tests/long_vcpu.c)  # t2 acceptance: 8K prefill GPU vs CPU
  target_link_libraries(long_vcpu PRIVATE sgl_cuda)
  add_executable(cuda_perf tests/cuda_perf.c)  # G1/G2/G3 perf A/B (no assertions)
  target_link_libraries(cuda_perf PRIVATE sgl_cuda)
  add_executable(cuda_http apps/cuda_http.c)
  # ---- sglc 单 exe：内嵌 GPU 引擎（sglc --serve --device auto 时优先用它）----
  # cuda_http.c 的 main 被 SGL_EMBED_CUDA_HTTP 包住，sglc 只调它的 sgl_cuda_http_serve()。
  # 只链 sgl_cuda（含 CPU 侧全部源），不再链 sgl：同一份源进两个静态库会撞符号。
  target_sources(sglc PRIVATE apps/cuda_http.c)
  target_compile_definitions(sglc PRIVATE SGL_EMBED_CUDA_HTTP=1)
  set_target_properties(sglc PROPERTIES LINK_LIBRARIES "")
  target_link_libraries(sglc PRIVATE sgl_cuda)
  if(WIN32)
    target_link_libraries(cuda_http PRIVATE sgl_cuda ws2_32) # winsock
  else()
    target_link_libraries(cuda_http PRIVATE sgl_cuda) # POSIX socket 在 libc
  endif()
  if(MSVC)
    # Windows 的 CUDA 库在 lib\<arch>\（x64 / arm64）下；Windows on Arm 用 arm64 子目录，
    # 找不到时退回 lib 根（个别安装器版本会把 .lib 直接放 lib\）。
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "ARM64|arm64|aarch64")
      set(SGL_WIN_CUDA_ARCH_DIR "arm64")
    else()
      set(SGL_WIN_CUDA_ARCH_DIR "x64")
    endif()
    if(EXISTS "${CUDA_TOOLKIT_DIR}/lib/${SGL_WIN_CUDA_ARCH_DIR}/cublas.lib")
      set(SGL_WIN_CUDA_LIBDIR "${CUDA_TOOLKIT_DIR}/lib/${SGL_WIN_CUDA_ARCH_DIR}")
    elseif(EXISTS "${CUDA_TOOLKIT_DIR}/lib/cublas.lib")
      set(SGL_WIN_CUDA_LIBDIR "${CUDA_TOOLKIT_DIR}/lib")
    else()
      message(FATAL_ERROR "cublas.lib not found under ${CUDA_TOOLKIT_DIR}/lib (arch ${SGL_WIN_CUDA_ARCH_DIR})")
    endif()
    message(STATUS "CUDA libs: ${SGL_WIN_CUDA_LIBDIR}")
    foreach(t cuda_test cuda_cli long_vcpu cuda_perf cuda_http sglc)
      target_link_libraries(${t} PRIVATE
          "${SGL_WIN_CUDA_LIBDIR}/cudart_static.lib"
          "${SGL_WIN_CUDA_LIBDIR}/cublas.lib")
      target_compile_options(${t} PRIVATE /O2 /utf-8)
      target_compile_definitions(${t} PRIVATE _CRT_SECURE_NO_WARNINGS)
    endforeach()
  else()
    # Linux/macOS：Toolkit 的 lib64（老包是 lib）里取静态 cudart + cublas；
    # nvcc 的 host 代码是 C++，静态 cudart 需要 libstdc++；cudart 自己还要 pthread/dl/rt。
    find_library(SGL_CUDART_STATIC NAMES libcudart_static.a cudart_static
                 HINTS "${CUDA_TOOLKIT_DIR}/lib64" "${CUDA_TOOLKIT_DIR}/lib" /usr/local/cuda/lib64)
    find_library(SGL_CUBLAS_LIB NAMES cublas
                 HINTS "${CUDA_TOOLKIT_DIR}/lib64" "${CUDA_TOOLKIT_DIR}/lib" /usr/local/cuda/lib64)
    if(NOT SGL_CUDART_STATIC OR NOT SGL_CUBLAS_LIB)
      message(FATAL_ERROR "CUDA runtime/cublas not found under ${CUDA_TOOLKIT_DIR} (lib64/lib)")
    endif()
    # nvcc 生成的 host 代码是 C++（device stub 里有 __cxa_guard_* 等），链接时必须有
    # libstdc++。用 -lstdc++ 而不是 find_library：后者在某些镜像上找不到，漏到链接期
    # 才报 "undefined reference to __cxa_guard_acquire"。
    set(SGL_CUDA_RUNTIME_LIBS ${SGL_CUDART_STATIC} ${SGL_CUBLAS_LIB} stdc++ pthread dl rt m)
    message(STATUS "CUDA link libs: ${SGL_CUDA_RUNTIME_LIBS}")
    foreach(t cuda_test cuda_cli long_vcpu cuda_perf cuda_http sglc)
      target_link_libraries(${t} PRIVATE ${SGL_CUDA_RUNTIME_LIBS})
    endforeach()
  endif()
  message(STATUS "CUDA backend: enabled (sm_${SGL_CUDA_ARCH})")
endif()
