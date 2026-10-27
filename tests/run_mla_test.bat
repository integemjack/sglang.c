@echo off
REM tests/run_mla_test.bat - MLA 数值测试的独立构建+运行（不改 CMakeLists.txt）
REM   1) python 生成随机 MLA 小模型 + numpy 参考
REM   2) gcc 编译：新文件 src/mla.c + 现有 build/libsgl.a（不重编整个库）
REM   3) 跑数值比对（prefill / KV cache / 增量 decode / pos 越界）
REM 用法: tests\run_mla_test.bat            全流程
REM       tests\run_mla_test.bat <模型目录>  只跑已有模型
setlocal
set ROOT=%~dp0..
cd /d "%ROOT%"
if not "%~1"=="" goto run_existing

if not exist build-mla mkdir build-mla

echo [1/3] gen random MLA models + numpy reference ...
python tests\gen_random_mla.py --out build-mla\mla-rand --seed 7 --q-lora 12
if errorlevel 1 exit /b 1
python tests\gen_random_mla.py --out build-mla\mla-rand-q0 --seed 11 --q-lora 0
if errorlevel 1 exit /b 1

echo [2/3] gcc build test_mla ...
gcc -O2 -std=c11 -Wall -Wextra -Wno-unused-parameter -Iinclude -fopenmp tests\test_mla.c src\mla.c build\libsgl.a -o build-mla\test_mla.exe -fopenmp -static -lws2_32
if errorlevel 1 exit /b 1

echo [3/3] run ...
set SGL_TEST_MLA=build-mla\mla-rand
set SGL_TEST_MLA_QLORA0=build-mla\mla-rand-q0
build-mla\test_mla.exe
exit /b %errorlevel%

:run_existing
if not exist build-mla\test_mla.exe (
  echo need build first: tests\run_mla_test.bat
  exit /b 1
)
build-mla\test_mla.exe %*
exit /b %errorlevel%