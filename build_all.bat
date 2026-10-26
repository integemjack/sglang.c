@echo off
rem ============================================================
rem build_all.bat - sglang-c one-shot build + test
rem   CPU  : MinGW (WinLibs GCC)  -> sglang-c/build   + ctest
rem   CUDA : MSVC + nvcc 13.3     -> sglang-c/build-cuda + cuda_test
rem Usage (from sglang/ root or anywhere):
rem   sglang-c\build_all.bat
rem Exits 0 only if: CPU build OK, CUDA build OK, ctest all green,
rem                  cuda_test prints "all OK".
rem ============================================================
setlocal EnableDelayedExpansion

set "ROOT=%~dp0.."
pushd "%ROOT%"
if errorlevel 1 (
    echo [build_all] ERROR: cannot cd to %ROOT%
    exit /b 1
)

rem --- toolchain PATH: WinLibs mingw64 (CPU) + CUDA v13.3 bin (runtime DLLs) ---
set "PATH=C:\Users\ultra\AppData\Local\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin;%PATH%"
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin;%PATH%"

echo [build_all 1/4] CPU build (MinGW, sglang-c/build) ...
cmake --build sglang-c\build
if errorlevel 1 (
    echo [build_all] ERROR: CPU build FAILED ^(step 1/4^)
    popd & exit /b 1
)

echo [build_all 2/4] CUDA build (MSVC, sglang-c/build-cuda, targets: cuda_test cuda_cli cuda_http) ...
cmake --build sglang-c\build-cuda --config Release --target cuda_test cuda_cli cuda_http
if errorlevel 1 (
    echo [build_all] ERROR: CUDA build FAILED ^(step 2/4^)
    popd & exit /b 1
)

echo [build_all 3/4] CPU tests: ctest --test-dir sglang-c/build ...
ctest --test-dir sglang-c\build --output-on-failure
if errorlevel 1 (
    echo [build_all] ERROR: ctest FAILED ^(step 3/4^)
    popd & exit /b 1
)

echo [build_all 4/4] CUDA tests: cuda_test.exe models/qwen05 ...
sglang-c\build-cuda\bin\Release\cuda_test.exe models\qwen05
if errorlevel 1 (
    echo [build_all] ERROR: cuda_test FAILED ^(step 4/4^)
    popd & exit /b 1
)

popd
echo [build_all] ALL OK
endlocal
exit /b 0
