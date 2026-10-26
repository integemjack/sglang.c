# build_all2.ps1 — 两栈重建 + 全量回归（CPU ctest / CUDA cuda_test）
# 用法: powershell -File build_all2.ps1 [-Root <dir>] [-Ctest] [-CudaTest] [-SkipCpu]
param(
  [string]$Root = "C:\Users\ultra\Desktop\works\sglang\sglang-c",
  [switch]$Ctest,
  [switch]$CudaTest,
  [switch]$SkipCpu,
  [string]$Model = "..\models\qwen05"
)
$ErrorActionPreference = 'Continue'
Set-Location $Root
$MinGW = "C:\Users\ultra\AppData\Local\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin"
$CUDA = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin"
$env:PATH = "$MinGW;$CUDA;" + $env:PATH

powershell -ExecutionPolicy Bypass -File tests\bom_fix.ps1 -Root $Root

if (-not $SkipCpu) {
  Write-Output "=== CPU (Ninja/MinGW) build ==="
  cmake --build build 2>&1 | Select-String -Pattern "error|FAILED" | Select-Object -First 10
  Write-Output ("cpu_build_rc=" + $LASTEXITCODE)
}
Write-Output "=== CUDA (MSVC/nvcc) build ==="
cmake -S . -B build-cuda 2>&1 | Select-String -Pattern "CUDA backend|BOM|error"
cmake --build build-cuda --config Release 2>&1 | Select-String -Pattern "error|\.exe$" | Select-Object -First 20
Write-Output ("cuda_build_rc=" + $LASTEXITCODE)

if ($Ctest) {
  Write-Output "=== ctest (CPU) ==="
  ctest --test-dir build --output-on-failure 2>&1 | Select-Object -Last 8
}
if ($CudaTest) {
  Write-Output "=== cuda_test (GPU) ==="
  & ".\build-cuda\bin\Release\cuda_test.exe" $Model 2>&1 |
    Select-String -Pattern "\[prefix\]|\[long\]|\[w8\]|\[mm\]|\[batch\]|FAIL|all OK|maxerr" |
    Select-Object -First 20
  Write-Output ("cuda_test_rc=" + $LASTEXITCODE)
}
