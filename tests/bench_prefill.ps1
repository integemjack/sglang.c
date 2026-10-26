# G16: prefill bench (three precisions) + dp4a/imma crossover + flash-vs-two-pass A/B.
# Usage: powershell -File tests\bench_prefill.ps1   (writes build\g16_final.log)
#   模型用 ..\models\qwen05（D=64，24 层），ctx 默认 16384；每步 rounds=2 取优，
#   最后一项是 --ab（同进程两遍 vs flash，WDDM 下绝对值会飘，比值才可信）。
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin;" + $env:PATH
$exe = Join-Path $PWD "build-cuda\bin\Release\cuda_perf.exe"
$out = "build\g16_final.log"
$tmp = Join-Path $env:TEMP "g16s.log"
if (Test-Path $out) { Remove-Item $out -Force }

function Run-Step([string]$label, [string]$argsLine) {
    "=== $label ===" | Add-Content $out
    $line = '"' + $exe + '" ' + $argsLine + ' > "' + $tmp + '" 2>&1'
    cmd /c $line | Out-Null
    Get-Content $tmp -Encoding UTF8 | Where-Object { $_ -match 'prefill|\[cfg\]|decode B' } | Add-Content $out
    Write-Output "done: $label"
}

$env:SGL_CUDA_FP16 = $null; $env:SGL_CUDA_W8 = $null
Run-Step "fp32 prefill (default = flash above 64 tok)" "..\models\qwen05 --prompt 512,2048,8192 --rounds 2 --batch 1"
$env:SGL_CUDA_FP16 = "1"
Run-Step "fp16 prefill" "..\models\qwen05 --prompt 512,2048,8192 --rounds 2 --batch 1"
$env:SGL_CUDA_W8 = "1"
Run-Step "int8 (imma) prefill" "..\models\qwen05 --prompt 512,2048,8192 --rounds 2 --batch 1"
Run-Step "int8 prefill: dp4a vs imma" "..\models\qwen05 --prompt 128,256,512,1024 --ab --rounds 1 --batch 1"
$env:SGL_CUDA_W8 = $null
Run-Step "fp32 prefill: flash vs two-pass A/B" "..\models\qwen05 --prompt 512,2048,8192 --ab --rounds 1 --batch 1"
Remove-Item $tmp -Force -ErrorAction SilentlyContinue
Write-Output "---- result ----"
Get-Content $out