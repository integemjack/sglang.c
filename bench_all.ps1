# ============================================================
# bench_all.ps1 - sglang-c performance snapshot (Markdown table)
#
# Usage (from anywhere):
#   powershell -ExecutionPolicy Bypass -File sglang-c\bench_all.ps1
#
# Env switches (passthrough to cuda_cli, no script edit needed):
#   SGL_CUDA_FP16=1  -> also benchmark CUDA fp16 row
#   SGL_CUDA_W8=1    -> also benchmark CUDA int8 row
#
# Output: Markdown table to stdout + sglang-c/docs/BENCH.md (rewritten each run,
# idempotent). Each benchmark runs twice, the better tok/s is kept.
# --soak mode is NOT implemented here (planned in t11).
# ============================================================
$ErrorActionPreference = 'Continue'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root      = Split-Path -Parent $ScriptDir          # sglang/ repo root
Set-Location $Root

$env:Path = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin;" + $env:Path

$Sglc     = Join-Path $Root "sglang-c\build\bin\sglc.exe"
$CudaCli  = Join-Path $Root "sglang-c\build-cuda\bin\Release\cuda_cli.exe"
$CudaTest = Join-Path $Root "sglang-c\build-cuda\bin\Release\cuda_test.exe"
$BenchMd  = Join-Path $ScriptDir "docs\BENCH.md"

# Fixed benchmark params (do not change between runs, keeps snapshots comparable)
$MaxNew   = 64                 # cpu single-stream tokens
$CudaTok  = 40                 # cuda single-stream tokens
$CudaPrompt = "The meaning of life is"

function Write-Warn([string]$msg) { Write-Host "[bench_all] WARN: $msg" -ForegroundColor Yellow }

# ---- parsers (match actual binary output formats) ----
# sglc:    "generated 64 tokens in 24.76s (2.6 tok/s, 387 ms/tok, cache_hit=0)"
# cuda_cli:"[done] 40 tokens in 0.82s = 48.7 tok/s (GPU)"
# cuda_test:"[batch] ... 96 tok / 0.29s = 334.3 tok/s aggregate"
function Parse-Cpu([string]$out) {
    if ($out -match 'generated (\d+) tokens in ([\d.]+)s \(([\d.]+) tok/s') {
        [pscustomobject]@{ tok=[int]$Matches[1]; sec=[double]$Matches[2]; tps=[double]$Matches[3] }
    } else { $null }
}
function Parse-Cuda([string]$out) {
    if ($out -match '\[done\] (\d+) tokens in ([\d.]+)s = ([\d.]+) tok/s') {
        [pscustomobject]@{ tok=[int]$Matches[1]; sec=[double]$Matches[2]; tps=[double]$Matches[3] }
    } else { $null }
}

# ---- CPU run: 2 attempts, best tok/s ----
function Bench-Cpu([string[]]$CliArgs) {
    $best = $null
    foreach ($i in 1..2) {
        $out = & $Sglc @CliArgs 2>&1 | Out-String
        $r = Parse-Cpu $out
        if ($null -eq $r) { Write-Warn "sglc output not parseable (attempt $i): $($out | Select-Object -Last 1)"; continue }
        if ($null -eq $best -or $r.tps -gt $best.tps) { $best = $r }
    }
    return $best
}

# ---- CUDA run: 2 attempts, best tok/s; VRAM peak sampled via nvidia-smi ----
function Bench-Cuda([string[]]$CliArgs) {
    $tmpOut = [System.IO.Path]::GetTempFileName()
    $tmpErr = [System.IO.Path]::GetTempFileName()
    $best = $null; $peak = 0
    foreach ($i in 1..2) {
        # NB: PS5.1 ArgumentList array does not quote args with spaces -> pass one pre-quoted string
        $argStr = ($CliArgs | ForEach-Object { if ($_ -match ' ') { '"' + $_ + '"' } else { $_ } }) -join ' '
        $p = Start-Process -FilePath $CudaCli -ArgumentList $argStr -NoNewWindow -PassThru `
              -RedirectStandardOutput $tmpOut -RedirectStandardError $tmpErr
        while (-not $p.HasExited) {
            $m = (& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null)
            if ($m -match '(\d+)') { $v = [int]$Matches[1]; if ($v -gt $peak) { $peak = $v } }
            Start-Sleep -Milliseconds 120
        }
        $out = (Get-Content $tmpOut -Raw) + (Get-Content $tmpErr -Raw)
        $r = Parse-Cuda $out
        if ($null -eq $r) { $tail = ($out -split "`n" | Select-Object -Last 3) -join ' | '; Write-Warn "cuda_cli output not parseable (attempt $i): $tail"; continue }
        if ($null -eq $best -or $r.tps -gt $best.tps) { $best = $r }
    }
    Remove-Item $tmpOut, $tmpErr -ErrorAction SilentlyContinue
    [pscustomobject]@{ tok=$best.tok; sec=$best.sec; tps=$best.tps; vram=$peak }
}

# ---- CUDA batch: from cuda_test aggregate line ----
function Bench-CudaBatch {
    $out = & $CudaTest models\qwen05 2>&1 | Out-String
    if ($out -match '\[batch\].*?(\d+) tok / ([\d.]+)s = ([\d.]+) tok/s aggregate') {
        [pscustomobject]@{ tok=[int]$Matches[1]; sec=[double]$Matches[2]; tps=[double]$Matches[3] }
    } else {
        Write-Warn "cuda_test: no [batch] aggregate line found"
        $null
    }
}

$rows = New-Object System.Collections.Generic.List[string]
function Add-Row([string]$stack,[string]$prec,[string]$model,[string]$scene,[string]$tps,[string]$vram,[string]$note) {
    $rows.Add("| $stack | $prec | $model | $scene | $tps | $vram | $note |")
}

Write-Host "[bench_all] 1/6 CPU fp32 tinyllama single (max-new=$MaxNew) ..."
$r = Bench-Cpu @("--model","models/tinyllama","--max-new","$MaxNew")
if ($r) { Add-Row "cpu" "fp32" "tinyllama" "single" "$($r.tps)" "-" "$($r.tok) tok / $($r.sec)s (best of 2)" } else { Add-Row "cpu" "fp32" "tinyllama" "single" "ERROR" "-" "parse failed" }

Write-Host "[bench_all] 2/6 CPU w8+kv8 tinyllama single ..."
$r = Bench-Cpu @("--model","models/tinyllama","--max-new","$MaxNew","--w8","--kv8")
if ($r) { Add-Row "cpu" "w8" "tinyllama" "single" "$($r.tps)" "-" "W8A16 + int8 KV, $($r.tok) tok / $($r.sec)s (best of 2)" } else { Add-Row "cpu" "w8" "tinyllama" "single" "ERROR" "-" "parse failed" }

Write-Host "[bench_all] 3/6 CPU w8+kv8 qwen05 single ..."
$r = Bench-Cpu @("--model","models/qwen05","--max-new","$MaxNew","--w8","--kv8")
if ($r) { Add-Row "cpu" "w8" "qwen05" "single" "$($r.tps)" "-" "W8A16 + int8 KV, $($r.tok) tok / $($r.sec)s (best of 2)" } else { Add-Row "cpu" "w8" "qwen05" "single" "ERROR" "-" "parse failed" }

Write-Host "[bench_all] 4/6 CUDA fp32 qwen05 single (VRAM sampling via nvidia-smi) ..."
$r = Bench-Cuda @("models/qwen05",$CudaPrompt,"$CudaTok","0")
if ($r -and $r.tps) { Add-Row "cuda" "fp32" "qwen05" "single" "$($r.tps)" "$($r.vram)" "prompt='$CudaPrompt' greedy $($r.tok) tok / $($r.sec)s (best of 2)" } else { Add-Row "cuda" "fp32" "qwen05" "single" "ERROR" "-" "parse failed" }

# fp16 / int8 rows: real numbers only when env switches are set (t1 / t8 will set them)
if ($env:SGL_CUDA_FP16 -eq "1") {
    Write-Host "[bench_all] 5/6 CUDA fp16 qwen05 single (SGL_CUDA_FP16=1) ..."
    $r = Bench-Cuda @("models/qwen05",$CudaPrompt,"$CudaTok","0")
    if ($r -and $r.tps) { Add-Row "cuda" "fp16" "qwen05" "single" "$($r.tps)" "$($r.vram)" "SGL_CUDA_FP16=1, $($r.tok) tok / $($r.sec)s (best of 2)" }
    else { Add-Row "cuda" "fp16" "qwen05" "single" "ERROR" "-" "SGL_CUDA_FP16=1 but run/parse failed" }
    Write-Host "[bench_all] 5b/6 CUDA fp16 tinyllama-1.1B single (SGL_CUDA_FP16=1) ..."
    $r = Bench-Cuda @("models/tinyllama",$CudaPrompt,"$CudaTok","0")
    if ($r -and $r.tps) { Add-Row "cuda" "fp16" "tinyllama" "single" "$($r.tps)" "$($r.vram)" "1.1B fp16 ~2.2GB fits 6GB card, $($r.tok) tok / $($r.sec)s (best of 2)" }
    else { Add-Row "cuda" "fp16" "tinyllama" "single" "ERROR" "-" "SGL_CUDA_FP16=1 but run/parse failed" }
} else {
    Write-Host "[bench_all] 5/6 CUDA fp16 rows: placeholder (set SGL_CUDA_FP16=1 to measure)"
    Add-Row "cuda" "fp16" "qwen05" "single" "TODO(t1)" "-" "set SGL_CUDA_FP16=1 and rerun"
}
if ($env:SGL_CUDA_W8 -eq "1") {
    Write-Host "[bench_all] CUDA int8 qwen05 single (SGL_CUDA_W8=1) ..."
    $r = Bench-Cuda @("models/qwen05",$CudaPrompt,"$CudaTok","0")
    if ($r -and $r.tps) { Add-Row "cuda" "int8" "qwen05" "single" "$($r.tps)" "$($r.vram)" "SGL_CUDA_W8=1, $($r.tok) tok / $($r.sec)s (best of 2)" }
    else { Add-Row "cuda" "int8" "qwen05" "single" "ERROR" "-" "SGL_CUDA_W8=1 but run/parse failed" }
} else {
    Add-Row "cuda" "int8" "qwen05" "single" "TODO(待 t8)" "-" "set SGL_CUDA_W8=1 and rerun"
}

Write-Host "[bench_all] 6/6 CUDA batch aggregate (from cuda_test [batch] line) ..."
$r = Bench-CudaBatch
if ($r) { Add-Row "cuda" "fp32" "qwen05" "batch" "$($r.tps)" "-" "cuda_test [batch] B=4 x 24 steps, $($r.tok) tok / $($r.sec)s aggregate" } else { Add-Row "cuda" "fp32" "qwen05" "batch" "ERROR" "-" "cuda_test aggregate line not found" }

# ---- emit table to stdout ----
$stamp = Get-Date -Format "yyyy-MM-dd HH:mm"
$header = @(
    "# sglang-c 性能快照 (BENCH)",
    "",
    "行 schema: 栈(cpu/cuda) | 精度(fp32/fp16/w8/int8) | 模型 | 场景(single/batch) | tok/s | 显存峰值 MB | 备注",
    "",
    "由 ``sglang-c/bench_all.ps1`` 维护，每次运行整表重写（幂等）。固定 prompt 与 max-new，每项跑 2 次取较好值。",
    "CUDA fp16 / int8 行：设环境变量 ``SGL_CUDA_FP16=1`` / ``SGL_CUDA_W8=1`` 后重跑本脚本即出数。显存峰值由 nvidia-smi 每 120ms 采样（仅 cuda 行）。",
    "",
    "## 最新快照 ($stamp)",
    "",
    "| 栈 | 精度 | 模型 | 场景 | tok/s | 显存峰值 MB | 备注 |",
    "|----|------|------|------|-------|-------------|------|"
)
$md = ($header + $rows.ToArray() + "") -join "`r`n"
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $BenchMd) | Out-Null
Set-Content -Path $BenchMd -Value $md -Encoding UTF8
Write-Host ""
Write-Output $md
Write-Host ""
Write-Host "[bench_all] done -> $BenchMd"
