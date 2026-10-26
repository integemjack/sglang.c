# soak_curve.ps1 — G5：长跑 RSS/VRAM 曲线与斜率判定
# 用法: powershell -File soak_curve.ps1 -Exe ..\build-cuda\bin\Release\cuda_http.exe `
#         -Model ..\models\qwen05 -Port 30024 -Minutes 6 [-IntervalSec 10] [-Env "SGL_CUDA_CTX=8192"]
# 做法：起服务 → 稳定 4 并发请求压 → 每 IntervalSec 采一次 (进程 RSS, GPU 已用显存)
#      → 停压停服务 → 输出曲线、最小二乘斜率（MB/min，跳过前 3 个预热样本）与判定。
# 判定：RSS 与显存的斜率绝对值 < 各自均值的 0.5%/min 视为无泄漏；否则 FAIL。
param(
  [Parameter(Mandatory = $true)][string]$Exe,
  [Parameter(Mandatory = $true)][string]$Model,
  [int]$Port = 30024,
  [int]$Minutes = 6,
  [int]$IntervalSec = 10,
  [string[]]$Env = @(),
  [int]$LoadConc = 4
)
$ErrorActionPreference = 'Stop'
$base = "http://127.0.0.1:$Port"
$logf = Join-Path $env:TEMP ("soak_curve_{0}.log" -f $Port)
if (Test-Path $logf) { Remove-Item $logf -Force }

foreach ($kv in $Env) { $k, $v = $kv -split '=', 2; Set-Item -Path "env:$k" -Value $v }
Write-Output ("== soak_curve: {0} {1} port={2} minutes={3} interval={4}s load={5} env={6} ==" -f (Split-Path $Exe -Leaf), $Model, $Port, $Minutes, $IntervalSec, $LoadConc, ($Env -join ','))

$p = Start-Process -FilePath (Resolve-Path $Exe) -ArgumentList @($Model, "$Port") `
     -RedirectStandardOutput $logf -RedirectStandardError "$logf.err" -PassThru -WindowStyle Hidden
$ready = $false
for ($i = 0; $i -lt 120; $i++) {
  Start-Sleep -Milliseconds 500
  if ((Get-Content $logf -Raw -ErrorAction SilentlyContinue) -match 'serving on') { $ready = $true; break }
  if ($p.HasExited) { break }
}
if (-not $ready) { Write-Output "FAIL: server not serving (exit=$($p.ExitCode))"; Get-Content $logf, "$logf.err" -ErrorAction SilentlyContinue | Select-Object -Last 8; exit 1 }

# 负载：固定并发，长短 max_tokens 轮换，一直打到 stop 标志
$stopf = Join-Path $env:TEMP "soak_stop_$Port.flag"
if (Test-Path $stopf) { Remove-Item $stopf -Force }
$jobs = @()
for ($c = 0; $c -lt $LoadConc; $c++) {
  $jobs += Start-Job -ScriptBlock {
    param($u, $c, $stop)
    $mt = 8; $i = $c
    while (-not (Test-Path $stop)) {
      $b = Join-Path $env:TEMP "soak_body_$($PID)_$($i).json"
      '{"text":"The capital of France is","max_tokens":' + $mt + ',"temperature":0}' | Set-Content $b -Encoding ascii
      $null = curl.exe -s -m 120 -X POST "$u/generate" -H 'Content-Type: application/json' --data-binary "@$b"
      Remove-Item $b -Force -ErrorAction SilentlyContinue
      $i++; $mt = @(4, 8, 16, 32, 64)[$i % 5]
    }
  } -ArgumentList $base, $c, $stopf
}

$samples = @()
$t0 = Get-Date
$n = [math]::Ceiling($Minutes * 60 / $IntervalSec)
for ($s = 0; $s -lt $n; $s++) {
  Start-Sleep -Seconds $IntervalSec
  $rss = $null; $gpu = $null
  try { $rss = [math]::Round(((Get-Process -Id $p.Id).WS / 1MB), 1) } catch { }
  $smi = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null
  if ($smi) { $gpu = [double]($smi | Select-Object -First 1) }
  $el = [math]::Round(((Get-Date) - $t0).TotalMinutes, 2)
  $samples += [pscustomobject]@{ min = $el; rss_mb = $rss; vram_mb = $gpu }
  Write-Output ("  t={0,5} min  RSS {1,9} MB  VRAM(whole GPU) {2,7} MB" -f $el, $rss, $gpu)
  if ($p.HasExited) { Write-Output 'FAIL: server exited during soak'; break }
}
New-Item -ItemType File -Path $stopf -Force | Out-Null
$null = $jobs | Wait-Job -Timeout 150
$jobs | Remove-Job -Force -ErrorAction SilentlyContinue
Remove-Item $stopf -Force -ErrorAction SilentlyContinue
$alive = -not $p.HasExited
if ($alive) { Stop-Process -Id $p.Id -Force }
Start-Sleep 1

function Slope($pts) {   # 最小二乘斜率 MB/min
  if ($pts.Count -lt 3) { return $null }
  $sx = 0.0; $sy = 0.0; $sxx = 0.0; $sxy = 0.0; $k = $pts.Count
  foreach ($q in $pts) { $sx += $q.min; $sy += $q.v; $sxx += $q.min * $q.min; $sxy += $q.min * $q.v }
  return (($k * $sxy - $sx * $sy) / ($k * $sxx - $sx * $sx))
}
$skip = [math]::Min(3, [math]::Max(0, $samples.Count - 3))
$rp = @($samples | Select-Object -Skip $skip | Where-Object { $null -ne $_.rss_mb } | ForEach-Object { [pscustomobject]@{ min = $_.min; v = $_.rss_mb } })
$vp = @($samples | Select-Object -Skip $skip | Where-Object { $null -ne $_.vram_mb } | ForEach-Object { [pscustomobject]@{ min = $_.min; v = $_.vram_mb } })
$rs = Slope $rp; $vs = Slope $vp
$rmean = ($rp | Measure-Object -Property v -Average).Average
$vmean = ($vp | Measure-Object -Property v -Average).Average
if ($null -eq $rmean -or $null -eq $vmean -or $rmean -le 0 -or $vmean -le 0) {
  Write-Output 'FAIL: 采样不足，无法算斜率'
  exit 1
}
Write-Output ("samples: {0} (slope 从第 {1} 个起算)" -f $samples.Count, ($skip + 1))
Write-Output ("RSS  : mean {0:N1} MB, slope {1:N3} MB/min ({2:N4} %/min)" -f $rmean, $rs, (100 * $rs / $rmean))
Write-Output ("VRAM : mean {0:N1} MB, slope {1:N3} MB/min ({2:N4} %/min)" -f $vmean, $vs, (100 * $vs / $vmean))
# G17：旧版把 server stderr 的任何含 error/Error 的行都当错误报 —— 实测是误报。
# 例：qwen2 加载器对可选张量打的 "q_norm not found / k_norm not found" 是无害 INFO；
# 而 `-match 'error'` 连路径里的子串也算。现改为：只有词界匹配 ERROR|FATAL|abort
# 且不在白名单的行才算真错误，干净服务不再报错误段。
$whitelist = 'not found|INFO|SKIP|debug'
$errLines = @()
if (Test-Path "$logf.err") {
  $errLines = @(Get-Content "$logf.err" | Where-Object {
      ($_ -match '(^|[^A-Za-z])(ERROR|FATAL|abort|Abort)([^A-Za-z]|$)') -and ($_ -notmatch $whitelist)
    })
}
if ($errLines.Count -gt 0) {
  Write-Output ("server stderr 有 {0} 条错误行（已过滤 INFO/not found 白名单）：" -f $errLines.Count)
  $errLines | Select-Object -Last 8 | ForEach-Object { Write-Output ("  " + $_) }
} else {
  Write-Output 'server stderr: no ERROR/FATAL lines (INFO whitelist filtered)'
}
$ok = $alive -and ($rs -ne $null) -and ([math]::Abs(100 * $rs / $rmean) -lt 0.5) -and ([math]::Abs(100 * $vs / $vmean) -lt 0.5)
Write-Output ("soak_curve: {0}" -f $(if ($ok) { 'PASS (无泄漏：斜率 <0.5%/min)' } else { 'FAIL' }))
Remove-Item $logf, "$logf.err" -Force -ErrorAction SilentlyContinue
if (-not $ok) { exit 1 }
