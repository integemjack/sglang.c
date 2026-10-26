# prefix_e2e.ps1 — G6 端到端：同一个长前缀的第二个请求 prefill 显著变快，
# 且输出与「无缓存时单独发」逐字一致。
# 用法: powershell -File prefix_e2e.ps1 -Port 30025 [-Exe path] [-Model dir]
# 步骤：起服务 → 单独发 B（无缓存基线）→ 重启 → 先发 A（长前缀+S1）再发 B（同前缀+S2）
#      → 比输出、比耗时、比 /health 的 hits 计数。
param(
  [int]$Port = 30025,
  [string]$Exe = ".\sglang-c-gpu\build-cuda\bin\Release\cuda_http.exe",
  [string]$Model = ".\models\qwen05",
  [string]$CUDA = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin"
)
$ErrorActionPreference = 'Stop'
$env:PATH = $CUDA + ';' + $env:PATH
$env:SGL_CUDA_CTX = '8192'
$base = "http://127.0.0.1:$Port"

# 长前缀（约 640 token）+ 不同后缀
$prefix = (1..160 | ForEach-Object { "The quick brown fox jumps over the lazy dog number $_." }) -join ' '
$pre = $prefix.Substring(0, [math]::Min(3000, $prefix.Length))
$s1 = ' Now describe the weather in spring.'
$s2 = ' Now list three kinds of trees.'

function Body($text, $maxTok) {
  $f = Join-Path $env:TEMP ("pfx_{0}_{1}.json" -f $maxTok, ([Math]::Abs($text.GetHashCode())))
  (@{ text = $text; max_tokens = $maxTok; temperature = 0.0 } | ConvertTo-Json -Compress) |
    Set-Content -Path $f -Encoding ascii
  return $f
}

function RunServer() {
  $logf = Join-Path $env:TEMP "prefix_e2e_$Port.log"
  $p = Start-Process -FilePath (Resolve-Path $Exe) -ArgumentList @($Model, "$Port") `
       -RedirectStandardOutput $logf -RedirectStandardError "$logf.err" -PassThru -WindowStyle Hidden
  for ($i = 0; $i -lt 240; $i++) {
    Start-Sleep -Milliseconds 500
    if ((Get-Content $logf -Raw -ErrorAction SilentlyContinue) -match 'serving on') { return @($p, $logf) }
    if ($p.HasExited) { break }
  }
  throw "server did not come up; see $logf"
}

function StopServer($r) { Stop-Process -Id $r[0].Id -Force -ErrorAction SilentlyContinue; Start-Sleep 2 }

function Ask($text, $maxTok) {
  $f = Body $text $maxTok
  $o = curl.exe -s -m 300 -w "`n%{time_total}" -X POST "$base/generate" -H 'Content-Type: application/json' --data-binary "@$f"
  Remove-Item $f -Force -ErrorAction SilentlyContinue
  $lines = $o -split "`n"
  $t = [double]($lines[-1])
  $json = ($lines[0..($lines.Length - 2)] -join '')
  $txt = ''
  try { $txt = ($json | ConvertFrom-Json).completion } catch { $txt = "<bad json: $json>" }
  return [pscustomobject]@{ time = $t; text = $txt }
}

Write-Output "== G6 prefix e2e =="
# --- 1) 基线：重启后直接发 B（无缓存） ---
$r = RunServer
$b0 = Ask ($pre + $s2) 16
$hb0 = (curl.exe -s "$base/health") | ConvertFrom-Json
Write-Output ("  baseline B (cold)  : {0}s  hits={1}" -f [math]::Round($b0.time, 2), $hb0.prefix.hits)
StopServer $r

# --- 2) 先发 A 建立租让，再发 B ---
$r = RunServer
$a = Ask ($pre + $s1) 16
Write-Output ("  A (cold, prefix+S1): {0}s" -f [math]::Round($a.time, 2))
Start-Sleep 1
$b1 = Ask ($pre + $s2) 16
$hb1 = (curl.exe -s "$base/health") | ConvertFrom-Json
Write-Output ("  B (warm, prefix+S2): {0}s  hits={1} hit_rows={2} leased={3}" -f `
  [math]::Round($b1.time, 2), $hb1.prefix.hits, $hb1.prefix.hit_rows, $hb1.prefix.leased_rows)
StopServer $r

$ok_time = $b1.time -lt ($b0.time * 0.85)
$ok_txt = ($b1.text -eq $b0.text) -and ($b1.text -ne '')
$ok_hit = $hb1.prefix.hits -ge 1
Write-Output ("  输出一致(缓存 vs 无缓存): {0}" -f $(if ($ok_txt) { 'PASS' } else { "FAIL`n    cold=$($b0.text)`n    warm=$($b1.text)" }))
Write-Output ("  二发更快(>=15%): {0}  ({1}s vs {2}s)" -f $(if ($ok_time) { 'PASS' } else { 'FAIL' }), [math]::Round($b1.time, 2), [math]::Round($b0.time, 2))
Write-Output ("  hits 计数生效: {0}" -f $(if ($ok_hit) { 'PASS' } else { 'FAIL' }))
if ($ok_time -and $ok_txt -and $ok_hit) { Write-Output 'prefix_e2e: ALL PASS'; exit 0 } else { Write-Output 'prefix_e2e: FAIL'; exit 1 }
