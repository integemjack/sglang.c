# conc_test.ps1 — G3 并发/动态 batch 稳定性验收（graph ON/OFF 都跑）
# 用法: powershell -File conc_test.ps1 -Port 30022 -Rounds 5 [-Conc 12] [-GraphOn]
# 每轮：Conc 个并发 /generate，max_tokens 长短混合（1..48）→ 运行中 batch 在
# B=1..8 之间反复变化，逼出 decode graph 的跨 B 缓存 / 缓冲增长交互。
# 判定：全部请求拿到 completion + finish_reason；单轮墙钟不超预算；服务仍存活。
param([int]$Port = 30022, [int]$Rounds = 5, [int]$Conc = 12, [switch]$GraphOn)
$ErrorActionPreference = 'Continue'
$base = "http://127.0.0.1:$Port"
$fails = @()
$lens = @(1, 3, 8, 16, 24, 48, 2, 12, 32, 4, 6, 40)
$prompts = @('The capital of France is', 'Water boils at', 'My name is John and',
             'Paris is famous for', 'The sky is', 'Rivers flow', 'Mountains are',
             'The sun rises in', 'Ice melts at', 'Birds fly', 'Fish swim', 'Snow falls')
Write-Output ("== conc_test: {0} rounds x {1} concurrent (graph {2}) ==" -f $Rounds, $Conc, $(if ($GraphOn) { 'ON' } else { 'OFF/default' }))
for ($r = 1; $r -le $Rounds; $r++) {
  $jobs = @()
  $t0 = Get-Date
  for ($i = 0; $i -lt $Conc; $i++) {
    $f = Join-Path $env:TEMP ("ct_{0}_{1}.json" -f $r, $i)
    @{ text = $prompts[$i % $prompts.Length]; max_tokens = $lens[$i % $lens.Length]; temperature = 0 } |
      ConvertTo-Json -Compress | Set-Content -Path $f -Encoding ascii
    $jobs += Start-Job -ScriptBlock {
      param($u, $f)
      $o = curl.exe -s -m 180 -w "`n%{http_code} %{time_total}" -X POST "$u/generate" -H 'Content-Type: application/json' --data-binary "@$f"
      "$o"
    } -ArgumentList $base, $f
  }
  $null = $jobs | Wait-Job
  $done = 0; $bad = @(); $times = @()
  foreach ($j in $jobs) {
    $txt = (Receive-Job $j) -join ''
    if ($txt -match 'completion' -and $txt -match '200\s+([0-9.]+)$') {
      $done++; $times += [double]$Matches[1]
    } else { $bad += ("[{0}]" -f ($txt -replace '\s+', ' ')) }
    Remove-Job $j -Force
  }
  $dt = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
  $srv = ''
  if ($times.Count) {
    $srt = $times | Sort-Object
    $srv = ("| server-side req time: min {0}s med {1}s max {2}s" -f $srt[0], $srt[[int]($srt.Count/2)], $srt[-1])
  }
  Write-Output ("  round {0}: {1}/{2} done in {3}s wall {4} {5}" -f $r, $done, $Conc, $dt, $srv, $(if ($bad.Count) { ('FAIL: ' + ($bad -join ' ')) } else { '' }))
  if ($done -ne $Conc) { $fails += ("round {0}: {1}/{2} completed" -f $r, $done, $Conc) }
  Get-ChildItem (Join-Path $env:TEMP "ct_${r}_*.json") -ErrorAction SilentlyContinue | Remove-Item -Force
}
Start-Sleep 2
$h = curl.exe -s -m 8 -X POST "$base/health" -H 'Content-Type: application/json' --data-binary 'x'
if ($h -notmatch 'true') { $fails += 'server not alive after stress' }
Write-Output ("server alive: {0}" -f $(if ($h -match 'true') { 'PASS' } else { 'FAIL' }))
if ($fails.Count) { Write-Output ("conc_test: {0} FAILURES" -f $fails.Count); $fails | ForEach-Object { Write-Output "  $_" }; exit 1 }
Write-Output 'conc_test: ALL PASS'
