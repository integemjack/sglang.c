# t5 验收：流式==非流式；并发>8 排队完成；finish_reason=length 正确
$ErrorActionPreference = 'Stop'
$base = 'http://127.0.0.1:30002'
function PF($ok) { if ($ok) { 'PASS' } else { 'FAIL' } }

# 1) 流式 vs 非流式（chat，同 prompt 同参数）—— body 走临时文件（PS5.1 内联引号会被剥）
$bf = Join-Path $env:TEMP 't5_chat.json'      # plain (no stream)
$bs = Join-Path $env:TEMP 't5_chat_sse.json'  # streaming
'{"messages":[{"role":"user","content":"The capital of France is"}],"max_tokens":20,"temperature":0}' | Out-File -Encoding ascii $bf
'{"messages":[{"role":"user","content":"The capital of France is"}],"max_tokens":20,"temperature":0,"stream":true}' | Out-File -Encoding ascii $bs
$plain = (curl.exe -s -m 60 -X POST "$base/v1/chat/completions" -H 'Content-Type: application/json' --data-binary "@$bf") | ConvertFrom-Json
$plain_text = $plain.choices[0].message.content
$sse_raw = curl.exe -s -N -m 60 -X POST "$base/v1/chat/completions" -H 'Content-Type: application/json' --data-binary "@$bs"
$stream_text = ""
foreach ($line in ($sse_raw -split "`n")) {
  if ($line -match '^data: (.+)$' -and $Matches[1] -ne '[DONE]') {
    $d = $Matches[1] | ConvertFrom-Json
    if ($d.choices[0].delta.content) { $stream_text += $d.choices[0].delta.content }
  }
}
$ok1 = ($plain_text -eq $stream_text) -and ($plain_text.Length -gt 0)
Write-Output ("stream==plain: {0}  ({1} vs {2} chars)" -f (PF $ok1), $plain_text.Length, $stream_text.Length)

# 2) 12 并发（> 8 段）：排队后全部完成
$prompts = @('The capital of France is','Water boils at','My name is John and','Paris is famous for','The sky is','Rivers flow','Mountains are','The sun rises in','Ice melts','Birds fly','Fish swim','Snow falls')
$jobs = @()
$i = 0
foreach ($p in $prompts) {
  $t = Join-Path $env:TEMP "q12_$i.json"
  @{ text = $p; max_tokens = 16; temperature = 0 } | ConvertTo-Json -Compress | Set-Content -Path $t -Encoding ascii
  $jobs += Start-Job -ScriptBlock {
    param($f)
    curl.exe -s -m 120 -X POST 'http://127.0.0.1:30002/generate' -H 'Content-Type: application/json' --data-binary "@$f"
  } -ArgumentList $t
  $i++
}
$jobs | Wait-Job | Out-Null
$ok = 0
foreach ($j in $jobs) {
  $txt = Receive-Job $j
  if ($txt -match 'completion') { $ok++ } else { Write-Output "BAD: $txt" }
}
$jobs | Remove-Job -Force
$ok2 = ($ok -eq $prompts.Count)
Write-Output ("12-concurrent queued: {0}  ({1}/{2} done)" -f (PF $ok2), $ok, $prompts.Count)

# 3) finish_reason: max_tokens=1 -> length
$b3 = Join-Path $env:TEMP 't5_min.json'
'{"text":"Hi","max_tokens":1,"temperature":0}' | Out-File -Encoding ascii $b3
$r3 = (curl.exe -s -m 60 -X POST "$base/generate" -H 'Content-Type: application/json' --data-binary "@$b3") | ConvertFrom-Json
$ok3 = ($r3.meta_info.finish_reason -eq 'length')
Write-Output ("finish_reason=length: {0}  (got {1})" -f (PF $ok3), $r3.meta_info.finish_reason)

# 4) health alive after all（t5 已知：无 body 请求的 conn_thread 间歇卡死，
#    监控改用 POST /health -d x；并发刚结束的瞬间连接池未清理完，等 2s，见 docs/NOTES_t5.md）
Start-Sleep 2
$h = curl.exe -s -m 8 -X POST "$base/health" -H 'Content-Type: application/json' --data-binary 'x'
$ok4 = ($h -match 'true')
Write-Output ("server alive: {0}" -f (PF $ok4))
Get-ChildItem (Join-Path $env:TEMP 'q12_*.json') -ErrorAction SilentlyContinue | Remove-Item -Force
Remove-Item $bf, $bs, $b3 -ErrorAction SilentlyContinue
if ($ok1 -and $ok2 -and $ok3 -and $ok4) { Write-Output 't5 ALL PASS' } else { Write-Output 't5 HAS FAILURES'; exit 1 }
