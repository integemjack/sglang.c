# prefix_e2e.ps1 — G15 端到端：前缀缓存四组验收
#   1) 同前缀      : A=前缀+S1 → B=前缀+S2，B 命中 len(前缀) 行，输出与关缓存逐字一致
#   2) 跳段/续写   : A=前缀+S1（生成 16 token）→ B2=前缀+S1+A的生成结果+S2，
#                    B2 的命中行数 > 同前缀时的命中行数（多出来的就是「生成 token 的
#                    KV 行」，以前完全不登记），输出与关缓存逐字一致
#   3) 带图        : 两条同文本、不同图片的视觉请求，第二条命中 text 部分，
#                    输出与关缓存逐字一致
#   4) 显存紧张    : SGL_CUDA_CTX 压到一段装不下长 prompt → 请求被明确拒绝、
#                    prefix.flushes 增长、随后的小请求仍然正确
# 用法: powershell -File tests\prefix_e2e.ps1 [-Port 30025] [-Model ..\models\qwen05]
#        [-MmModel ..\models\mm-test] [-Exe build-cuda\bin\Release\cuda_http.exe]
# 关于「跳段共享」：本服务的租让表是扁平表，每条登记都从第 0 行开始，所以一次请求
#   能复用的行数上界就是「所有登记中公共前缀最长的那一条」——跨两个段拼接永远不可能
#   更长（短的那份必然被长的那份覆盖）。所以这里不去实现测不出收益的“两轮扫描”，
#   而是把真正能多拿行数的方向（把生成 token 的行也登记）做成第 2 组并实测。
param(
  [int]$Port = 30025,
  [string]$Model = "..\models\qwen05",
  [string]$MmModel = "..\models\mm-test",
  [string]$Exe = "build-cuda\bin\Release\cuda_http.exe",
  [string]$CUDA = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin"
)
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $PSScriptRoot
Set-Location $here
$env:PATH = $CUDA + ';' + $env:PATH
$base = "http://127.0.0.1:$Port"
$fails = 0
function Check($name, $ok, $detail) {
  if ($ok) { Write-Output ("  {0,-26} PASS  {1}" -f $name, $detail) }
  else { Write-Output ("  {0,-26} FAIL  {1}" -f $name, $detail); $script:fails++ }
}

$tmp = Join-Path $env:TEMP "pfx15"
New-Item -ItemType Directory -Force $tmp | Out-Null

function Start-Engine([string]$model, [hashtable]$envs) {
  $log = Join-Path $tmp ("srv_{0}.log" -f $Port)
  foreach ($k in $envs.Keys) { Set-Item -Path ("env:" + $k) -Value $envs[$k] }
  $p = Start-Process -FilePath (Resolve-Path $Exe) -ArgumentList @($model, "$Port") `
       -RedirectStandardOutput $log -RedirectStandardError "$log.err" -PassThru -WindowStyle Hidden
  foreach ($k in @($envs.Keys)) { Remove-Item ("env:" + $k) -ErrorAction SilentlyContinue }
  for ($i = 0; $i -lt 400; $i++) {
    Start-Sleep -Milliseconds 250
    if ((Get-Content $log -Raw -ErrorAction SilentlyContinue) -match 'serving on') { return @($p, $log) }
    if ($p.HasExited) { break }
  }
  throw ("engine did not start; see {0}{1}" -f $log, (Get-Content "$log.err" -Raw -ErrorAction SilentlyContinue))
}
function Stop-Engine($r) { Stop-Process -Id $r[0].Id -Force -ErrorAction SilentlyContinue; Start-Sleep -Milliseconds 700 }

function Health() { return (curl.exe -s "$base/health" | ConvertFrom-Json) }
# 请求返回后 worker 还要一轮才 reap（登记租让），所以命中计数要等一下
function HitsAfter() {
  Start-Sleep -Milliseconds 350
  for ($i = 0; $i -lt 20; $i++) {
    try { return [long](Health).prefix.hit_rows } catch { Start-Sleep -Milliseconds 200 }
  }
  return -1
}

function Gen([string]$text, [int]$maxTok) {
  $f = Join-Path $tmp ("g_{0}.json" -f ([guid]::NewGuid().ToString('N')))
  (@{ text = $text; max_tokens = $maxTok; temperature = 0.0 } | ConvertTo-Json -Compress) |
    Set-Content -Path $f -Encoding ascii
  $o = curl.exe -s -m 300 -w "`n%{time_total}" -X POST "$base/generate" `
       -H 'Content-Type: application/json' --data-binary "@$f"
  Remove-Item $f -Force -ErrorAction SilentlyContinue
  $lines = $o -split "`n"
  $t = 0.0; try { $t = [double]($lines[-1]) } catch { }
  $json = ($lines[0..($lines.Length - 2)] -join '')
  $txt = $null
  try { $txt = [string](($json | ConvertFrom-Json).completion) } catch { $txt = "<bad json: $json>" }
  if ($null -eq $txt) { $txt = '' }
  return [pscustomobject]@{ time = $t; text = $txt }
}

function ChatMm([string]$text, [string]$imgPath, [int]$maxTok) {
  $b64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes((Resolve-Path $imgPath)))
  $body = @{
    messages = @(@{ role = 'user'; content = @(
      @{ type = 'text'; text = $text },
      @{ type = 'image_url'; image_url = @{ url = "data:image/png;base64,$b64" } }) })
    max_tokens = $maxTok; temperature = 0.0
  } | ConvertTo-Json -Depth 8 -Compress
  $f = Join-Path $tmp ("mm_{0}.json" -f ([guid]::NewGuid().ToString('N')))
  Set-Content -Path $f -Value $body -Encoding ascii
  $o = curl.exe -s -m 300 -w "`n%{http_code}" -X POST "$base/v1/chat/completions" `
       -H 'Content-Type: application/json' --data-binary "@$f"
  Remove-Item $f -Force -ErrorAction SilentlyContinue
  $lines = $o -split "`n"
  $code = 0; try { $code = [int]($lines[-1]) } catch { }
  $json = ($lines[0..($lines.Length - 2)] -join '')
  $txt = $null
  try { $txt = [string](($json | ConvertFrom-Json).choices[0].message.content) } catch { $txt = "<bad json: $json>" }
  if ($null -eq $txt) { $txt = '' }
  return [pscustomobject]@{ code = $code; text = $txt }
}

# ---------- 语料 ----------
$pfx = (1..40 | ForEach-Object { "The quick brown fox jumps over the lazy dog number $_." }) -join ' '
$s1 = ' Now describe the weather in spring.'
$s2 = ' Now list three kinds of trees.'

Write-Output "== G15 prefix cache e2e (port $Port) =="

# =========== 组 1+2：同前缀 / 续写（先生成无缓存基线，再比缓存路径）===========
$r = Start-Engine $Model @{ SGL_CUDA_CTX = '4096'; SGL_HTTP_CONC = '2'; SGL_PFX_MAX = '0' }
$coldB = Gen ($pfx + $s2) 8
$coldA = Gen ($pfx + $s1) 16
$coldB2 = Gen ($pfx + $s1 + $coldA.text + $s2) 8
Check "组1 关缓存基线可用" ($coldB.text -ne '' -and $coldA.text -ne '') ("B='{0}'" -f $coldB.text.Substring(0, [Math]::Min(20, $coldB.text.Length)))
Stop-Engine $r

$r = Start-Engine $Model @{ SGL_CUDA_CTX = '4096'; SGL_HTTP_CONC = '2' }
$h0 = HitsAfter
$a = Gen ($pfx + $s1) 16
$h1 = HitsAfter
$b = Gen ($pfx + $s2) 8
$h2 = HitsAfter
$hitSame = $h2 - $h1
Check "组1 A 登记了租让" (($h1 -eq $h0) -and ((Health).prefix.leased_rows -gt 32)) ("leased_rows={0}" -f (Health).prefix.leased_rows)
Check "组1 同前缀命中" ($hitSame -gt 32) ("命中 {0} 行（> PFX_MIN=32）" -f $hitSame)
Check "组1 输出与冷启一致" ($b.text -eq $coldB.text) ("warm='{0}'" -f $b.text.Substring(0, [Math]::Min(20, $b.text.Length)))

# 组 2：紧接着再做一次「A → 续写」，续写 prompt = 前缀+S1+A 的生成结果+S2
$a2 = Gen ($pfx + $s1) 16
$h3 = HitsAfter
$b2 = Gen ($pfx + $s1 + $a2.text + $s2) 8
$h4 = HitsAfter
$hitMulti = $h4 - $h3
Check "组2 续写命中 > 同前缀" ($hitMulti -gt $hitSame) ("续写 {0} 行 vs 同前缀 {1} 行（多 {2} 行 = 生成 token 的 KV）" -f $hitMulti, $hitSame, ($hitMulti - $hitSame))
Check "组2 输出与冷启一致" ($b2.text -eq $coldB2.text) ("warm='{0}'" -f $b2.text.Substring(0, [Math]::Min(20, $b2.text.Length)))
Stop-Engine $r

# =========== 组 3：带图（同文本不同图片，复用 text 部分）===========
$imgA = Join-Path $MmModel 'test.png'
$imgB = Join-Path $MmModel 'test.jpg'
if ((Test-Path (Join-Path $MmModel 'clip_config.json')) -and (Test-Path $imgA) -and (Test-Path $imgB)) {
  $mtxt = (1..10 | ForEach-Object { "A short caption about picture number $_ in the set." }) -join ' '
  $r = Start-Engine $MmModel @{ SGL_CUDA_CTX = '2048'; SGL_HTTP_CONC = '2'; SGL_PFX_MAX = '0' }
  $m2c = ChatMm $mtxt $imgB 6
  Stop-Engine $r
  $r = Start-Engine $MmModel @{ SGL_CUDA_CTX = '2048'; SGL_HTTP_CONC = '2' }
  $hh0 = HitsAfter
  $m1 = ChatMm $mtxt $imgA 6
  $hh1 = HitsAfter
  $m2 = ChatMm $mtxt $imgB 6
  $hh2 = HitsAfter
  Check "组3 视觉请求已登记" ($m1.code -eq 200 -and (Health).prefix.leased_rows -gt 32) ("http={0} leased_rows={1}" -f $m1.code, (Health).prefix.leased_rows)
  Check "组3 同文本不同图命中 text" (($hh2 - $hh1) -gt 32) ("命中 {0} 行" -f ($hh2 - $hh1))
  Check "组3 输出与关缓存一致" ($m2.text -eq $m2c.text -and $m2.text -ne '') ("warm='{0}'" -f $m2.text.Substring(0, [Math]::Min(20, $m2.text.Length)))
  Stop-Engine $r
} else { Check "组3 带图" $false "缺 $MmModel 的 clip_config.json / test.png / test.jpg" }

# =========== 组 4：显存紧张（一段装不下长 prompt）===========
$r = Start-Engine "..\models\mistral-rand" @{ SGL_CUDA_CTX = '512'; SGL_HTTP_CONC = '8' }
$hs = Health
$longP = (1..40 | ForEach-Object { "overflow token number $_ ." }) -join ' '
$o1 = Gen $longP 4
Start-Sleep -Milliseconds 350
$hAfter = Health
Check "组4 超容量被拒绝" ($o1.text -eq '' -or $o1.text -match 'error|bad') ("completion='{0}'" -f $o1.text.Substring(0, [Math]::Min(40, $o1.text.Length)))
Check "组4 触发租让回收" ([long]$hAfter.prefix.flushes -ge 1) ("flushes={0}（seg={1}）" -f $hAfter.prefix.flushes, $hAfter.seg)
$o2 = Gen 'Short prompt that still fits.' 4
$o2b = Gen 'Short prompt that still fits.' 4
Check "组4 后续小请求仍正确" (($o2.text -ne '') -and ($o2.text -eq $o2b.text)) ("'{0}' vs '{1}'" -f $o2.text, $o2b.text)
Stop-Engine $r

Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
if ($fails -eq 0) { Write-Output 'prefix_e2e: ALL PASS (4 组)'; exit 0 }
else { Write-Output ("prefix_e2e: {0} FAIL" -f $fails); exit 1 }