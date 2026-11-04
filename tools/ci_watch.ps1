param([string]$Sha = 'main', [int]$TimeoutSec = 540)
# 轮询一个 GitHub Actions run 到完成并按 job 列结论（发版盯 CI 用）。
# 认证：先 $env:GITHUB_TOKEN = <token>（不落盘，不吃进仓库触发 secret 扫描）。
$tok = $env:GITHUB_TOKEN
if (-not $tok) { Write-Output 'set $env:GITHUB_TOKEN first'; exit 1 }
$h = @{ Authorization = "Bearer $tok" }
$t0 = Get-Date
$run = $null
while (-not $run -and ((Get-Date) - $t0).TotalSeconds -lt 120) {
  $runs = Invoke-RestMethod -Headers $h -Uri 'https://api.github.com/repos/integemjack/sglang.c/actions/runs?per_page=6'
  if ($Sha -eq 'main') { $run = $runs.workflow_runs | Select-Object -First 1 }
  else { $run = $runs.workflow_runs | Where-Object { $_.head_sha -like "$Sha*" } | Select-Object -First 1 }
  if (-not $run) { Start-Sleep -Seconds 15 }
}
if (-not $run) { Write-Output 'RUN NOT FOUND'; exit 1 }
Write-Output "run $($run.id)"
while ($run.status -ne 'completed' -and ((Get-Date) - $t0).TotalSeconds -lt $TimeoutSec) {
  Start-Sleep -Seconds 45
  $run = Invoke-RestMethod -Headers $h -Uri "https://api.github.com/repos/integemjack/sglang.c/actions/runs/$($run.id)"
  $j = Invoke-RestMethod -Headers $h -Uri "https://api.github.com/repos/integemjack/sglang.c/actions/runs/$($run.id)/jobs?per_page=40"
  $pend = @($j.jobs | Where-Object { $_.status -ne 'completed' }).Count
  Write-Output ("[{0:mm}] status={1} pending={2}" -f ((Get-Date) - $t0), $run.status, $pend)
}
$j = Invoke-RestMethod -Headers $h -Uri "https://api.github.com/repos/integemjack/sglang.c/actions/runs/$($run.id)/jobs?per_page=40"
Write-Output "== FINAL: $($run.conclusion) =="
$j.jobs | Sort-Object name | ForEach-Object { '{0,-28} {1}' -f $_.name, $_.conclusion }
