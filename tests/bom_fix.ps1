# bom_fix.ps1 — 给所有 .c/.h/.cu/.ps1 补 UTF-8 BOM（edit_file 会吃掉 BOM，铁律 2）
# 用法: powershell -File bom_fix.ps1 [-Root <dir>] [-Check]
param([string]$Root = ".", [switch]$Check)
$ErrorActionPreference = 'Stop'
$exts = @('.c', '.h', '.cu', '.ps1')
$fixed = 0; $missing = @()
Get-ChildItem -LiteralPath $Root -Recurse -File |
  Where-Object { $exts -contains $_.Extension -and $_.FullName -notmatch '\\(build[^\\]*|third_party|\.git|node_modules)(\\|$)' } |
  ForEach-Object {
    $b = [System.IO.File]::ReadAllBytes($_.FullName)
    $has = $b.Length -ge 3 -and $b[0] -eq 0xEF -and $b[1] -eq 0xBB -and $b[2] -eq 0xBF
    if (-not $has) {
      $missing += $_.FullName
      if (-not $Check) {
        $nb = New-Object byte[] ($b.Length + 3)
        $nb[0] = 0xEF; $nb[1] = 0xBB; $nb[2] = 0xBF
        [Array]::Copy($b, 0, $nb, 3, $b.Length)
        [System.IO.File]::WriteAllBytes($_.FullName, $nb)
        $fixed++
      }
    }
  }
if ($Check) {
  Write-Output ("missing BOM: " + $missing.Count)
  $missing | ForEach-Object { Write-Output ("  " + $_) }
} else {
  Write-Output ("BOM restored: " + $fixed + " (checked " + ($missing.Count) + " missing)")
}
