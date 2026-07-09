# Project-local UTF-8 defaults for Windows PowerShell 5.1 and PowerShell 7+.
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)

[Console]::InputEncoding = $utf8NoBom
[Console]::OutputEncoding = $utf8NoBom
$OutputEncoding = $utf8NoBom

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

if (Get-Command chcp.com -ErrorAction SilentlyContinue) {
    $null = chcp.com 65001
}

$PSDefaultParameterValues["Get-Content:Encoding"] = "UTF8"
$PSDefaultParameterValues["Set-Content:Encoding"] = "UTF8"
$PSDefaultParameterValues["Add-Content:Encoding"] = "UTF8"
$PSDefaultParameterValues["Out-File:Encoding"] = "UTF8"
$PSDefaultParameterValues["Export-Csv:Encoding"] = "UTF8"
$PSDefaultParameterValues["Import-Csv:Encoding"] = "UTF8"
$PSDefaultParameterValues["Select-String:Encoding"] = "UTF8"
