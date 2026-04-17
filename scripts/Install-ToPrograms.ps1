param(
    [string]$InstallRoot = "E:\Programs",
    [string]$AppName = "Gemma4Chat",
    [switch]$Replace
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$sourceDir = Join-Path $repoRoot "dist\$AppName"
$sourceExe = Join-Path $sourceDir "$AppName.exe"
$targetDir = Join-Path $InstallRoot $AppName
$targetExe = Join-Path $targetDir "$AppName.exe"

if (-not (Test-Path $sourceExe)) {
    throw "Build output not found at $sourceExe. Run .\scripts\Build.ps1 first."
}

if (-not (Test-Path $InstallRoot)) {
    New-Item -ItemType Directory -Path $InstallRoot | Out-Null
}

if ((Test-Path $targetDir) -and $Replace) {
    Remove-Item -LiteralPath $targetDir -Recurse -Force
}

if (-not (Test-Path $targetDir)) {
    New-Item -ItemType Directory -Path $targetDir | Out-Null
}

Copy-Item -Path (Join-Path $sourceDir "*") -Destination $targetDir -Recurse -Force

Write-Host ""
Write-Host "Installed $AppName to:"
Write-Host "  $targetDir"
Write-Host ""
Write-Host "Executable:"
Write-Host "  $targetExe"
