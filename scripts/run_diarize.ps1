<#
run_diarize.ps1  (stable wrapper)

Fixes:
- Activates the diarize venv for THIS session
- Optionally sets FFMPEG_BIN (and PATH) for this session
- Runs transcribe.ps1 in a *fresh PowerShell process* (pwsh -NoProfile -File ...),
  so parameter binding behaves exactly like a direct invocation.

Wrapper-only flags (optional):
  -VenvRoot  <path>   (default: %USERPROFILE%\Documents\Python\envs\diarize)
  -FfmpegBin <path>   (folder containing ffmpeg.exe; if omitted, we auto-detect ffmpeg from PATH)

Everything else is forwarded to transcribe.ps1 unchanged.
Example:
  .\run_diarize.ps1 Biweekly -NumSpeakers 3 -Extensions wav
#>

$ErrorActionPreference = "Stop"

function Show-Usage {
  Write-Host "Usage:"
  Write-Host "  .\run_diarize.ps1 <Batch> [wrapper flags] [transcribe.ps1 flags]"
  Write-Host ""
  Write-Host "Wrapper flags:"
  Write-Host "  -VenvRoot  <path>   (default: %USERPROFILE%\Documents\Python\envs\diarize)"
  Write-Host "  -FfmpegBin <path>   (optional; folder containing ffmpeg.exe)"
  Write-Host ""
  Write-Host "Examples:"
  Write-Host "  .\run_diarize.ps1 Biweekly -NumSpeakers 3 -Extensions wav"
  Write-Host "  .\run_diarize.ps1 Biweekly -FfmpegBin ""C:\path\to\ffmpeg\bin"" -NumSpeakers 3"
}

if ($args.Count -lt 1 -or $args[0] -in @("-h","--help","-?","/?")) {
  Show-Usage
  exit 1
}

$Batch = [string]$args[0]

# Defaults
$VenvRoot  = "$env:USERPROFILE\Documents\Python\envs\diarize"
$FfmpegBin = ""

# Pass-through to transcribe.ps1
$PassThru = @()

# Parse wrapper-only flags; everything else goes through
for ($i = 1; $i -lt $args.Count; $i++) {
  $tok = [string]$args[$i]

  if ($tok -ieq "-VenvRoot" -or $tok -ieq "-VenvPath") {
    if ($i + 1 -ge $args.Count) { throw "Missing value after $tok" }
    $VenvRoot = [string]$args[$i + 1]
    $i++
    continue
  }
  if ($tok -match '^(?i)\-(VenvRoot|VenvPath):(.*)$') {
    $VenvRoot = $Matches[2]
    continue
  }

  if ($tok -ieq "-FfmpegBin") {
    if ($i + 1 -ge $args.Count) { throw "Missing value after -FfmpegBin" }
    $FfmpegBin = [string]$args[$i + 1]
    $i++
    continue
  }
  if ($tok -match '^(?i)\-FfmpegBin:(.*)$') {
    $FfmpegBin = $Matches[1]
    continue
  }

  $PassThru += $tok
}

# Locate scripts
$transcribe = Join-Path $PSScriptRoot "transcribe.ps1"
if (!(Test-Path $transcribe)) { throw "transcribe.ps1 not found next to run_diarize.ps1: $transcribe" }

# Resolve venv
if (!(Test-Path $VenvRoot)) { throw "VenvRoot does not exist: $VenvRoot" }
$VenvRoot = (Resolve-Path $VenvRoot).Path
$activate  = Join-Path $VenvRoot "Scripts\Activate.ps1"
$pythonExe = Join-Path $VenvRoot "Scripts\python.exe"

if (!(Test-Path $activate))  { throw "venv Activate.ps1 not found: $activate" }
if (!(Test-Path $pythonExe)) { throw "venv python.exe not found: $pythonExe" }

Write-Host "Activating venv: $VenvRoot"
. $activate | Out-Null
Write-Host "Using Python: $(& $pythonExe -c 'import sys; print(sys.executable)')"

# Ffmpeg: auto-detect if not provided
if (-not $FfmpegBin -or $FfmpegBin.Trim() -eq "") {
  $ff = Get-Command ffmpeg -ErrorAction SilentlyContinue
  if ($ff -and $ff.Source) {
    $FfmpegBin = Split-Path $ff.Source -Parent
  }
}

if ($FfmpegBin -and $FfmpegBin.Trim() -ne "") {
  if (Test-Path $FfmpegBin) {
    $FfmpegBin = (Resolve-Path $FfmpegBin).Path
    $env:FFMPEG_BIN = $FfmpegBin
    $env:Path = "$FfmpegBin;$env:Path"
    if (Test-Path (Join-Path $FfmpegBin "ffmpeg.exe")) {
      Write-Host "FFMPEG_BIN: $env:FFMPEG_BIN"
    } else {
      Write-Host "WARNING: FfmpegBin set, but ffmpeg.exe not found in: $FfmpegBin" -ForegroundColor Yellow
    }
  } else {
    Write-Host "WARNING: -FfmpegBin path does not exist (ignored): $FfmpegBin" -ForegroundColor Yellow
  }
}

# Choose shell for the child process
$childShell = $null
$pwshCmd = Get-Command pwsh -ErrorAction SilentlyContinue
if ($pwshCmd -and $pwshCmd.Source) {
  $childShell = $pwshCmd.Source
} else {
  $psCmd = Get-Command powershell -ErrorAction SilentlyContinue
  if ($psCmd -and $psCmd.Source) {
    $childShell = $psCmd.Source
  }
}
if (-not $childShell) {
  throw "Neither pwsh nor powershell was found on PATH."
}

# Run transcribe in a fresh process (stable parameter binding)
# Environment (VIRTUAL_ENV, PATH, FFMPEG_BIN, etc.) is inherited by the child.
$childArgs = @(
  "-NoProfile",
  "-ExecutionPolicy", "Bypass",
  "-File", $transcribe,
  $Batch
) + $PassThru

# Ensure we run from the scripts folder for predictable relative paths
Push-Location $PSScriptRoot
try {
  & $childShell @childArgs
  exit $LASTEXITCODE
}
finally {
  Pop-Location
}
