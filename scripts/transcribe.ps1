param(
  [Parameter(Mandatory=$true, Position=0)]
  [string]$Batch,

  # Project root. Default assumes:
  #   %USERPROFILE%\Documents\Python\diarize
  [string]$ProjectRoot = (Join-Path $env:USERPROFILE "Documents\Python\diarize"),

  # Python exe to run diarize.py (default: diarize venv)
  [string]$PythonExe = (Join-Path $env:USERPROFILE "Documents\Python\envs\diarize\Scripts\python.exe"),

  # faster-whisper model
  [string]$Model = "large-v3",

  # faster-whisper language code (recommended: de | en)
  [string]$Language = "de",

  [ValidateSet("cuda","cpu")]
  [string]$Device = "cuda",

  # CTranslate2 compute type for faster-whisper
  #   float16 | int8_float16 | int8 | float32
  [ValidateSet("float16","int8_float16","int8","float32")]
  [string]$ComputeType = "float16",

  [ValidateSet("txt","json","srt","all")]
  [string]$OutputFormat = "all",

  # Diarization speaker constraints (optional)
  [int]$NumSpeakers,
  [int]$MinSpeakers,
  [int]$MaxSpeakers,

  # Context prompting (optional)
  # If -NoContext is NOT set, we auto-load:
  #   <input\<Batch>\expert_interview_context.txt>
  # You can override with -ContextFile (absolute or relative to batch folder)
  [string]$ContextFile = "expert_interview_context.txt",
  [switch]$NoContext,

  # HuggingFace token (optional). If not provided, diarize.py will try:
  #   HF_TOKEN env var  OR  <ProjectRoot>\.hf_token
  [string]$HfToken = "",

  # Include subfolders under input\<Batch>
  [switch]$Recurse,

  # File types to include (extensions without dot)
  [string[]]$Extensions = @("wav","mp3","m4a","mp4","flac","aac","ogg","webm","mkv"),

  # If set, skip a file when the expected output already exists
  [switch]$SkipExisting
)

$ErrorActionPreference = "Stop"

# ── paths ───────────────────────────────────────────────────────────
$InputDir  = Join-Path $ProjectRoot "input\$Batch"
$OutRoot   = Join-Path $ProjectRoot "out\$Batch"
$ModelDir  = Join-Path $ProjectRoot "models"
$DiarizePy = Join-Path $PSScriptRoot "diarize.py"

if (!(Test-Path $InputDir))  { throw "Input batch folder not found: $InputDir" }
if (!(Test-Path $DiarizePy)) { throw "diarize.py not found next to this script: $DiarizePy" }

New-Item -ItemType Directory -Force -Path $OutRoot  | Out-Null
New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null

if (!(Test-Path $PythonExe)) {
  throw "PythonExe not found: $PythonExe`nSet -PythonExe to your venv python.exe (e.g. ...\\envs\\diarize\\Scripts\\python.exe)."
}

$LogPath = Join-Path $OutRoot "_log.txt"
"=== Diarize batch: $Batch ===" | Out-File -FilePath $LogPath -Encoding UTF8
("Start: " + (Get-Date)) | Out-File -FilePath $LogPath -Append -Encoding UTF8
("ProjectRoot: $ProjectRoot") | Out-File -FilePath $LogPath -Append -Encoding UTF8
("PythonExe: $PythonExe") | Out-File -FilePath $LogPath -Append -Encoding UTF8
("Model: $Model | Language: $Language | Device: $Device | ComputeType: $ComputeType | OutputFormat: $OutputFormat") |
  Out-File -FilePath $LogPath -Append -Encoding UTF8

# ── context prompt ──────────────────────────────────────────────────
$InitialPrompt = $null
if (-not $NoContext) {
  $ctxPath = $null

  if ($ContextFile -and $ContextFile.Trim() -ne "") {
    if (Split-Path $ContextFile -IsAbsolute) {
      $ctxPath = $ContextFile
    } else {
      $ctxPath = Join-Path $InputDir $ContextFile
    }
  }

  if ($ctxPath -and (Test-Path $ctxPath)) {
    $InitialPrompt = (Get-Content -Path $ctxPath -Raw -Encoding UTF8).Trim()
    if ($InitialPrompt -ne "") {
      # Normalise whitespace so it stays one CLI arg
      $InitialPrompt = ($InitialPrompt -replace "\s+", " ").Trim()
      ("Context: ON  ($ctxPath)") | Out-File -FilePath $LogPath -Append -Encoding UTF8
    } else {
      $InitialPrompt = $null
      ("Context: OFF (empty file)") | Out-File -FilePath $LogPath -Append -Encoding UTF8
    }
  } else {
    ("Context: OFF") | Out-File -FilePath $LogPath -Append -Encoding UTF8
  }
} else {
  ("Context: OFF  (-NoContext)") | Out-File -FilePath $LogPath -Append -Encoding UTF8
}

# ── collect media files (FORCE ARRAY so .Count always works) ─────────
$patterns = $Extensions | ForEach-Object { "*.$_" }
$files = @()
foreach ($pat in $patterns) {
  $files += Get-ChildItem -Path $InputDir -File -Filter $pat -ErrorAction SilentlyContinue -Recurse:$Recurse
}
$files = @($files | Sort-Object FullName -Unique)

if ($files.Count -eq 0) {
  Write-Host "No media files found in $InputDir"
  "No media files found." | Out-File -FilePath $LogPath -Append -Encoding UTF8
  exit 0
}

Write-Host "Found $($files.Count) media file(s)"
"Files: $($files.Count)" | Out-File -FilePath $LogPath -Append -Encoding UTF8

# Helpful warning for the common case: you kept both the original video and an extracted wav.
if ($files.Count -gt 1) {
  Write-Host "NOTE: Multiple media files found. The script will process ALL of them (one output folder per file)." -ForegroundColor Yellow
  Write-Host "      If you only want ONE (e.g. only the .wav), either remove the others or run with -Extensions wav." -ForegroundColor Yellow
}

# Group by folder to implement: single file → output directly; multiple → one subfolder per file
$inputFull = (Resolve-Path $InputDir).Path
$groups = $files | Group-Object DirectoryName

foreach ($g in $groups) {
  $groupDir  = $g.Name
  $groupFiles = @($g.Group)   # force array
  $groupN = $groupFiles.Count

  # mirror folder structure relative to InputDir
  $dirFull = (Resolve-Path $groupDir).Path
  $rel = ""
  if ($dirFull.Length -ge $inputFull.Length -and $dirFull.Substring(0, $inputFull.Length).ToLower() -eq $inputFull.ToLower()) {
    $rel = $dirFull.Substring($inputFull.Length).TrimStart("\\")
  }
  $outBase = if ($rel -and $rel.Trim() -ne "") { Join-Path $OutRoot $rel } else { $OutRoot }
  New-Item -ItemType Directory -Force -Path $outBase | Out-Null

  foreach ($f in $groupFiles) {
    $targetOut = if ($groupN -gt 1) { Join-Path $outBase $f.BaseName } else { $outBase }
    New-Item -ItemType Directory -Force -Path $targetOut | Out-Null

    # Skip existing outputs (based on combined txt/json/srt)
    if ($SkipExisting) {
      $expected = @()
      if ($OutputFormat -eq "all" -or $OutputFormat -eq "txt")  { $expected += (Join-Path $targetOut "$($f.BaseName).txt") }
      if ($OutputFormat -eq "all" -or $OutputFormat -eq "json") { $expected += (Join-Path $targetOut "$($f.BaseName).json") }
      if ($OutputFormat -eq "all" -or $OutputFormat -eq "srt")  { $expected += (Join-Path $targetOut "$($f.BaseName).srt") }
      if ($expected | Where-Object { Test-Path $_ }) {
        Write-Host "SKIP (exists): $($f.Name)"
        ("SKIP (exists): " + $f.FullName) | Out-File -FilePath $LogPath -Append -Encoding UTF8
        continue
      }
    }

    Write-Host "────────────────────────────────────────"
    Write-Host "File:    $($f.FullName)"
    Write-Host "Output:  $targetOut"
    Write-Host ("Context: " + ($(if ($InitialPrompt) { "ON" } else { "OFF" })))
    ("File: $($f.FullName)") | Out-File -FilePath $LogPath -Append -Encoding UTF8
    ("Out:  $targetOut") | Out-File -FilePath $LogPath -Append -Encoding UTF8

    $args = @(
      $DiarizePy,
      $f.FullName,
      "--output_dir", $targetOut,
      "--model", $Model,
      "--language", $Language,
      "--device", $Device,
      "--compute_type", $ComputeType,
      "--output_format", $OutputFormat,
      "--model_dir", $ModelDir,
      "--project_root", $ProjectRoot
    )

    if ($NumSpeakers) { $args += @("--num_speakers", $NumSpeakers) }
    if ($MinSpeakers) { $args += @("--min_speakers", $MinSpeakers) }
    if ($MaxSpeakers) { $args += @("--max_speakers", $MaxSpeakers) }
    if ($InitialPrompt) { $args += @("--initial_prompt", $InitialPrompt) }
    if ($HfToken -and $HfToken.Trim() -ne "") { $args += @("--hf_token", $HfToken) }

    # Run
    $t0 = Get-Date
    & $PythonExe @args
    $exit = $LASTEXITCODE
    $dt = New-TimeSpan -Start $t0 -End (Get-Date)

    if ($exit -ne 0) {
      Write-Host "FAILED (exit $exit): $($f.Name)" -ForegroundColor Red
      ("FAILED (exit $exit) in $($dt.ToString()))") | Out-File -FilePath $LogPath -Append -Encoding UTF8
      continue
    }

    Write-Host "OK: $($f.Name)  ($($dt.ToString()))"
    ("OK in $($dt.ToString()))") | Out-File -FilePath $LogPath -Append -Encoding UTF8
  }
}

("End: " + (Get-Date)) | Out-File -FilePath $LogPath -Append -Encoding UTF8
Write-Host "════════════════════════════════════════"
Write-Host "Done. Output: $OutRoot"
Write-Host "Log:    $LogPath"
