# Diarized transcription (PowerShell + Bash + Python)

Speaker-attributed transcription of German audio/video using
**faster-whisper large-v3** (ASR) and **pyannote community-1** (diarization).

> See `INSTALL_diarize.md` for first-time setup.

---

## Folder structure

Windows example:

```
C:\Users\josef\Documents\Python\diarize\
    .hf_token
    input\          # place audio/video files here
    out\            # transcription results (per batch)
    models\         # whisper model cache (automatic)
    scripts\
        diarize.py
        transcribe.ps1
        run_diarize.ps1
```

macOS example:

```
~/Documents/Python/diarize/
    .hf_token
    input/          # place audio/video files here
    out/            # transcription results (per batch)
    models/         # whisper model cache (automatic)
    scripts/
        diarize.py
        transcribe.sh
        run_diarize.sh
```

Each run processes **one batch folder** under `input\`.

---

## Run transcription

### Windows (PowerShell)

```powershell
cd C:\Users\josef\Documents\Python\diarize\scripts
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

.\run_diarize.ps1 Expert_03 -NumSpeakers 2
```

### macOS (zsh)

```bash
cd ~/Documents/Python/diarize/scripts
./run_diarize.sh Expert_03 --num-speakers 2
```

If you have subfolders under the batch:

```powershell
.\run_diarize.ps1 Expert_03 -NumSpeakers 2 -Recurse
```

```bash
./run_diarize.sh Expert_03 --num-speakers 2 --recurse
```

---

## Output

For each audio file you get **combined** files (all speakers) plus
**per-speaker** files:

```
out\Expert_03\
    Expert_03_Recording_1\
        Expert_03_Recording_1.txt             ← all speakers
        Expert_03_Recording_1.json
        Expert_03_Recording_1.srt
        Expert_03_Recording_1_SPEAKER_00.txt  ← speaker 00 only
        Expert_03_Recording_1_SPEAKER_00.json
        Expert_03_Recording_1_SPEAKER_00.srt
        Expert_03_Recording_1_SPEAKER_01.txt  ← speaker 01 only
        Expert_03_Recording_1_SPEAKER_01.json
        Expert_03_Recording_1_SPEAKER_01.srt
    Expert_03_Recording_2\
        ...
    _log.txt
```

### File formats

| File | Content |
|------|---------|
| `*.txt` | Transcript with speaker labels and timestamps |
| `*.json` | Structured data: utterances, word-level timestamps, speaker timeline |
| `*.srt` | Subtitles with speaker labels |

### Example `.txt` output

```
[00:00:02 – 00:00:56]  SPEAKER_01:
Gut, also, ich wechsle jetzt auch auf Hochdeutsch, damit die
Transkription einfacher ist...

[00:00:57 – 00:00:59]  SPEAKER_00:
Ich bin 45.
```

### Output folder rules (same as whisper setup)

- **1 media file in folder** → output goes directly into mirrored folder
- **>1 media file in folder** → each file gets its own subfolder

---

## Common options

### Model / device

```powershell
.\run_diarize.ps1 Expert_03 -Model "large-v3"
.\run_diarize.ps1 Expert_03 -Device "cpu"
.\run_diarize.ps1 Expert_03 -ComputeType "int8_float16"   # faster, slightly less accurate
.\run_diarize.ps1 Expert_03 -OutputFormat "txt"
```

```bash
./run_diarize.sh Expert_03 --model "large-v3"
./run_diarize.sh Expert_03 --device "cpu"
./run_diarize.sh Expert_03 --compute-type "int8"
./run_diarize.sh Expert_03 --output-format "txt"
```

### Speaker hints

If you know how many speakers are in the recording:

```powershell
.\run_diarize.ps1 Expert_03 -NumSpeakers 2
```

```bash
./run_diarize.sh Expert_03 --num-speakers 2
```

Or provide a range:

```powershell
.\run_diarize.ps1 Expert_03 -MinSpeakers 2 -MaxSpeakers 4
```

```bash
./run_diarize.sh Expert_03 --min-speakers 2 --max-speakers 4
```

> For expert interviews (interviewer + interviewee), `-NumSpeakers 2`
> gives the best results.

### Context / initial prompt

Works identically to the whisper setup.

```powershell
# Use default context file (expert_interview_context.txt)
.\run_diarize.ps1 Expert_03

# Different context filename
.\run_diarize.ps1 Expert_03 -ContextFile "template_context.txt"

# Disable context entirely
.\run_diarize.ps1 Expert_03 -NoContext
```

```bash
# Use default context file (expert_interview_context.txt)
./run_diarize.sh Expert_03

# Different context filename
./run_diarize.sh Expert_03 --context-file "template_context.txt"

# Disable context entirely
./run_diarize.sh Expert_03 --no-context
```

### Python environment

```powershell
.\run_diarize.ps1 Expert_03 -PythonExe "C:\path\to\envs\diarize\Scripts\python.exe"
```

```bash
./run_diarize.sh Expert_03 --python-exe "/path/to/envs/diarize/bin/python"
```

### HuggingFace token

If you prefer not to use a `.hf_token` file:

```powershell
.\run_diarize.ps1 Expert_03 -HfToken "hf_abc123..."
```

```bash
./run_diarize.sh Expert_03 --hf-token "hf_abc123..."
```

---

## Context files

Whisper's initial prompt helps with domain vocabulary. Keep prompts short:
1–2 lines describing the interview, 10–40 keywords, 3–8 question stems.

Your existing `expert_interview_context.txt` works unchanged.

---

## Pipeline details

For each audio file, `diarize.py` runs three steps:

1. **Diarization** (pyannote community-1): identifies *who* speaks *when*
2. **Transcription** (faster-whisper large-v3): word-level ASR with timestamps
3. **Alignment**: maps each transcribed word to a speaker using temporal overlap

The pyannote model is loaded first, run, then freed from GPU memory before
loading the whisper model. This keeps VRAM usage manageable even on 8 GB cards.

Audio files in formats not supported by libsndfile (m4a, mp4, aac, etc.)
are automatically converted to 16 kHz mono WAV via ffmpeg before diarization.

---

## Troubleshooting

- **"No HuggingFace token found"**
  Create `.hf_token` in project root, set `HF_TOKEN` env var, or pass `-HfToken`.

- **"PythonExe not found"**
  Pass `-PythonExe` with the correct venv path.

- **CUDA out of memory**
  Try `-ComputeType "int8_float16"` or `-Device "cpu"`.

- **First run is slow**
  Expected — downloads whisper model (~3 GB) and pyannote models (~500 MB).
  Cached after first run.

- **Speaker labels are wrong / too many speakers**
  Use `-NumSpeakers 2` for two-person interviews.

- **No media files found**
  Check files are in `input\<Batch>\` and enable `-Recurse` if in subfolders.

- **torch.cuda.is_available() returns False**
  pyannote.audio installs CPU-only torch. Reinstall with CUDA:
  `pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126 --force-reinstall --no-deps`

---

## Differences from the whisper-only setup

| | whisper setup | diarize setup |
|---|---|---|
| ASR engine | openai-whisper | faster-whisper (CTranslate2) |
| Diarization | none | pyannote community-1 |
| Speaker labels | no | yes |
| Per-speaker files | no | yes |
| Word timestamps | segment-level | word-level |
| Speed | 1× | ~4× faster (batched inference) |
| Output | txt/srt/vtt/tsv/json | txt/srt/json (with speakers) |
| venv | `envs\whisper\` | `envs\diarize\` |
| project | `Python\whisper\` | `Python\diarize\` |
