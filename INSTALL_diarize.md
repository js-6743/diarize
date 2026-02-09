# Diarize Installation — Windows (PowerShell) + macOS (zsh)

This sets up **faster-whisper + pyannote community-1** for speaker-attributed
transcription on Windows or macOS.

Prerequisites:
- Windows: PowerShell, Python 3.12, ffmpeg, NVIDIA GPU with recent drivers
- macOS: zsh, Python 3.11+, ffmpeg (CPU-only by default)

---

# macOS (zsh)

## 1. Install dependencies

```bash
brew install ffmpeg
```

Verify:

```bash
ffmpeg -version
python3 --version
```

---

## 2. Create project folders

```bash
proj=\"$HOME/Documents/Python/diarize\"

mkdir -p \"$proj/input\" \"$proj/out\" \"$proj/models\" \"$proj/scripts\"
```

---

## 3. Create Python virtual environment

```bash
python3 -m venv \"$HOME/Documents/Python/envs/diarize\"
source \"$HOME/Documents/Python/envs/diarize/bin/activate\"
python --version
```

---

## 4. Install pyannote.audio and faster-whisper

```bash
python -m pip install -U pip
pip install pyannote.audio
pip install faster-whisper
```

---

## 5. Set up HuggingFace access (required for pyannote)

Follow the same steps as Windows (account, accept terms, create token).

Create `.hf_token` in your project root:

```bash
printf \"%s\" \"hf_YOUR_TOKEN_HERE\" > \"$proj/.hf_token\"
```

---

## 6. Copy scripts

Copy these files into your scripts folder:

```
~/Documents/Python/diarize/scripts/
    diarize.py
    transcribe.sh
    run_diarize.sh
```

---

## 7. Copy input data

Copy your batch folder into `input/`. Example:

```
~/Documents/Python/diarize/input/Expert_03/
    Expert Interview - Vorlage.docx
    Expert_03_Recording_1.m4a
    Expert_03_Recording_2.m4a
    expert_interview_context.txt
```

---

## 8. Test

```bash
python -c \"from faster_whisper import WhisperModel; print('faster-whisper OK')\"
python -c \"from pyannote.audio import Pipeline; print('pyannote OK')\"
```

Then run a batch:

```bash
cd \"$HOME/Documents/Python/diarize/scripts\"
./run_diarize.sh Expert_03 --num-speakers 2
```

First run downloads models (~3 GB for large-v3, ~500 MB for pyannote).

---

# Windows (PowerShell)

---

## 1. Check NVIDIA driver

```powershell
nvidia-smi
```

Note the **Driver Version** and **CUDA Version** at the top.
You need Driver ≥ 528 (any driver from 2023+ is fine).

---

## 2. Create project folders

```powershell
$proj = "$env:USERPROFILE\Documents\Python\diarize"

mkdir $proj
mkdir $proj\input
mkdir $proj\out
mkdir $proj\models
mkdir $proj\scripts
```

---

## 3. Create Python virtual environment

```powershell
cd $env:USERPROFILE\Documents\Python\envs
python -m venv diarize
.\diarize\Scripts\Activate.ps1
python --version
```

---

## 4. Install pyannote.audio and faster-whisper

Install pyannote first — it pins specific torch versions:

```powershell
python -m pip install -U pip
pip install pyannote.audio
pip install faster-whisper
```

---

## 5. Reinstall PyTorch with CUDA

The previous step installs CPU-only PyTorch. Replace it with a CUDA build.

**Important:** pyannote.audio 4.x pins `torch==2.8.0`. The cu124 index does
not carry 2.8.0 yet, so use **cu126**:

```powershell
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126 --force-reinstall --no-deps
```

Verify GPU access:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Should print `2.8.0+cu126 True` and your GPU name.

---

## 6. Set up HuggingFace access (required for pyannote)

The pyannote community-1 model is free but gated. One-time setup:

### 6a. Create HuggingFace account

Go to <https://huggingface.co/join> and register.

### 6b. Accept model terms

Visit **both** pages and click "Agree and access repository":

- <https://huggingface.co/pyannote/speaker-diarization-community-1>
- <https://huggingface.co/pyannote/segmentation-3.0>

(You fill in name / affiliation / use case — access is instant.)

### 6c. Create access token

Go to <https://huggingface.co/settings/tokens> → **Create new token** →
give it a name like `diarize` → type **Read** → copy the token.

### 6d. Store the token

Create a file called `.hf_token` in your project root:

```powershell
$proj = "$env:USERPROFILE\Documents\Python\diarize"
Set-Content -Path "$proj\.hf_token" -Value "hf_YOUR_TOKEN_HERE" -NoNewline
```

Replace `hf_YOUR_TOKEN_HERE` with your actual token.

> Alternative: set an environment variable instead:
> `$env:HF_TOKEN = "hf_YOUR_TOKEN_HERE"`

---

## 7. Copy scripts

Copy both files into the scripts folder:

```
C:\Users\josef\Documents\Python\diarize\scripts\
    diarize.py
    transcribe.ps1
```

---

## 8. Copy input data

Copy your batch folder into `input\`. Example:

```
C:\Users\josef\Documents\Python\diarize\input\Expert_03\
    Expert Interview - Vorlage.docx
    Expert_03_Recording_1.m4a
    Expert_03_Recording_2.m4a
    expert_interview_context.txt
```

---

## 9. Test

With the venv activated:

```powershell
python -c "from faster_whisper import WhisperModel; print('faster-whisper OK')"
python -c "from pyannote.audio import Pipeline; print('pyannote OK')"
```

Then run a batch:

```powershell
cd $env:USERPROFILE\Documents\Python\diarize\scripts
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\transcribe.ps1 Expert_03 -NumSpeakers 2
```

First run downloads models (~3 GB for large-v3, ~500 MB for pyannote).

---

## Quick reference: folder layout

```
C:\Users\josef\Documents\Python\
    envs\
        diarize\                ← venv
    diarize\                    ← project root
        .hf_token               ← HuggingFace token
        input\
            Expert_03\
                *.m4a
                expert_interview_context.txt
        out\
            Expert_03\
                Expert_03_Recording_1\
                    *.txt  *.json  *.srt          (combined)
                    *_SPEAKER_00.txt/json/srt     (per speaker)
                    *_SPEAKER_01.txt/json/srt
                Expert_03_Recording_2\
                    ...
                _log.txt
        models\                 ← whisper model cache
        scripts\
            diarize.py
            transcribe.ps1
```

---

## Known issues

### pyannote overrides CUDA torch with CPU-only torch

`pip install pyannote.audio` pins `torch==2.8.0` and pulls the CPU-only
wheel from PyPI, replacing any CUDA build you installed earlier. This is
why step 5 must come **after** step 4. The `--force-reinstall --no-deps`
flags are required because pip considers the version "already satisfied"
even though the CPU/CUDA variant differs.

### torchcodec warning

pyannote 4.x ships torchcodec as a dependency but it does not work on
Windows (missing FFmpeg DLLs at the expected paths). The script works
around this by preloading audio via torchaudio/soundfile instead.
The warning is suppressed in `diarize.py`.

### m4a / AAC files

`libsndfile` (used by torchaudio's soundfile backend) cannot decode m4a/AAC.
The script auto-converts to 16 kHz mono WAV via ffmpeg before diarization.
Make sure ffmpeg is on your PATH.
