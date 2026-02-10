# Diarize Installation — Windows (PowerShell) + macOS (zsh)

This sets up **faster-whisper + pyannote community-1** for speaker-attributed
transcription on macOS.

Prerequisites:
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

Follow the same steps for account creation, accepting model terms, and creating a token.

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

## Known issues

### m4a / AAC files

`libsndfile` (used by torchaudio's soundfile backend) cannot decode m4a/AAC.
The script auto-converts to 16 kHz mono WAV via ffmpeg before diarization.
Make sure ffmpeg is on your PATH.
