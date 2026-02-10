# diarize/scripts (macOS Bash)

This folder contains a working pipeline for **speaker diarization + transcription**:

- `diarize.py` – Python pipeline (pyannote community-1 + faster-whisper alignment) fileciteturn9file0
- `transcribe.sh` – macOS/Linux batch runner that scans `../input/<Batch>/`
- `run_diarize.sh` – macOS/Linux wrapper: activates the diarize venv + (optionally) sets FFmpeg, then runs `transcribe.sh`

---

## Install / Replace

Copy these files into:

macOS:

`~/Documents/Python/diarize/scripts/`

and **replace existing files**:

- `run_diarize.sh`
- `transcribe.sh`
- `diarize.py`
- `README.md` (this file)

---

## Recommended run

```bash
cd ~/Documents/Python/diarize/scripts
./run_diarize.sh Biweekly --num-speakers 3 --extensions wav
```

### macOS (zsh)

```bash
cd ~/Documents/Python/diarize/scripts
./run_diarize.sh Biweekly --num-speakers 3 --extensions wav
```

You will see:

- `Activating venv: ...`
- `Using Python: ...`
- (optional) `FFMPEG_BIN: ...`

Then the normal `transcribe.sh` output.

---

## FFmpeg

You usually do **not** need to pass `--ffmpeg-bin` if `which ffmpeg` already works.  
`run_diarize.sh` will auto-detect `ffmpeg` from PATH and set `FFMPEG_BIN` accordingly.

To force a specific ffmpeg folder (the folder that contains `ffmpeg`):

```bash
./run_diarize.sh Biweekly --ffmpeg-bin "/opt/homebrew/bin" --num-speakers 3
```

If the path doesn’t exist, the wrapper prints a warning and continues.

On macOS/Linux you can pass the folder that contains `ffmpeg`:

```bash
./run_diarize.sh Biweekly --ffmpeg-bin "/opt/homebrew/bin" --num-speakers 3
```

---

## Notes

- Your shell prompt may show `base` or `diarize` depending on your prompt theme and whether conda is active.
  The wrapper prints the actual `python` path used, which is the reliable signal.
