#!/usr/bin/env python3
"""
Diarized transcription: faster-whisper + pyannote community-1.

This script is designed to be called by your batch wrapper (transcribe.sh).
It performs:

  1) speaker diarization (pyannote community-1, exclusive mode if available)
  2) ASR with word timestamps (faster-whisper / CTranslate2)
  3) alignment: assign a speaker label to each word
  4) rendering: build human-readable utterances and write txt/json/srt

Improvements vs. the "midpoint + split-on-speaker-change" baseline:
  - assign word speakers by maximum overlap with diarization segments (more stable at boundaries)
  - optional smoothing of short A–B–A flips (reduces "tail moved to next speaker" artifacts)
  - optional utterance building by pauses & sentence punctuation + majority-vote speaker
    (reduces mid-sentence speaker flips in the human-readable transcript)

Note:
  - CTranslate2/faster-whisper can crash during model destruction or interpreter shutdown.
    This script avoids those crash paths and exits safely after writing outputs.
"""

# ── suppress known harmless warnings before any imports ──────────────
import warnings
import os

warnings.filterwarnings("ignore", message=r"(?s).*torchcodec is not installed correctly.*")
warnings.filterwarnings("ignore", message=".*this function's implementation will be changed to use torchaudio.*")
warnings.filterwarnings("ignore", message=".*TensorFloat-32.*")
warnings.filterwarnings("ignore", message=".*std\\(\\): degrees of freedom is <= 0.*")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import sys

import argparse
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# ── helpers ──────────────────────────────────────────────────────────

def fmt_ts(seconds: float) -> str:
    """HH:MM:SS"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def fmt_srt(seconds: float) -> str:
    """HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def resolve_hf_token(cli_token: str | None, project_root: str | None) -> str | None:
    """Try CLI arg → env var → .hf_token file."""
    if cli_token:
        return cli_token
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    if project_root:
        p = Path(project_root) / ".hf_token"
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
    return None


def find_speaker(mid: float, timeline: list[dict]) -> str:
    """Return speaker label for a point in time."""
    for seg in timeline:
        if seg["start"] <= mid <= seg["end"]:
            return seg["speaker"]
    # fallback: nearest segment
    best, dist = "UNKNOWN", float("inf")
    for seg in timeline:
        d = min(abs(mid - seg["start"]), abs(mid - seg["end"]))
        if d < dist:
            dist, best = d, seg["speaker"]
    return best


def _is_sentence_end(token: str) -> bool:
    token = token.strip()
    return token.endswith(".") or token.endswith("?") or token.endswith("!")


def load_audio(audio_path: str) -> tuple:
    """
    Load audio via torchaudio; convert m4a/mp4/etc to WAV first via ffmpeg.

    Returns: (waveform, sample_rate, tmp_wav_path_or_None)
      - waveform is Tensor shape (channels, time)
    """
    import torchaudio

    wav_formats = {".wav", ".flac", ".ogg"}
    audio_ext = Path(audio_path).suffix.lower()
    tmp_wav = None

    if audio_ext not in wav_formats:
        tmp_wav = Path(tempfile.gettempdir()) / f"_diarize_{Path(audio_path).stem}.wav"
        print(f"      Converting {audio_ext} → WAV via ffmpeg …")
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path, "-ac", "1", "-ar", "16000",
             "-acodec", "pcm_s16le", str(tmp_wav)],
            check=True, capture_output=True,
        )
        load_path = str(tmp_wav)
    else:
        load_path = audio_path

    waveform, sample_rate = torchaudio.load(load_path)
    return waveform, sample_rate, tmp_wav


def write_txt(filepath: Path, utterances: list[dict]):
    """Write timestamped speaker-labelled transcript."""
    with open(filepath, "w", encoding="utf-8") as f:
        for u in utterances:
            f.write(
                f"[{fmt_ts(u['start'])} – {fmt_ts(u['end'])}]  {u['speaker']}:\n"
                f"{u['text']}\n\n"
            )


def write_srt(filepath: Path, utterances: list[dict]):
    """Write SRT subtitles with speaker labels."""
    with open(filepath, "w", encoding="utf-8") as f:
        for i, u in enumerate(utterances, 1):
            f.write(
                f"{i}\n"
                f"{fmt_srt(u['start'])} --> {fmt_srt(u['end'])}\n"
                f"[{u['speaker']}] {u['text']}\n\n"
            )


def write_json(filepath: Path, utterances: list[dict], *,
               audio_path: str, language: str, model: str,
               speakers_found: list[str], speaker_timeline: list[dict],
               include_word_speakers: bool):
    """Write structured JSON output."""
    blob = {
        "audio": str(audio_path),
        "language": language,
        "model": model,
        "num_speakers": len(speakers_found),
        "speakers": speakers_found,
        "utterances": [
            {
                "speaker": u["speaker"],
                "start": round(u["start"], 3),
                "end": round(u["end"], 3),
                "text": u["text"],
                "words": [
                    {
                        "word": w["word"].strip(),
                        "start": round(w["start"], 3),
                        "end": round(w["end"], 3),
                        "probability": w.get("probability"),
                        **({"speaker": w.get("speaker")} if include_word_speakers else {}),
                    }
                    for w in u["words"]
                ],
            }
            for u in utterances
        ],
        "speaker_timeline": [
            {
                "speaker": s["speaker"],
                "start": round(s["start"], 3),
                "end": round(s["end"], 3),
            }
            for s in speaker_timeline
        ],
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(blob, f, ensure_ascii=False, indent=2)


# ── speaker assignment improvements ──────────────────────────────────

def assign_speakers_midpoint(words: list[dict], timeline: list[dict]) -> None:
    """Baseline: midpoint lookup (kept for A/B comparisons)."""
    for w in words:
        mid = (w["start"] + w["end"]) / 2
        w["speaker"] = find_speaker(mid, timeline)


def assign_speakers_overlap(words: list[dict], timeline: list[dict], *, collar: float = 0.05) -> None:
    """
    Assign speaker per word by maximum overlap with diarization segments.

    collar (sec) slightly expands each word interval to reduce boundary jitter.
    """
    if not words or not timeline:
        return

    segs = sorted(timeline, key=lambda s: (s["start"], s["end"]))
    i = 0
    n = len(segs)

    for w in words:
        ws = w["start"] - collar
        we = w["end"] + collar

        # advance pointer to first segment that could overlap
        while i < n and segs[i]["end"] < ws:
            i += 1

        best_sp = None
        best_ov = 0.0

        j = i
        while j < n and segs[j]["start"] <= we:
            seg = segs[j]
            ov = min(we, seg["end"]) - max(ws, seg["start"])
            if ov > best_ov:
                best_ov = ov
                best_sp = seg["speaker"]
            j += 1

        if best_sp is None or best_ov <= 0:
            # fallback to midpoint/nearest
            mid = (w["start"] + w["end"]) / 2
            best_sp = find_speaker(mid, segs)

        w["speaker"] = best_sp


@dataclass
class _Run:
    speaker: str
    i0: int
    i1: int
    start: float
    end: float

    @property
    def n_words(self) -> int:
        return self.i1 - self.i0 + 1

    @property
    def duration(self) -> float:
        return self.end - self.start


def _word_runs(words: list[dict]) -> list[_Run]:
    runs: list[_Run] = []
    if not words:
        return runs
    cur_sp = words[0]["speaker"]
    i0 = 0
    start = words[0]["start"]
    end = words[0]["end"]

    for i in range(1, len(words)):
        w = words[i]
        if w["speaker"] != cur_sp:
            runs.append(_Run(cur_sp, i0, i-1, start, end))
            cur_sp = w["speaker"]
            i0 = i
            start = w["start"]
            end = w["end"]
        else:
            end = w["end"]

    runs.append(_Run(cur_sp, i0, len(words)-1, start, end))
    return runs


def smooth_short_aba_flips(words: list[dict], *, max_run_sec: float = 0.35, max_run_words: int = 3) -> int:
    """
    Optional smoothing:
      If a short run B is sandwiched between the same speaker A (A–B–A),
      relabel B to A.

    This specifically targets the 'sentence tail moved to next speaker' artifact,
    but it can also remove legitimate short backchannels. Keep it optional.
    """
    runs = _word_runs(words)
    if len(runs) < 3:
        return 0

    changed = 0
    for k in range(1, len(runs)-1):
        prev_run = runs[k-1]
        mid_run = runs[k]
        next_run = runs[k+1]

        if prev_run.speaker == next_run.speaker and mid_run.speaker != prev_run.speaker:
            if mid_run.duration <= max_run_sec and mid_run.n_words <= max_run_words:
                for i in range(mid_run.i0, mid_run.i1 + 1):
                    words[i]["speaker"] = prev_run.speaker
                changed += mid_run.n_words

    return changed


# ── utterance building ───────────────────────────────────────────────

def utterances_by_speaker_change(words: list[dict]) -> list[dict]:
    """Group consecutive words with identical speaker."""
    utterances: list[dict] = []
    cur = None
    for w in words:
        if cur is None or w["speaker"] != cur["speaker"]:
            if cur is not None:
                cur["text"] = cur["text"].strip()
                utterances.append(cur)
            cur = {
                "speaker": w["speaker"],
                "start": w["start"],
                "end": w["end"],
                "text": w["word"].strip(),
                "words": [w],
            }
        else:
            cur["end"] = w["end"]
            cur["text"] += w["word"]  # faster-whisper words often include leading space
            cur["words"].append(w)

    if cur:
        cur["text"] = cur["text"].strip()
        utterances.append(cur)

    return utterances


def utterances_by_pause_and_sentence(words: list[dict], *, pause_sec: float = 0.6,
                                    speaker_strategy: str = "majority",
                                    force_words_to_utterance_speaker: bool = True) -> list[dict]:
    """
    Build utterances by (a) long pauses and (b) sentence-ending punctuation.
    Assign utterance speaker by:
      - majority: duration-weighted majority over its words (recommended)
      - first: speaker of first word
      - last: speaker of last word
    """
    if not words:
        return []

    utterances: list[dict] = []
    cur_words: list[dict] = [words[0]]

    def choose_speaker(ws: list[dict]) -> str:
        if speaker_strategy == "first":
            return ws[0]["speaker"]
        if speaker_strategy == "last":
            return ws[-1]["speaker"]

        # duration-weighted majority (default)
        scores: dict[str, float] = {}
        for w in ws:
            sp = w["speaker"]
            scores[sp] = scores.get(sp, 0.0) + max(0.01, w["end"] - w["start"])
        return max(scores.items(), key=lambda kv: kv[1])[0]

    def flush():
        nonlocal cur_words
        if not cur_words:
            return

        sp = choose_speaker(cur_words)
        if force_words_to_utterance_speaker:
            for w in cur_words:
                w["speaker"] = sp

        text = "".join(w["word"] for w in cur_words).strip()
        utterances.append(
            {
                "speaker": sp,
                "start": cur_words[0]["start"],
                "end": cur_words[-1]["end"],
                "text": text,
                "words": list(cur_words),
            }
        )
        cur_words = []

    for w in words[1:]:
        # cur_words is guaranteed non-empty here
        prev = cur_words[-1]
        gap = w["start"] - prev["end"]

        # Decide boundary BEFORE adding the next word.
        # If we already hit a boundary after the previous word,
        # flush the current utterance and start a new one with `w`.
        if gap >= pause_sec or _is_sentence_end(prev["word"]):
            flush()
            cur_words = [w]
        else:
            cur_words.append(w)

    flush()
    return utterances


# ── pyannote tuning (optional) ───────────────────────────────────────

def apply_pyannote_overrides(pipeline, *,
                            segmentation_threshold: float | None,
                            min_duration_off: float | None,
                            clustering_threshold: float | None,
                            show_params: bool) -> None:
    """
    Apply common overrides if the loaded pipeline exposes those knobs.
    Safe: if a key is missing, it is ignored.
    """
    if not any(v is not None for v in [segmentation_threshold, min_duration_off, clustering_threshold, show_params]):
        return

    params = pipeline.parameters(instantiated=True)

    if show_params:
        print("      pyannote parameters(instantiated=True):")
        print(json.dumps(params, indent=2, ensure_ascii=False))

    changed = False

    if segmentation_threshold is not None:
        try:
            if "segmentation" in params and "threshold" in params["segmentation"]:
                params["segmentation"]["threshold"] = float(segmentation_threshold)
                changed = True
        except Exception:
            pass

    if min_duration_off is not None:
        try:
            if "segmentation" in params and "min_duration_off" in params["segmentation"]:
                params["segmentation"]["min_duration_off"] = float(min_duration_off)
                changed = True
        except Exception:
            pass

    if clustering_threshold is not None:
        # key name differs across versions; try common locations
        updated = False
        for path in [("clustering", "threshold"), ("clustering", "clustering_threshold")]:
            try:
                a, b = path
                if a in params and b in params[a]:
                    params[a][b] = float(clustering_threshold)
                    updated = True
            except Exception:
                pass
        changed = changed or updated

    if changed:
        pipeline.instantiate(params)
        print("      Applied pyannote overrides.")


# ── main pipeline ────────────────────────────────────────────────────

def run(args):
    # lazy imports so --help is instant
    import torch
    from faster_whisper import WhisperModel
    from pyannote.audio import Pipeline
    from pyannote.audio.pipelines.utils.hook import ProgressHook

    audio_path = args.audio
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    basename = Path(audio_path).stem

    hf_token = resolve_hf_token(args.hf_token, args.project_root)
    if not hf_token:
        sys.exit(
            "ERROR  No HuggingFace token found.\n"
            "       Set HF_TOKEN env var, pass --hf_token, or create .hf_token in project root."
        )

    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    device_tag = "cuda" if use_cuda else "cpu"

    # ── 1. Speaker diarization ───────────────────────────────────────
    t0 = time.perf_counter()
    print(f"[1/3] Speaker diarization  (device={device_tag}) …")

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=hf_token,
    )
    apply_pyannote_overrides(
        pipeline,
        segmentation_threshold=args.segmentation_threshold,
        min_duration_off=args.min_duration_off,
        clustering_threshold=args.clustering_threshold,
        show_params=args.show_pyannote_params,
    )

    if use_cuda:
        pipeline.to(torch.device("cuda"))

    diar_kw = {}
    if args.num_speakers is not None:
        diar_kw["num_speakers"] = args.num_speakers
    if args.min_speakers is not None:
        diar_kw["min_speakers"] = args.min_speakers
    if args.max_speakers is not None:
        diar_kw["max_speakers"] = args.max_speakers

    # Preload audio with torchaudio to bypass torchcodec issues.
    waveform, sample_rate, tmp_wav = load_audio(audio_path)
    audio_input = {"waveform": waveform, "sample_rate": sample_rate}

    with ProgressHook() as hook:
        diarization = pipeline(audio_input, hook=hook, **diar_kw)

    # Build speaker timeline (prefer exclusive mode for cleaner alignment)
    speaker_timeline: list[dict] = []
    try:
        for turn, speaker in diarization.exclusive_speaker_diarization:
            speaker_timeline.append({"start": turn.start, "end": turn.end, "speaker": speaker})
    except AttributeError:
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_timeline.append({"start": turn.start, "end": turn.end, "speaker": speaker})

    speakers_found = sorted(set(s["speaker"] for s in speaker_timeline))
    print(f"      {len(speakers_found)} speakers: {', '.join(speakers_found)}")

    # clean up temp WAV and free VRAM
    del pipeline, waveform, audio_input
    if tmp_wav and tmp_wav.exists():
        tmp_wav.unlink()
    if use_cuda:
        torch.cuda.empty_cache()

    # ── 2. Transcription with word timestamps ────────────────────────
    t1 = time.perf_counter()
    print(f"[2/3] Transcribing  (model={args.model}, device={device_tag}, compute_type={args.compute_type}) …")

    model_kw = {"device": device_tag, "compute_type": args.compute_type}
    if args.model_dir:
        model_kw["download_root"] = args.model_dir

    model = WhisperModel(args.model, **model_kw)

    transcribe_kw: dict = {
        "language": args.language,
        "word_timestamps": True,
        "vad_filter": True,
    }
    if args.initial_prompt:
        transcribe_kw["initial_prompt"] = args.initial_prompt

    segments_iter, info = model.transcribe(audio_path, **transcribe_kw)

    words: list[dict] = []
    for seg in segments_iter:
        if seg.words:
            for w in seg.words:
                words.append(
                    {
                        "start": w.start,
                        "end": w.end,
                        "word": w.word,
                        "probability": round(w.probability, 4),
                    }
                )

    # IMPORTANT: CTranslate2 may crash on model destruction.
    # Print progress and continue to writing outputs BEFORE any cleanup.
    print(f"      {len(words)} words transcribed")
    sys.stdout.flush()

    if not (use_cuda and sys.platform.startswith("win")):
        del model
        if use_cuda:
            torch.cuda.empty_cache()

    # ── 3. Align words → speakers ────────────────────────────────────
    t2 = time.perf_counter()
    print("[3/3] Aligning words to speakers …")

    if args.assign_mode == "midpoint":
        assign_speakers_midpoint(words, speaker_timeline)
    else:
        assign_speakers_overlap(words, speaker_timeline, collar=args.collar)

    n_smoothed = 0
    if args.smooth_max_run_sec is not None and args.smooth_max_run_words is not None:
        if args.smooth_max_run_sec > 0 and args.smooth_max_run_words > 0:
            n_smoothed = smooth_short_aba_flips(
                words,
                max_run_sec=args.smooth_max_run_sec,
                max_run_words=args.smooth_max_run_words,
            )
    if n_smoothed:
        print(f"      Smoothed {n_smoothed} word(s) across short A–B–A flips")

    # Build utterances
    if args.utterance_mode == "speaker_change":
        utterances = utterances_by_speaker_change(words)
    else:
        utterances = utterances_by_pause_and_sentence(
            words,
            pause_sec=args.pause_sec,
            speaker_strategy=args.utterance_speaker_strategy,
            force_words_to_utterance_speaker=not args.keep_word_speakers_in_txt,
        )

    # ── 4. Write outputs ─────────────────────────────────────────────
    fmts = {"txt", "json", "srt"} if args.output_format == "all" else {args.output_format}

    # combined
    print("  Combined:")
    if "txt" in fmts:
        p = out_dir / f"{basename}.txt"
        write_txt(p, utterances)
        print(f"    → {p}")

    if "json" in fmts:
        p = out_dir / f"{basename}.json"
        write_json(
            p,
            utterances,
            audio_path=audio_path,
            language=args.language,
            model=args.model,
            speakers_found=speakers_found,
            speaker_timeline=speaker_timeline,
            include_word_speakers=args.include_word_speakers_in_json,
        )
        print(f"    → {p}")

    if "srt" in fmts:
        p = out_dir / f"{basename}.srt"
        write_srt(p, utterances)
        print(f"    → {p}")

    # per-speaker
    if len(speakers_found) > 1:
        for sp in speakers_found:
            sp_utterances = [u for u in utterances if u["speaker"] == sp]
            sp_timeline = [s for s in speaker_timeline if s["speaker"] == sp]
            tag = sp.replace(" ", "_")

            print(f"  {sp}:")
            if "txt" in fmts:
                p = out_dir / f"{basename}_{tag}.txt"
                write_txt(p, sp_utterances)
                print(f"    → {p}")

            if "json" in fmts:
                p = out_dir / f"{basename}_{tag}.json"
                write_json(
                    p,
                    sp_utterances,
                    audio_path=audio_path,
                    language=args.language,
                    model=args.model,
                    speakers_found=[sp],
                    speaker_timeline=sp_timeline,
                    include_word_speakers=args.include_word_speakers_in_json,
                )
                print(f"    → {p}")

            if "srt" in fmts:
                p = out_dir / f"{basename}_{tag}.srt"
                write_srt(p, sp_utterances)
                print(f"    → {p}")

    # summary
    t3 = time.perf_counter()
    print(
        f"\nDone  ({len(utterances)} utterances, {len(words)} words)\n"
        f"  diarize  {t1-t0:.1f}s · transcribe {t2-t1:.1f}s · align {t3-t2:.1f}s · total {t3-t0:.1f}s"
    )
    for sp in speakers_found:
        n_utt = sum(1 for u in utterances if u["speaker"] == sp)
        n_wrd = sum(len(u["words"]) for u in utterances if u["speaker"] == sp)
        print(f"  {sp}: {n_utt} turns, {n_wrd} words")

    # FINAL SAFETY EXIT
    if use_cuda and sys.platform.startswith("win"):
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Diarized transcription (faster-whisper + pyannote community-1)")
    ap.add_argument("audio", help="Path to audio/video file")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--language", default="de")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--compute_type", default="float16",
                    help="float16 | int8_float16 | int8 | float32")
    ap.add_argument("--model_dir", default=None,
                    help="Download root for whisper models")
    ap.add_argument("--output_format", default="all",
                    choices=["txt", "json", "srt", "all"])

    # diarization
    ap.add_argument("--num_speakers", type=int, default=None)
    ap.add_argument("--min_speakers", type=int, default=None)
    ap.add_argument("--max_speakers", type=int, default=None)

    # context / prompt
    ap.add_argument("--initial_prompt", default=None)

    # auth
    ap.add_argument("--hf_token", default=None,
                    help="HuggingFace access token (or set HF_TOKEN env var)")
    ap.add_argument("--project_root", default=None,
                    help="Project root to locate .hf_token file")

    # alignment / formatting
    ap.add_argument("--assign_mode", default="overlap", choices=["overlap", "midpoint"],
                    help="How to assign speaker labels to words. overlap is usually best.")
    ap.add_argument("--collar", type=float, default=0.05,
                    help="Seconds to expand each word interval for overlap assignment (default 0.05).")
    ap.add_argument("--utterance_mode", default="pause_sentence", choices=["pause_sentence", "speaker_change"],
                    help="How to build human-readable utterances.")
    ap.add_argument("--pause_sec", type=float, default=0.6,
                    help="Pause threshold in seconds when utterance_mode=pause_sentence.")
    ap.add_argument("--utterance_speaker_strategy", default="majority", choices=["majority", "first", "last"],
                    help="How to assign an utterance speaker when utterance_mode=pause_sentence.")
    ap.add_argument("--keep_word_speakers_in_txt", action="store_true",
                    help="If set, keep per-word speakers as-is when writing TXT (utterance label may be majority).")
    ap.add_argument("--include_word_speakers_in_json", action="store_true",
                    help="If set, include each word's assigned speaker in JSON words[].speaker")

    # smoothing (optional; default disabled)
    ap.add_argument("--smooth_max_run_sec", type=float, default=0.0,
                    help="Enable A–B–A smoothing: max duration of the middle run in seconds (0 disables).")
    ap.add_argument("--smooth_max_run_words", type=int, default=0,
                    help="Enable A–B–A smoothing: max number of words in the middle run (0 disables).")

    # pyannote tuning (optional; safe no-ops if keys are missing)
    ap.add_argument("--segmentation_threshold", type=float, default=None,
                    help="Override pyannote segmentation.threshold if available.")
    ap.add_argument("--min_duration_off", type=float, default=None,
                    help="Override pyannote segmentation.min_duration_off if available.")
    ap.add_argument("--clustering_threshold", type=float, default=None,
                    help="Override pyannote clustering.threshold if available.")
    ap.add_argument("--show_pyannote_params", action="store_true",
                    help="Print pyannote pipeline parameters(instantiated=True) and continue.")

    args = ap.parse_args()

    # normalize smoothing params: treat 0/0 as disabled
    if args.smooth_max_run_sec <= 0 or args.smooth_max_run_words <= 0:
        args.smooth_max_run_sec = None
        args.smooth_max_run_words = None

    run(args)


if __name__ == "__main__":
    main()
