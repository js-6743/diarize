#!/usr/bin/env bash
set -euo pipefail

show_usage() {
  cat <<'USAGE'
Usage:
  ./run_diarize.sh <Batch> [wrapper flags] [transcribe.sh flags]

Wrapper flags:
  --venv-root <path>   (default: ~/Documents/Python/envs/diarize)
  --ffmpeg-bin <path>  (optional; folder containing ffmpeg)
  -h, --help

Examples:
  ./run_diarize.sh Biweekly --num-speakers 3 --extensions wav
  ./run_diarize.sh Biweekly --ffmpeg-bin /opt/homebrew/bin --num-speakers 3
USAGE
}

if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
  show_usage
  exit 1
fi

BATCH="$1"
shift

VENV_ROOT="$HOME/Documents/Python/envs/diarize"
FFMPEG_BIN=""
PASS_THRU=("$BATCH")

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      show_usage
      exit 0
      ;;
    --venv-root)
      VENV_ROOT="$2"; shift 2;;
    --ffmpeg-bin)
      FFMPEG_BIN="$2"; shift 2;;
    *)
      PASS_THRU+=("$1"); shift;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRANSCRIBE_SH="$SCRIPT_DIR/transcribe.sh"

if [[ ! -f "$TRANSCRIBE_SH" ]]; then
  echo "transcribe.sh not found next to run_diarize.sh: $TRANSCRIBE_SH" >&2
  exit 1
fi
if [[ ! -d "$VENV_ROOT" ]]; then
  echo "Venv root does not exist: $VENV_ROOT" >&2
  exit 1
fi

ACTIVATE="$VENV_ROOT/bin/activate"
PYTHON_EXE="$VENV_ROOT/bin/python"

if [[ ! -f "$ACTIVATE" ]]; then
  echo "venv activate not found: $ACTIVATE" >&2
  exit 1
fi
if [[ ! -x "$PYTHON_EXE" ]]; then
  echo "venv python not found: $PYTHON_EXE" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$ACTIVATE"

echo "Activating venv: $VENV_ROOT"
echo "Using Python: $($PYTHON_EXE -c 'import sys; print(sys.executable)')"

if [[ -z "$FFMPEG_BIN" ]]; then
  if command -v ffmpeg >/dev/null 2>&1; then
    FFMPEG_BIN="$(dirname "$(command -v ffmpeg)")"
  fi
fi

if [[ -n "$FFMPEG_BIN" ]]; then
  if [[ -d "$FFMPEG_BIN" ]]; then
    export FFMPEG_BIN="$FFMPEG_BIN"
    export PATH="$FFMPEG_BIN:$PATH"
    echo "FFMPEG_BIN: $FFMPEG_BIN"
  else
    echo "WARNING: --ffmpeg-bin path does not exist (ignored): $FFMPEG_BIN" >&2
  fi
fi

"$TRANSCRIBE_SH" --python-exe "$PYTHON_EXE" "${PASS_THRU[@]}"
