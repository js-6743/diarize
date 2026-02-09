#!/usr/bin/env bash
set -euo pipefail

show_usage() {
  cat <<'USAGE'
Usage:
  ./transcribe.sh <Batch> [options]

Options:
  --project-root <path>       (default: ~/Documents/Python/diarize)
  --python-exe <path>         (default: ~/Documents/Python/envs/diarize/bin/python)
  --model <name>              (default: large-v3)
  --language <code>           (default: de)
  --device <cuda|cpu>         (default: cpu)
  --compute-type <type>       (default: int8)
  --output-format <txt|json|srt|all> (default: all)
  --num-speakers <n>
  --min-speakers <n>
  --max-speakers <n>
  --context-file <path>       (default: expert_interview_context.txt)
  --no-context
  --hf-token <token>
  --recurse
  --extensions <csv>          (default: wav,mp3,m4a,mp4,flac,aac,ogg,webm,mkv)
  --skip-existing
  -h, --help

Examples:
  ./transcribe.sh Expert_03 --num-speakers 2
  ./transcribe.sh Expert_03 --recurse --extensions wav
USAGE
}

if [[ $# -lt 1 ]]; then
  show_usage
  exit 1
fi

BATCH=""
PROJECT_ROOT="$HOME/Documents/Python/diarize"
PYTHON_EXE="$HOME/Documents/Python/envs/diarize/bin/python"
MODEL="large-v3"
LANGUAGE="de"
DEVICE="cpu"
COMPUTE_TYPE="int8"
OUTPUT_FORMAT="all"
NUM_SPEAKERS=""
MIN_SPEAKERS=""
MAX_SPEAKERS=""
CONTEXT_FILE="expert_interview_context.txt"
NO_CONTEXT=0
HF_TOKEN=""
RECURSE=0
SKIP_EXISTING=0
EXTENSIONS=(wav mp3 m4a mp4 flac aac ogg webm mkv)

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      show_usage
      exit 0
      ;;
    --project-root)
      PROJECT_ROOT="$2"; shift 2;;
    --python|--python-exe)
      PYTHON_EXE="$2"; shift 2;;
    --model)
      MODEL="$2"; shift 2;;
    --language)
      LANGUAGE="$2"; shift 2;;
    --device)
      DEVICE="$2"; shift 2;;
    --compute-type)
      COMPUTE_TYPE="$2"; shift 2;;
    --output-format)
      OUTPUT_FORMAT="$2"; shift 2;;
    --num-speakers)
      NUM_SPEAKERS="$2"; shift 2;;
    --min-speakers)
      MIN_SPEAKERS="$2"; shift 2;;
    --max-speakers)
      MAX_SPEAKERS="$2"; shift 2;;
    --context-file)
      CONTEXT_FILE="$2"; shift 2;;
    --no-context)
      NO_CONTEXT=1; shift;;
    --hf-token)
      HF_TOKEN="$2"; shift 2;;
    --recurse)
      RECURSE=1; shift;;
    --extensions)
      IFS=',' read -r -a EXTENSIONS <<< "$2"; shift 2;;
    --skip-existing)
      SKIP_EXISTING=1; shift;;
    *)
      if [[ -z "$BATCH" ]]; then
        BATCH="$1"; shift
      else
        echo "Unknown argument: $1"
        show_usage
        exit 1
      fi
      ;;
  esac
done

if [[ -z "$BATCH" ]]; then
  show_usage
  exit 1
fi

INPUT_DIR="$PROJECT_ROOT/input/$BATCH"
OUT_ROOT="$PROJECT_ROOT/out/$BATCH"
MODEL_DIR="$PROJECT_ROOT/models"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIARIZE_PY="$SCRIPT_DIR/diarize.py"

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "Input batch folder not found: $INPUT_DIR" >&2
  exit 1
fi
if [[ ! -f "$DIARIZE_PY" ]]; then
  echo "diarize.py not found next to this script: $DIARIZE_PY" >&2
  exit 1
fi
if [[ ! -x "$PYTHON_EXE" ]]; then
  echo "Python executable not found: $PYTHON_EXE" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT" "$MODEL_DIR"

LOG_PATH="$OUT_ROOT/_log.txt"
{
  echo "=== Diarize batch: $BATCH ==="
  echo "Start: $(date)"
  echo "ProjectRoot: $PROJECT_ROOT"
  echo "PythonExe: $PYTHON_EXE"
  echo "Model: $MODEL | Language: $LANGUAGE | Device: $DEVICE | ComputeType: $COMPUTE_TYPE | OutputFormat: $OUTPUT_FORMAT"
} > "$LOG_PATH"

INITIAL_PROMPT=""
if [[ $NO_CONTEXT -eq 0 ]]; then
  if [[ -n "$CONTEXT_FILE" ]]; then
    if [[ "$CONTEXT_FILE" = /* ]]; then
      CTX_PATH="$CONTEXT_FILE"
    else
      CTX_PATH="$INPUT_DIR/$CONTEXT_FILE"
    fi

    if [[ -f "$CTX_PATH" ]]; then
      INITIAL_PROMPT=$(
        "$PYTHON_EXE" - <<PY
import re
from pathlib import Path
text = Path(r"$CTX_PATH").read_text(encoding="utf-8").strip()
print(re.sub(r"\s+", " ", text).strip())
PY
      )
      if [[ -n "$INITIAL_PROMPT" ]]; then
        echo "Context: ON  ($CTX_PATH)" >> "$LOG_PATH"
      else
        echo "Context: OFF (empty file)" >> "$LOG_PATH"
      fi
    else
      echo "Context: OFF" >> "$LOG_PATH"
    fi
  else
    echo "Context: OFF" >> "$LOG_PATH"
  fi
else
  echo "Context: OFF  (--no-context)" >> "$LOG_PATH"
fi

find_cmd=(find "$INPUT_DIR")
if [[ $RECURSE -eq 0 ]]; then
  find_cmd+=(-maxdepth 1)
fi
find_cmd+=(-type f)
expr=()
for ext in "${EXTENSIONS[@]}"; do
  expr+=(-iname "*.${ext}" -o)
done
unset 'expr[${#expr[@]}-1]'
find_cmd+=( "(" "${expr[@]}" ")" )

mapfile -t files < <("${find_cmd[@]}" | sort -u)

if [[ ${#files[@]} -eq 0 ]]; then
  echo "No media files found in $INPUT_DIR"
  echo "No media files found." >> "$LOG_PATH"
  exit 0
fi

echo "Found ${#files[@]} media file(s)"
echo "Files: ${#files[@]}" >> "$LOG_PATH"

if [[ ${#files[@]} -gt 1 ]]; then
  echo "NOTE: Multiple media files found. The script will process ALL of them (one output folder per file)."
  echo "      If you only want ONE (e.g. only the .wav), either remove the others or run with --extensions wav."
fi

INPUT_FULL="$(cd "$INPUT_DIR" && pwd)"

declare -A dir_counts
for f in "${files[@]}"; do
  dir_counts["$(dirname "$f")"]=$(( ${dir_counts["$(dirname "$f")"]:-0} + 1 ))
done

for f in "${files[@]}"; do
  file_dir="$(dirname "$f")"
  group_n=${dir_counts["$file_dir"]}
  dir_full="$(cd "$file_dir" && pwd)"

  rel=""
  if [[ "$dir_full" == "$INPUT_FULL" ]]; then
    rel=""
  elif [[ "$dir_full" == "$INPUT_FULL"/* ]]; then
    rel="${dir_full#"$INPUT_FULL/"}"
  fi

  if [[ -n "$rel" ]]; then
    out_base="$OUT_ROOT/$rel"
  else
    out_base="$OUT_ROOT"
  fi
  mkdir -p "$out_base"

  base_name="$(basename "$f")"
  stem="${base_name%.*}"
  if [[ $group_n -gt 1 ]]; then
    target_out="$out_base/$stem"
  else
    target_out="$out_base"
  fi
  mkdir -p "$target_out"

  if [[ $SKIP_EXISTING -eq 1 ]]; then
    expected=()
    if [[ "$OUTPUT_FORMAT" == "all" || "$OUTPUT_FORMAT" == "txt" ]]; then
      expected+=("$target_out/$stem.txt")
    fi
    if [[ "$OUTPUT_FORMAT" == "all" || "$OUTPUT_FORMAT" == "json" ]]; then
      expected+=("$target_out/$stem.json")
    fi
    if [[ "$OUTPUT_FORMAT" == "all" || "$OUTPUT_FORMAT" == "srt" ]]; then
      expected+=("$target_out/$stem.srt")
    fi
    for e in "${expected[@]}"; do
      if [[ -f "$e" ]]; then
        echo "SKIP (exists): $base_name"
        echo "SKIP (exists): $f" >> "$LOG_PATH"
        continue 2
      fi
    done
  fi

  echo "────────────────────────────────────────"
  echo "File:    $f"
  echo "Output:  $target_out"
  echo "Context: $(if [[ -n "$INITIAL_PROMPT" ]]; then echo ON; else echo OFF; fi)"
  echo "File: $f" >> "$LOG_PATH"
  echo "Out:  $target_out" >> "$LOG_PATH"

  args=(
    "$DIARIZE_PY"
    "$f"
    "--output_dir" "$target_out"
    "--model" "$MODEL"
    "--language" "$LANGUAGE"
    "--device" "$DEVICE"
    "--compute_type" "$COMPUTE_TYPE"
    "--output_format" "$OUTPUT_FORMAT"
    "--model_dir" "$MODEL_DIR"
    "--project_root" "$PROJECT_ROOT"
  )

  if [[ -n "$NUM_SPEAKERS" ]]; then
    args+=("--num_speakers" "$NUM_SPEAKERS")
  fi
  if [[ -n "$MIN_SPEAKERS" ]]; then
    args+=("--min_speakers" "$MIN_SPEAKERS")
  fi
  if [[ -n "$MAX_SPEAKERS" ]]; then
    args+=("--max_speakers" "$MAX_SPEAKERS")
  fi
  if [[ -n "$INITIAL_PROMPT" ]]; then
    args+=("--initial_prompt" "$INITIAL_PROMPT")
  fi
  if [[ -n "$HF_TOKEN" ]]; then
    args+=("--hf_token" "$HF_TOKEN")
  fi

  t0=$(date +%s)
  "$PYTHON_EXE" "${args[@]}"
  exit_code=$?
  t1=$(date +%s)
  dt=$((t1 - t0))

  if [[ $exit_code -ne 0 ]]; then
    echo "FAILED (exit $exit_code): $base_name"
    echo "FAILED (exit $exit_code) in ${dt}s" >> "$LOG_PATH"
    continue
  fi

  echo "OK: $base_name  (${dt}s)"
  echo "OK in ${dt}s" >> "$LOG_PATH"
done

echo "End: $(date)" >> "$LOG_PATH"
echo "════════════════════════════════════════"
echo "Done. Output: $OUT_ROOT"
echo "Log:    $LOG_PATH"
