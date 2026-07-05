#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/$USER/reasoning-compression-lab}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs-hpc-2a100-main-$(date +%Y-%m-%d)}"
WATCH_JOB_IDS="${WATCH_JOB_IDS:-}"
WATCH_LABEL="${WATCH_LABEL:-b01 BF16 MATH-500 (Qwen-7B + Llama-8B)}"
INTERVAL="${PROGRESS_INTERVAL:-7200}"          # 2h default
PENDING_INTERVAL="${PENDING_PROGRESS_INTERVAL:-21600}"  # 6h when mostly pending
RUNNING_INTERVAL="${RUNNING_PROGRESS_INTERVAL:-2700}"   # 45m when jobs are running
STATE_POLL_INTERVAL="${STATE_POLL_INTERVAL:-300}"
LOG_PROGRESS_INTERVAL="${LOG_PROGRESS_INTERVAL:-25}"  # send on log marker advance (e.g. 1-25/500)
TG_SOURCE="$HOME/watch-job-52772.sh"

extract_var() {
  local name="$1"
  awk -F= -v key="$name" '$1 == key {gsub(/^"|"$/, "", $2); print $2; exit}' "$TG_SOURCE"
}

TG_TOKEN="${TG_TOKEN:-$(extract_var TG_TOKEN)}"
TG_CHAT_ID="${TG_CHAT_ID:-$(extract_var TG_CHAT_ID)}"

html_escape() {
  local s="$1"
  s="${s//&/&amp;}"
  s="${s//</&lt;}"
  s="${s//>/&gt;}"
  printf '%s' "$s"
}

send_telegram() {
  local text="$1" resp
  resp=$(curl -sS -X POST "https://api.telegram.org/bot${TG_TOKEN}/sendMessage" \
    -d chat_id="${TG_CHAT_ID}" \
    --data-urlencode "text=${text}" \
    -d parse_mode="HTML")
  if python3 -c "import json,sys; sys.exit(0 if json.loads(sys.argv[1]).get('ok') else 1)" "$resp" 2>/dev/null; then
    return 0
  fi
  echo "[telegram] HTML send failed: $resp" >&2
  resp=$(curl -sS -X POST "https://api.telegram.org/bot${TG_TOKEN}/sendMessage" \
    -d chat_id="${TG_CHAT_ID}" \
    --data-urlencode "text=${text}")
  python3 -c "import json,sys; sys.exit(0 if json.loads(sys.argv[1]).get('ok') else 1)" "$resp" 2>/dev/null \
    || { echo "[telegram] plain send failed: $resp" >&2; return 1; }
}

latest_marker() {
  local log="$1"
  [[ -f "$log" ]] || return 0
  grep -ao '\[[0-9]\+-[0-9]\+/[0-9]\+\]' "$log" | tail -n 1 | tr -d '[]' || true
}

row_count() {
  local path="$1"
  [[ -f "$path" ]] && wc -l < "$path" | tr -d ' ' || echo 0
}

expected_rows_for_base() {
  local base="$1"
  local cfg="$PROJECT_DIR/configs/cells/${base}.json"
  local from_py=""
  if [[ -f "$cfg" ]]; then
    from_py=$(python3 "$PROJECT_DIR/scripts/expected_rows.py" --cell-config "$cfg" \
      ${QREASON_INFERENCE_LIMIT:+--limit "$QREASON_INFERENCE_LIMIT"} 2>/dev/null || true)
    if [[ -n "$from_py" ]]; then
      echo "$from_py"
      return 0
    fi
  fi
  if [[ "$base" == *n50* ]]; then
    echo 50
  elif [[ "$base" == *gsm8k* ]]; then
    echo 1319
  else
    echo 500
  fi
}

short_cell_name() {
  local base="$1"
  echo "$base" | sed \
    -e 's/^diag_//' \
    -e 's/^level_[a-z]_//' \
    -e 's/_math500_seed[0-9]*_n50_64k/_64k/' \
    -e 's/_math500_seed[0-9]*_n50/_32k_n50/' \
    -e 's/_math500_seed[0-9]*//' \
    -e 's/_bf16//' -e 's/_fp8//' -e 's/_awq[0-9]*//' -e 's/_gptq[0-9]*//'
}

job_line() {
  # If WATCH_JOB_IDS provided use them; else dynamically list user's recent publication-related jobs
  local ids="$WATCH_JOB_IDS"
  if [[ -z "$ids" ]]; then
    ids=$(squeue -u "$USER" -h -o '%i' 2>/dev/null | head -n 8 | tr '\n' ' ' || true)
  fi
  local id line final
  for id in $ids; do
    [[ -z "$id" ]] && continue
    line=$(squeue -j "$id" -h -o '%i %T %M %l %R' 2>/dev/null | head -n 1 || true)
    if [[ -n "$line" ]]; then
      echo "<code>$(html_escape "$line")</code>"
      continue
    fi
    final=$(sacct -j "$id" -n -X -o State,ExitCode,Elapsed 2>/dev/null | awk 'NF {print $1 " " $2 " " $3; exit}' || true)
    if [[ -n "$final" ]]; then
      echo "<code>$(html_escape "$id $final")</code>"
    else
      echo "<code>$(html_escape "$id (not in queue/accounting)")</code>"
    fi
  done
}

current_job_state() {
  # Return RUNNING if any of user's jobs (or WATCH_JOB_IDS) is running
  local ids="$WATCH_JOB_IDS"
  if [[ -z "$ids" ]]; then
    ids=$(squeue -u "$USER" -h -o '%i' 2>/dev/null | head -n 4 | tr '\n' ' ' || true)
  fi
  local id state saw_pending=0
  for id in $ids; do
    [[ -z "$id" ]] && continue
    state=$(squeue -j "$id" -h -o '%T' 2>/dev/null | head -n 1 || true)
    case "$state" in
      RUNNING)
        echo RUNNING
        return 0
        ;;
      PENDING|CONFIGURING|COMPLETING)
        saw_pending=1
        ;;
    esac
  done
  if (( saw_pending == 1 )); then
    echo PENDING
  fi
}

next_interval() {
  local state="$1"
  case "$state" in
    PENDING)
      echo "$PENDING_INTERVAL"
      ;;
    RUNNING)
      echo "$RUNNING_INTERVAL"
      ;;
    *)
      echo "$INTERVAL"
      ;;
  esac
}

summary_line() {
  local summary="$1"
  [[ -f "$summary" ]] || return 0
  python - "$summary" <<'PY_SUMMARY'
import json, sys
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    s = json.load(f)
pass_at_1 = s.get('pass_at_1')
trunc = s.get('truncation_rate')
correct = s.get('num_correct')
n = s.get('n')
parts = []
if pass_at_1 is not None:
    parts.append(f"pass@1={pass_at_1*100:.1f}%")
if correct is not None and n is not None:
    parts.append(f"{correct}/{n}")
if trunc is not None:
    parts.append(f"trunc={trunc*100:.1f}%")
print(', '.join(parts))
PY_SUMMARY
}

repetition_status() {
  local raw_file="$1"
  [[ -s "$raw_file" ]] || return 0
  python - "$raw_file" <<'PY_REP'
import json, sys
path = sys.argv[1]
with open(path, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            row = json.loads(line)
            val = row.get('decoding_repetition_penalty')
            print(f"repetition_penalty={val}" if val is not None else "repetition_penalty=missing")
            break
PY_REP
}

build_message() {
  local raw="$OUTPUT_ROOT/raw"
  local logs="$OUTPUT_ROOT/logs"
  local results="$OUTPUT_ROOT/results"

  local watched queue cell_lines
  watched=$(job_line)
  queue=$(squeue -u "$USER" -h -o '%i %T %j %R' 2>/dev/null | grep -E 'qreason-|level_|b0[0-9]|diag|pathc|d0[0-9]' | head -n 10 | while IFS= read -r line; do echo "<code>$(html_escape "$line")</code>"; done || true)

  # Cells tied to WATCH_JOB_IDS (from Slurm job names) plus any with raw rows.
  declare -A seen_cells=()
  local -a active_bases=()
  local id jname base
  cell_base_from_job() {
    local jname="$1"
    case "$jname" in
      qreason-d02_pathc_64k_qwen)
        echo "diag_qwen7b_bf16_math500_seed42_n50_64k"
        ;;
      qreason-diag_*|qreason-level_*)
        echo "${jname#qreason-}"
        ;;
      *)
        echo ""
        ;;
    esac
  }

  for id in $WATCH_JOB_IDS; do
    [[ -z "$id" ]] && continue
    jname=$(squeue -j "$id" -h -o '%j' 2>/dev/null | head -n1 || true)
    if [[ -z "$jname" ]]; then
      jname=$(sacct -j "$id" -n -X -o JobName 2>/dev/null | awk 'NF {print $1; exit}' || true)
    fi
    base=$(cell_base_from_job "$jname")
    if [[ -n "$base" ]]; then
      [[ -n "${seen_cells[$base]:-}" ]] && continue
      active_bases+=("$base")
      seen_cells["$base"]=1
    fi
  done
  for f in "$raw"/diag_*.jsonl "$raw"/*math500*.jsonl "$raw"/*_seed*.jsonl; do
    [[ -f "$f" ]] || continue
    base=$(basename "$f" .jsonl)
    [[ -n "${seen_cells[$base]:-}" ]] && continue
    active_bases+=("$base")
    seen_cells["$base"]=1
  done
  if [[ "${#active_bases[@]}" -eq 0 ]]; then
    for log in "$logs"/diag_*.log "$logs"/level_*_math500*.log; do
      [[ -f "$log" ]] || continue
      base=$(basename "$log" .log)
      [[ -n "${seen_cells[$base]:-}" ]] && continue
      active_bases+=("$base")
      seen_cells["$base"]=1
    done
  fi

  cell_lines=""
  for base in "${active_bases[@]}"; do
    local name rows want marker rep summ result
    name=$(short_cell_name "$base")
    rows=$(row_count "$raw/${base}.jsonl")
    want=$(expected_rows_for_base "$base")
    marker=$(latest_marker "$logs/${base}.log" 2>/dev/null || true)
    rep=$(repetition_status "$raw/${base}.jsonl" 2>/dev/null || true)
    summ="$results/${base}_summary.json"
    result=$(summary_line "$summ" 2>/dev/null || true)
    cell_lines+=$'• <code>'"${name}"$'</code>: <code>'"${rows}/${want}"'</code>'${marker:+ (log:${marker})}${rep:+ ($(html_escape "$rep"))}${result:+ → $(html_escape "$result")}$'\n'
  done

  if [[ -z "$cell_lines" ]]; then
    cell_lines="No active cell logs or .jsonl in archive yet."
  fi

  cat <<MSG
📊 <b>PARAM Rudra HPC progress</b> (current batch)

<b>Watched:</b>
${watched}
<b>Task:</b> $(html_escape "$WATCH_LABEL")
<b>Archive:</b> <code>$(html_escape "$(basename "$OUTPUT_ROOT")")</code>

<b>Cell progress:</b>
${cell_lines}

<b>Recent queue (yours):</b>
${queue:-none}

<b>Time:</b> $(date +'%Y-%m-%d %H:%M')
MSG
}

send_once() {
  if [[ -z "$TG_TOKEN" || -z "$TG_CHAT_ID" ]]; then
    echo "Telegram config not found. Set TG_TOKEN/TG_CHAT_ID or keep $TG_SOURCE available." >&2
    exit 2
  fi
  send_telegram "$(build_message)"
}

case "${1:-once}" in
  once)
    send_once
    ;;
  loop)
    last_sent_at=0
    last_sent_state=""
    declare -A last_rows
    declare -A last_markers
    if [[ "${SUPPRESS_INITIAL_SEND:-0}" == "1" ]]; then
      last_sent_at="$(date +%s)"
      last_sent_state="$(current_job_state)"
    fi

    progress_changed() {
      local changed=0 threshold="${LOG_PROGRESS_INTERVAL:-25}"
      for f in "$OUTPUT_ROOT"/raw/diag_*.jsonl "$OUTPUT_ROOT"/raw/*math500*.jsonl "$OUTPUT_ROOT"/raw/*_seed*.jsonl; do
        [[ -f "$f" ]] || continue
        local cur prev
        cur=$(row_count "$f")
        prev=${last_rows["$f"]:-0}
        if (( cur > prev && (cur - prev >= threshold || cur == 1) )); then
          changed=1
        fi
        last_rows["$f"]=$cur
      done
      for log in "$OUTPUT_ROOT"/logs/diag_*.log "$OUTPUT_ROOT"/logs/level_*_math500*.log "$OUTPUT_ROOT"/logs/level_*_seed*.log; do
        [[ -f "$log" ]] || continue
        local curm prevm qdone qtotal
        curm=$(latest_marker "$log" || true)
        prevm=${last_markers["$log"]:-}
        if [[ -n "$curm" && "$curm" != "$prevm" ]]; then
          qdone="${curm%%-*}"
          qtotal="${curm##*/}"
          if [[ "$curm" == *-*/* && "$qdone" == "1" && -z "$prevm" ]]; then
            changed=1
          elif [[ "$curm" == *-*/* && "$qtotal" =~ ^[0-9]+$ && "$qdone" =~ ^[0-9]+$ ]]; then
            local prev_done=0
            if [[ "$prevm" == *-*/* ]]; then
              prev_done="${prevm%%-*}"
            fi
            if (( qdone - prev_done >= threshold )); then
              changed=1
            fi
          else
            changed=1
          fi
        fi
        last_markers["$log"]=$curm
      done
      (( changed == 1 )) && return 0 || return 1
    }

    while true; do
      state="$(current_job_state)"
      interval="$(next_interval "$state")"
      now="$(date +%s)"
      should_send=0

      if (( last_sent_at == 0 || now - last_sent_at >= interval )); then
        should_send=1
      elif [[ "$state" != "$last_sent_state" ]]; then
        should_send=1
      elif progress_changed; then
        should_send=1
      fi

      if (( should_send == 1 )); then
        send_once || true
        last_sent_at="$(date +%s)"
        last_sent_state="$state"
      fi

      remaining=$(( interval - (now - last_sent_at) ))
      (( remaining < 1 )) && remaining=1
      sleep_for="$STATE_POLL_INTERVAL"
      (( remaining < sleep_for )) && sleep_for="$remaining"
      echo "[$(date)] state=${state:-unknown} next=${remaining}s poll=${sleep_for}s"
      sleep "$sleep_for"
    done
    ;;
  preview)
    build_message
    ;;
  *)
    echo "Usage: $0 [once|loop|preview]" >&2
    exit 2
    ;;
esac
