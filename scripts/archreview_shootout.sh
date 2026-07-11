#!/usr/bin/env bash
# Archreview shootout — run one model-under-test repo review across multiple
# inference hosts and/or frontier APIs, reported side by side. Reproduces the
# Juice Shop floor-tier comparison used for the leaderboard.
#
# Robust by design: each model's full stdout+stderr is tee'd to a per-model log
# under $OUT; if a run produces no result row (e.g. a missing provider extra or
# an API error) we print an explicit ERROR with the tail of that log rather than
# emitting a silent blank row.
#
# Configure via env (no defaults point at any specific machine):
#   HOST_A       Ollama endpoint for the first host   (e.g. http://gpu-a:11434)
#   HOST_B       Ollama endpoint for the second host  (e.g. http://gpu-b:11434)
#   JUDGE_HOST   Ollama endpoint used to judge frontier runs (default: HOST_A)
#   JUDGE_MODEL  Judge model, fixed across all runs for parity (default qwen2.5:7b)
#   ROUNDS       Rounds per model (default 3)
#   JUICE_SHOP_PATH  Path to a local juice-shop checkout (required)
#   HOST_A_MODELS / HOST_B_MODELS  Space-separated model lists per host
#   FRONTIER     Space-separated "provider:model" pairs (e.g. "claude:claude-sonnet-4-6 openai:gpt-4o")
#   OUT          Log/output dir (default ./shootout-out)
#
# Frontier legs need the matching extras: uv sync --extra openai --extra anthropic
set -uo pipefail

cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

: "${JUICE_SHOP_PATH:?set JUICE_SHOP_PATH to a local juice-shop checkout}"
export JUICE_SHOP_PATH

HOST_A="${HOST_A:-}"
HOST_B="${HOST_B:-}"
JUDGE_HOST="${JUDGE_HOST:-$HOST_A}"
JUDGE_MODEL="${JUDGE_MODEL:-qwen2.5:7b}"
ROUNDS="${ROUNDS:-3}"
OUT="${OUT:-./shootout-out}"
HOST_A_MODELS="${HOST_A_MODELS:-qwen2.5:7b qwen3:14b phi4:latest}"
HOST_B_MODELS="${HOST_B_MODELS:-qwen2.5:7b qwen3:14b phi4:latest qwen3.6:27b}"
FRONTIER="${FRONTIER:-claude:claude-sonnet-4-6 openai:gpt-4o}"
mkdir -p "$OUT"

hr() { printf '=%.0s' {1..60}; echo; }

# run_one <label> <model> <provider> <model-host> <judge-host>
run_one() {
  local label="$1" model="$2" provider="$3" mhost="$4" jhost="$5"
  local log="$OUT/${label}_${model//[:\/]/_}.log"
  echo "--- $label: $model ($provider) ---"
  local args=(archreview --repo juice-shop --models "$model" --provider "$provider"
              --judge-provider ollama --judge-model "$JUDGE_MODEL" --judge-host "$jhost"
              --tier floor --rounds "$ROUNDS" --no-save)
  [[ "$provider" == "ollama" ]] && args+=(--ollama-host "$mhost")
  uv run atomics "${args[@]}" >"$log" 2>&1
  local rc=$?
  local row
  row=$(grep -E "^│ ?${model//\//\\/}" "$log" | tail -1)
  if [[ -n "$row" ]]; then
    echo "$row"
  else
    echo "  !! ERROR (rc=$rc) — no result row. Last lines:"
    tail -4 "$log" | sed 's/^/     /'
  fi
  echo
}

hr; echo "  ARCHREVIEW SHOOTOUT: Juice Shop (floor, ${ROUNDS} rounds)"
echo "  Started: $(date)"; hr; echo

if [[ -n "$HOST_A" ]]; then
  echo "=== LOCAL MODELS ON HOST A ($HOST_A) ==="; echo
  for m in $HOST_A_MODELS; do run_one "host-a" "$m" ollama "$HOST_A" "$HOST_A"; done
fi

if [[ -n "$HOST_B" ]]; then
  echo "=== LOCAL MODELS ON HOST B ($HOST_B) ==="; echo
  for m in $HOST_B_MODELS; do run_one "host-b" "$m" ollama "$HOST_B" "$HOST_B"; done
fi

if [[ -n "$FRONTIER" ]]; then
  echo "=== FRONTIER MODELS (judged on $JUDGE_HOST) ==="; echo
  for pair in $FRONTIER; do
    run_one "frontier" "${pair#*:}" "${pair%%:*}" "" "$JUDGE_HOST"
  done
fi

hr; echo "  SHOOTOUT COMPLETE — $(date)"; hr
