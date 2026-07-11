#!/usr/bin/env bash
set -euo pipefail

RUNS="${RUNS:-25}"
FIXTURES="${FIXTURES:-ev-01,ev-06,ev-08,ev-11,ev-17,ev-22}"
HOST="${HOST:-http://gpu-host:11434}"
MODELS=("qwen2.5:7b" "qwen3:14b" "deepseek-r1:14b")
REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
OUTDIR="${OUTDIR:-$REPO_ROOT/data/variance_sweep_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTDIR"

CSV="$OUTDIR/results.csv"
echo "run,model,quality,avg_latency_ms,tokens,fixtures" > "$CSV"

echo "=== Variance Sweep ==="
echo "Models: ${MODELS[*]}"
echo "Fixtures: $FIXTURES"
echo "Runs: $RUNS"
echo "Output: $OUTDIR"
echo ""

for run in $(seq 1 $RUNS); do
    echo "──────────────────────────────────────────"
    echo "RUN $run / $RUNS  ($(date '+%H:%M:%S'))"
    echo "──────────────────────────────────────────"

    for model in "${MODELS[@]}"; do
        echo "  [$run/$RUNS] $model ..."
        output=$(uv run atomics sweep \
            --models "$model" \
            --host "$HOST" \
            --fixtures "$FIXTURES" \
            2>&1)

        quality=$(echo "$output" | grep -E '^\│' | grep "$model" | \
            sed 's/[^0-9%]*//g' | grep -oE '[0-9]+%' | head -1 || echo "ERR")
        latency=$(echo "$output" | grep -E '^\│' | grep "$model" | \
            grep -oE '[0-9]+ms' | head -1 || echo "0ms")
        tokens=$(echo "$output" | grep -E '^\│' | grep "$model" | \
            grep -oE '[0-9,]+' | sed 's/,//g' | awk 'NR==4{print}' || echo "0")

        lat_num=${latency%ms}
        qual_num=${quality%%%}

        echo "$run,$model,$qual_num,$lat_num,$tokens,6" >> "$CSV"
        echo "  [$run/$RUNS] $model -> quality=${quality} latency=${latency}"
    done
done

echo ""
echo "=== SWEEP COMPLETE ==="
echo "Raw data: $CSV"
echo ""

# Summary statistics
echo "=== VARIANCE SUMMARY ===" | tee "$OUTDIR/summary.txt"
echo "" | tee -a "$OUTDIR/summary.txt"

for model in "${MODELS[@]}"; do
    echo "--- $model ---" | tee -a "$OUTDIR/summary.txt"
    awk -F',' -v m="$model" '
    NR>1 && $2==m && $3!="ERR" {
        n++; sum+=$3; sumsq+=$3*$3
        sumL+=$4; sumsqL+=$4*$4
        if(n==1 || $3<min) min=$3
        if(n==1 || $3>max) max=$3
        if(n==1 || $4<minL) minL=$4
        if(n==1 || $4>maxL) maxL=$4
    }
    END {
        if(n>0) {
            mean=sum/n
            var=(sumsq/n)-(mean*mean)
            sd=(var>0)?sqrt(var):0
            meanL=sumL/n
            varL=(sumsqL/n)-(meanL*meanL)
            sdL=(varL>0)?sqrt(varL):0
            printf "  Quality:  mean=%.1f%%  stddev=%.1f%%  min=%d%%  max=%d%%  (n=%d)\n", mean, sd, min, max, n
            printf "  Latency:  mean=%.0fms  stddev=%.0fms  min=%dms  max=%dms\n", meanL, sdL, minL, maxL
        } else {
            print "  No valid data"
        }
    }' "$CSV" | tee -a "$OUTDIR/summary.txt"
    echo "" | tee -a "$OUTDIR/summary.txt"
done

echo "Done at $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$OUTDIR/summary.txt"
