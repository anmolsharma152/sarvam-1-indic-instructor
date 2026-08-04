#!/usr/bin/env bash
# track_progress.sh — Reports dataset generation progress and speed.
# Reads previous count from a state file to compute rate over the last interval.

JSONL="data/raw_instructions.jsonl"
STATE_FILE="/tmp/sarvam_gen_tracker.state"
TARGET=15000

current=$(wc -l < "$JSONL" 2>/dev/null || echo 0)
now=$(date +%s)

if [[ -f "$STATE_FILE" ]]; then
    prev_count=$(cut -d',' -f1 "$STATE_FILE")
    prev_time=$(cut -d',' -f2 "$STATE_FILE")
    elapsed=$(( now - prev_time ))
    delta=$(( current - prev_count ))
    if [[ $elapsed -gt 0 ]]; then
        rate=$(python3 -c "print(round($delta / ($elapsed / 60), 1))")
    else
        rate="N/A"
    fi
else
    rate="N/A (first check)"
    prev_count="N/A"
fi

remaining=$(( TARGET - current ))
if [[ "$rate" =~ ^[0-9] && "$rate" != "0" ]]; then
    eta_str=$(python3 -c "
r=$rate; rem=$remaining
if r > 0:
    mins = rem / r
    print(f'{int(mins)} min (~{mins/60:.1f} hrs)')
else:
    print('calculating...')
")
else
    eta_str="calculating..."
fi

echo "current=$current prev=${prev_count} rate=${rate}/min remaining=$remaining eta=$eta_str target=$TARGET"

# Save state for next run
echo "${current},${now}" > "$STATE_FILE"
