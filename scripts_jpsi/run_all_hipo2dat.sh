#!/bin/bash
# Launch HIPO -> DAT conversion for all 9 J/psi datasets in parallel.
# Run inside tmux:   cd ~/fastmc/scripts_jpsi && ./run_all_hipo2dat.sh
#
# Per-dataset logs:  /volatile/clas12/vpk/fastmc/jpsi/<dataset>/v1/report/hipo2dat.log
# Output dat files:  /volatile/clas12/vpk/fastmc/jpsi/<dataset>/v1/dat/jpsi_tm_*.dat
set -u

BASE=/volatile/clas12/vpk/fastmc/jpsi
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BATCH_SIZE=5

# dataset : beam energy (GeV)
DATASETS=(
    "F18in_45nA  10.6"
    "F18in_50nA  10.6"
    "F18in_55nA  10.6"
    "F18out_40nA 10.6"
    "F18out_50nA 10.6"
    "S18in_35nA  10.6"
    "S18in_50nA  10.6"
    "S18out_45nA 10.6"
    "S19in_50nA  10.2"
)

PIDS=(); NAMES=()
for entry in "${DATASETS[@]}"; do
    DS=$(echo "$entry" | awk "{print \$1}")
    EB=$(echo "$entry" | awk "{print \$2}")
    V1=$BASE/$DS/v1
    HIPO=$V1/hipo/hipo_dir
    LOG=$V1/report/hipo2dat.log

    [ -d "$HIPO" ] || { echo "SKIP $DS: $HIPO not found"; continue; }
    mkdir -p "$V1/dat" "$V1/report"

    "$SCRIPT_DIR/run_batches_tm.sh" "$HIPO" "$V1/dat" "$BATCH_SIZE" "$EB" \
        > "$LOG" 2>&1 &
    PIDS+=($!); NAMES+=("$DS")
    echo "launched $DS  (Ebeam=$EB, PID $!)  log: $LOG"
done

echo ""
echo "$(date)  — waiting for ${#PIDS[@]} jobs ..."
echo "Monitor from another window with:"
echo "  tail -f $BASE/*/v1/report/hipo2dat.log"
echo ""

FAIL=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "$(date)  DONE  ${NAMES[$i]}"
    else
        echo "$(date)  FAILED  ${NAMES[$i]}  (check log)"
        FAIL=1
    fi
done

# ── final summary ──
echo ""
echo "================ SUMMARY ================"
for entry in "${DATASETS[@]}"; do
    DS=$(echo "$entry" | awk "{print \$1}")
    D=$BASE/$DS/v1/dat
    if [ -f "$D/jpsi_tm_train.dat" ]; then
        NT=$(grep -c '^[0-9]' "$D/jpsi_tm_train.dat" 2>/dev/null || echo 0)
        NG=$(grep -c '^[0-9]' "$D/jpsi_tm_hadgated_train.dat" 2>/dev/null || echo 0)
        printf "%-13s train: %9d ev   hadgated_train: %8d ev\n" "$DS" "$NT" "$NG"
    else
        printf "%-13s NO OUTPUT\n" "$DS"
    fi
done
echo "========================================="
[ "$FAIL" -eq 0 ] && echo "All datasets converted OK." || echo "Some datasets FAILED — check logs."
