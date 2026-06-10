#!/bin/bash
# Run the J/psi v1 training chain for all datasets, MAX_PAR at a time.
# (Each chain trains 4 MLPs in parallel with 4 threads each, so
#  MAX_PAR=3 means at most 48 threads — fair on the shared ifarm node.)
#
# Usage:  ./run_all_chains.sh                 # all 9 datasets
#         ./run_all_chains.sh F18in_45nA S19in_50nA   # subset
set -u

CODE=$HOME/fastmc/scripts_jpsi
BASE=/volatile/clas12/vpk/fastmc/jpsi
MAX_PAR=3

if [ $# -gt 0 ]; then
    DATASETS=("$@")
else
    DATASETS=(F18in_45nA F18in_50nA F18in_55nA F18out_40nA F18out_50nA
              S18in_35nA S18in_50nA S18out_45nA S19in_50nA)
fi

run_one() {
    local DS=$1
    echo "$(date +%H:%M)  starting chain: $DS"
    "$CODE/run_v1_chain.sh" "$DS" > /dev/null 2>&1
    local RC=$?
    if grep -q "COMPLETE" "$BASE/$DS/v1/report/CHAIN.log" 2>/dev/null; then
        echo "$(date +%H:%M)  DONE: $DS"
    else
        echo "$(date +%H:%M)  FAILED: $DS (rc=$RC, see $BASE/$DS/v1/report/CHAIN.log)"
    fi
}

PIDS=()
for DS in "${DATASETS[@]}"; do
    [ -f "$BASE/$DS/v1/dat/jpsi_tm_train.dat" ] || { echo "SKIP $DS: no dat"; continue; }
    run_one "$DS" &
    PIDS+=($!)
    # throttle to MAX_PAR concurrent chains
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_PAR" ]; do
        sleep 30
    done
done
wait
echo ""
echo "================ ALL CHAINS FINISHED ================"
for DS in "${DATASETS[@]}"; do
    M=$(ls "$BASE/$DS/v1/models/"*.pt 2>/dev/null | wc -l)
    V=$([ -f "$BASE/$DS/v1/plots/validation_MLP.pdf" ] && echo yes || echo no)
    printf "%-13s models: %d/4   validation pdf: %s\n" "$DS" "$M" "$V"
done
