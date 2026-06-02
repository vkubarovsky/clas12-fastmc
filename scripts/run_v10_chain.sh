#!/bin/bash
# v10 training chain for ifarm2402
# Usage:  ./run_v10_chain.sh inb
#         ./run_v10_chain.sh outb
# Starts from existing .dat files in /volatile/.../v10/dat/
set -euo pipefail

TAG=${1:?Usage: $0 <inb|outb>}
[[ "$TAG" == "inb" || "$TAG" == "outb" ]] || { echo "ERROR: must be inb or outb"; exit 1; }

# ── paths ──
PY=/work/clas12/vpk/fast_MC/venv/bin/python
CODE=$HOME/fastmc/scripts
V10=/volatile/clas12/vpk/fastmc/phi/rga_fall2018_${TAG}/v10

TRAIN_ALL=$V10/dat/phi_tm_train.dat
TRAIN_HAD=$V10/dat/phi_tm_hadgated_train.dat
VAL=$V10/dat/phi_tm_val.dat
CUTS=$V10/cuts/matching_cuts_phi_${TAG}.json

# ── sanity ──
for f in "$TRAIN_ALL" "$TRAIN_HAD" "$VAL"; do
    [ -f "$f" ] || { echo "ERROR: missing $f"; exit 1; }
done
[ -x "$PY" ]                              || { echo "ERROR: python not at $PY"; exit 1; }
[ -f "$CODE/train_single_particle.py" ]   || { echo "ERROR: code not in $CODE"; exit 1; }

# ── limit threads (4 procs x 4 threads = 16, fair on shared node) ──
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TORCH_NUM_THREADS=4

# ── clean previous output ──
rm -f "$V10/models"/*.pt
rm -f "$V10/plots"/*.pdf
rm -f "$V10/report"/*.log
mkdir -p "$V10"/{cuts,models,plots,report}

LOG=$V10/report/CHAIN.log
exec > >(tee -a "$LOG") 2>&1

cd "$CODE"

echo "===== v10 ${TAG} chain start: $(date) ====="
echo "  CODE:    $CODE"
echo "  DATA:    $V10/dat/"
echo "  OUTPUT:  $V10/"
echo "  PYTHON:  $($PY --version 2>&1)"
echo "  THREADS: $OMP_NUM_THREADS per process"
ls -lh "$TRAIN_ALL" "$TRAIN_HAD" "$VAL"
echo ""

# ── Step 1: resolution JSON ──
echo "--- Step 1: build resolution JSON ---"
$PY build_matching_cuts.py "$VAL" \
    -o "$CUTS" --n_sigma 10 \
    > "$V10/report/json.log" 2>&1
echo "--- JSON done: $(date) ---"
echo ""

# ── Step 2: train 4 MLPs in parallel ──
echo "--- Step 2: train 4 MLPs ---"
PIDS=()

$PY train_single_particle.py "$TRAIN_ALL" \
    -o "$V10/models" --particle_index 0 \
    --matching_cuts "$CUTS" --sanity_nsigma 20 \
    > "$V10/report/train_e.log" 2>&1 &
PIDS+=($!); echo "  e-  PID $!"

for I in 1 2 3; do
    $PY train_single_particle.py "$TRAIN_HAD" \
        -o "$V10/models" --particle_index $I \
        --matching_cuts "$CUTS" --sanity_nsigma 10 \
        > "$V10/report/train_${I}.log" 2>&1 &
    PIDS+=($!); echo "  idx=$I  PID $!"
done

echo "  waiting for: ${PIDS[*]}"
FAIL=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || { echo "  ** PID $pid FAILED **"; FAIL=1; }
done
echo "--- training done: $(date) ---"
ls -lh "$V10/models/"*.pt 2>/dev/null || echo "  WARNING: no .pt files"
echo ""

[ "$FAIL" -eq 0 ] || { echo "===== FAILED: $(date) ====="; exit 1; }

# ── Step 3: validate ──
echo "--- Step 3: validate ---"
touch /tmp/placeholder_grid.npz
$PY validate_event_full.py "$VAL" \
    --mlp_dir "$V10/models" \
    --grid /tmp/placeholder_grid.npz \
    -o "$V10/plots" \
    > "$V10/report/validate.log" 2>&1 \
    || echo "  (validate non-zero exit — Grid PDF junk, check MLP PDF)"
echo "--- validate done: $(date) ---"
ls -lh "$V10/plots/"*.pdf 2>/dev/null
echo ""
echo "===== v10 ${TAG} chain COMPLETE: $(date) ====="
