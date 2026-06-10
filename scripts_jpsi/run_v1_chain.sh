#!/bin/bash
# J/psi v1 training chain for ifarm2402: JSON -> Train 4 MLPs -> Validate
# Starts from existing .dat files in /volatile/.../jpsi/<dataset>/v1/dat/
#
# Usage:  ./run_v1_chain.sh F18in_45nA
#         ./run_v1_chain.sh S19in_50nA
#
# Particle indices (epe+e-):   0 = e- scattered (FT)  -> model e-.pt
#                              1 = p (FD)             -> model p.pt
#                              2 = e+ (FD)            -> model e+.pt
#                              3 = e- decay (FD)      -> model e-d.pt
set -euo pipefail

DS=${1:?Usage: $0 <dataset>   e.g. F18in_45nA}

# ── paths ──
PY=/work/clas12/vpk/fast_MC/venv/bin/python
CODE=$HOME/fastmc/scripts_jpsi
V1=/volatile/clas12/vpk/fastmc/jpsi/$DS/v1

TRAIN_ALL=$V1/dat/jpsi_tm_train.dat
TRAIN_HAD=$V1/dat/jpsi_tm_hadgated_train.dat
VAL=$V1/dat/jpsi_tm_val.dat
CUTS=$V1/cuts/matching_cuts_jpsi_${DS}.json

# ── sanity ──
for f in "$TRAIN_ALL" "$TRAIN_HAD" "$VAL"; do
    [ -f "$f" ] || { echo "ERROR: missing $f"; exit 1; }
done
[ -x "$PY" ]                              || { echo "ERROR: python not at $PY"; exit 1; }
[ -f "$CODE/train_single_particle.py" ]   || { echo "ERROR: code not in $CODE"; exit 1; }

# ── limit threads (4 procs x 4 threads, fair on shared node) ──
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TORCH_NUM_THREADS=4

# ── clean previous output ──
rm -f "$V1/models"/*.pt
rm -f "$V1/plots"/validation_*.pdf
mkdir -p "$V1"/{cuts,models,plots,report}

LOG=$V1/report/CHAIN.log
exec > >(tee -a "$LOG") 2>&1

cd "$CODE"

echo "===== jpsi v1 ${DS} chain start: $(date) ====="
echo "  CODE:    $CODE"
echo "  DATA:    $V1/dat/"
echo "  OUTPUT:  $V1/"
echo "  PYTHON:  $($PY --version 2>&1)"
ls -lh "$TRAIN_ALL" "$TRAIN_HAD" "$VAL"
echo ""

# ── Step 2: resolution JSON (FD + FT; species x det separates the two e-) ──
echo "--- Step 2: build resolution JSON ---"
$PY build_matching_cuts.py "$VAL" \
    -o "$CUTS" --n_sigma 10 \
    > "$V1/report/json.log" 2>&1
echo "--- JSON done: $(date) ---"
echo ""

# ── Step 3: train 4 MLPs in parallel ──
echo "--- Step 3: train 4 MLPs ---"
PIDS=()

# Scattered electron (FT), full file, 20-sigma sanity cut
$PY train_single_particle.py "$TRAIN_ALL" \
    -o "$V1/models" --particle_index 0 --model_name "e-" \
    --matching_cuts "$CUTS" --sanity_nsigma 20 \
    > "$V1/report/train_e.log" 2>&1 &
PIDS+=($!); echo "  e- (FT)   PID $!"

# Hadrons + decay leptons (FD) on the FT-electron-gated file, 10-sigma cut
declare -A MODEL_NAME=( [1]="p" [2]="e+" [3]="e-d" )
for I in 1 2 3; do
    $PY train_single_particle.py "$TRAIN_HAD" \
        -o "$V1/models" --particle_index $I --model_name "${MODEL_NAME[$I]}" \
        --matching_cuts "$CUTS" --sanity_nsigma 10 \
        > "$V1/report/train_${MODEL_NAME[$I]}.log" 2>&1 &
    PIDS+=($!); echo "  idx=$I (${MODEL_NAME[$I]})  PID $!"
done

echo "  waiting for: ${PIDS[*]}"
FAIL=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || { echo "  ** PID $pid FAILED **"; FAIL=1; }
done
echo "--- training done: $(date) ---"
ls -lh "$V1/models/"*.pt 2>/dev/null || echo "  WARNING: no .pt files"
echo ""

[ "$FAIL" -eq 0 ] || { echo "===== FAILED: $(date) ====="; exit 1; }

# ── Step 4: validate ──
echo "--- Step 4: validate ---"
touch /tmp/placeholder_grid.npz
$PY validate_event_full.py "$VAL" \
    --mlp_dir "$V1/models" \
    --grid /tmp/placeholder_grid.npz \
    -o "$V1/plots" \
    > "$V1/report/validate.log" 2>&1 \
    || echo "  (validate non-zero exit — Grid PDF junk, check MLP PDF)"
echo "--- validate done: $(date) ---"
ls -lh "$V1/plots/"*.pdf 2>/dev/null
echo ""
echo "===== jpsi v1 ${DS} chain COMPLETE: $(date) ====="
