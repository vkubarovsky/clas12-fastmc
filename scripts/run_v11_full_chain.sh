#!/bin/bash
# v11 FULL chain: HIPO → DAT → JSON → Train → Validate
# Usage:  ./run_v11_full_chain.sh inb
#         ./run_v11_full_chain.sh outb
#
# v11 = v10 + electron matching fixed to pidonly (pid=11 in FD).
# Hadrons unchanged: TruthMatch quality > 0.98.
#
# Expects:
#   /volatile/.../rga_fall2018_{inb|outb}/v11/hipo/  ← symlink to HIPO files
#   ~/fastmc/scripts/                                ← all python scripts
set -euo pipefail

TAG=${1:?Usage: $0 <inb|outb>}
[[ "$TAG" == "inb" || "$TAG" == "outb" ]] || { echo "ERROR: must be inb or outb"; exit 1; }

# ── paths ──
PY=/work/clas12/vpk/fast_MC/venv/bin/python
CODE=$HOME/fastmc/scripts
V11=/volatile/clas12/vpk/fastmc/phi/rga_fall2018_${TAG}/v11

# ── theta cut: 6° inbending, 4° outbending ──
if [ "$TAG" == "inb" ]; then
    MIN_E_THETA=6.0
else
    MIN_E_THETA=4.0
fi

# ── sanity checks ──
[ -x "$PY" ]                              || { echo "ERROR: python not at $PY"; exit 1; }
[ -f "$CODE/train_single_particle.py" ]   || { echo "ERROR: code not in $CODE"; exit 1; }
[ -d "$V11/hipo/hipo_dir" ]               || { echo "ERROR: $V11/hipo/hipo_dir not found — run setup_v11.sh first"; exit 1; }
NHIPO=$(ls "$V11/hipo/hipo_dir/"*.hipo 2>/dev/null | wc -l)
[ "$NHIPO" -gt 0 ]                       || { echo "ERROR: no .hipo files in $V11/hipo/"; exit 1; }

# ── limit CPU threads ──
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TORCH_NUM_THREADS=4

# ── directories ──
mkdir -p "$V11"/{dat,cuts,models,plots,report}

LOG=$V11/report/CHAIN.log
exec > >(tee -a "$LOG") 2>&1

cd "$CODE"

echo "===== v11 ${TAG} FULL chain start: $(date) ====="
echo "  CODE:      $CODE"
echo "  HIPO:      $V11/hipo/  ($NHIPO files)"
echo "  OUTPUT:    $V11/"
echo "  PYTHON:    $($PY --version 2>&1)"
echo "  THREADS:   $OMP_NUM_THREADS per process"
echo "  THETA CUT: >= ${MIN_E_THETA}°"
echo "  ELECTRON:  pidonly (pid=11 in FD)"
echo "  HADRONS:   TruthMatch quality > 0.98"
echo ""

# ══════════════════════════════════════════════════════════════════
# Step 1: HIPO → DAT (batched, robust against HIPO reader crashes)
# ══════════════════════════════════════════════════════════════════
echo "--- Step 1: HIPO -> DAT (batched) ---"
BATCH_SIZE=5
./run_batches_tm.sh "$V11/hipo/hipo_dir" "$V11/dat" "$BATCH_SIZE" "$MIN_E_THETA" \
    > "$V11/report/hipo2dat.log" 2>&1
echo "--- HIPO->DAT done: $(date) ---"

# Verify outputs
for f in phi_tm_train.dat phi_tm_val.dat phi_tm_hadgated_train.dat phi_tm_hadgated_val.dat; do
    [ -f "$V11/dat/$f" ] || { echo "ERROR: $V11/dat/$f not produced"; exit 1; }
done
ls -lh "$V11/dat/"phi_tm_*.dat
echo ""

# ══════════════════════════════════════════════════════════════════
# Step 2: Build resolution JSON
# ══════════════════════════════════════════════════════════════════
CUTS=$V11/cuts/matching_cuts_phi_${TAG}.json
echo "--- Step 2: build resolution JSON ---"
$PY build_matching_cuts.py "$V11/dat/phi_tm_val.dat" \
    -o "$CUTS" --n_sigma 10 \
    > "$V11/report/json.log" 2>&1
echo "--- JSON done: $(date) ---"
echo ""

# ══════════════════════════════════════════════════════════════════
# Step 3: Train 4 MLPs in parallel
# ══════════════════════════════════════════════════════════════════
echo "--- Step 3: train 4 MLPs ---"
rm -f "$V11/models"/*.pt
PIDS=()

# Electron on full file, 20-sigma sanity cut
$PY train_single_particle.py "$V11/dat/phi_tm_train.dat" \
    -o "$V11/models" --particle_index 0 \
    --matching_cuts "$CUTS" --sanity_nsigma 20 \
    > "$V11/report/train_e.log" 2>&1 &
PIDS+=($!); echo "  e-  PID $!"

# Hadrons on electron-gated file, 10-sigma sanity cut
for I in 1 2 3; do
    $PY train_single_particle.py "$V11/dat/phi_tm_hadgated_train.dat" \
        -o "$V11/models" --particle_index $I \
        --matching_cuts "$CUTS" --sanity_nsigma 10 \
        > "$V11/report/train_${I}.log" 2>&1 &
    PIDS+=($!); echo "  idx=$I  PID $!"
done

echo "  waiting for: ${PIDS[*]}"
FAIL=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || { echo "  ** PID $pid FAILED **"; FAIL=1; }
done
echo "--- training done: $(date) ---"
ls -lh "$V11/models/"*.pt 2>/dev/null || echo "  WARNING: no .pt files"
echo ""

[ "$FAIL" -eq 0 ] || { echo "===== FAILED: $(date) ====="; exit 1; }

# ══════════════════════════════════════════════════════════════════
# Step 4: Validate
# ══════════════════════════════════════════════════════════════════
echo "--- Step 4: validate ---"
touch /tmp/placeholder_grid.npz
$PY validate_event_full.py "$V11/dat/phi_tm_val.dat" \
    --mlp_dir "$V11/models" \
    --grid /tmp/placeholder_grid.npz \
    -o "$V11/plots" \
    > "$V11/report/validate.log" 2>&1 \
    || echo "  (validate non-zero exit — Grid PDF junk, check MLP PDF)"
echo "--- validate done: $(date) ---"
ls -lh "$V11/plots/"*.pdf 2>/dev/null
echo ""
echo "===== v11 ${TAG} FULL chain COMPLETE: $(date) ====="
