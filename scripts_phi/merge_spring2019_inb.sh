#!/bin/bash
# Merge rga_spring2019_1_inb + rga_spring2019_2_inb → rga_spring2019_inb
# then retrain v11 models on the merged data.
#
# Run on ifarm2402:  ./merge_spring2019_inb.sh
set -euo pipefail

PY=/work/clas12/vpk/fast_MC/venv/bin/python
CODE=$HOME/fastmc/scripts_phi
BASE=/volatile/clas12/vpk/fastmc/phi
S1=$BASE/rga_spring2019_1_inb/v11
S2=$BASE/rga_spring2019_2_inb/v11
OUT=$BASE/rga_spring2019_inb/v11

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TORCH_NUM_THREADS=4

# ── Create output dirs ──
mkdir -p "$OUT"/{dat,cuts,models,plots,report}

LOG=$OUT/report/CHAIN.log
exec > >(tee -a "$LOG") 2>&1

echo "===== merge rga_spring2019_inb start: $(date) ====="
echo "  Source 1: $S1"
echo "  Source 2: $S2"
echo "  Output:   $OUT"
echo ""

# ── Step 1: Merge .dat files (skip header from second file) ──
echo "--- Step 1: merge .dat files ---"
HEADER_LINES=8

for suffix in train val hadgated_train hadgated_val; do
    f1="$S1/dat/phi_tm_${suffix}.dat"
    f2="$S2/dat/phi_tm_${suffix}.dat"
    fo="$OUT/dat/phi_tm_${suffix}.dat"
    echo "  merging $suffix ..."
    cp "$f1" "$fo"
    tail -n +$((HEADER_LINES + 1)) "$f2" >> "$fo"
    echo "    $(wc -l < "$f1") + $(wc -l < "$f2") - $HEADER_LINES header = $(wc -l < "$fo") lines"
done
echo "--- merge done: $(date) ---"
ls -lh "$OUT/dat/"phi_tm_*.dat
echo ""

# ── Step 2: Build resolution JSON ──
CUTS=$OUT/cuts/matching_cuts.json
echo "--- Step 2: build resolution JSON ---"
cd "$CODE"
$PY build_matching_cuts.py "$OUT/dat/phi_tm_val.dat" \
    -o "$CUTS" --n_sigma 10 \
    > "$OUT/report/build_cuts.log" 2>&1
echo "--- JSON done: $(date) ---"
echo ""

# ── Step 3: Train 4 MLPs in parallel ──
echo "--- Step 3: train 4 MLPs ---"
PIDS=()

$PY train_single_particle.py "$OUT/dat/phi_tm_train.dat" \
    -o "$OUT/models" --particle_index 0 \
    --matching_cuts "$CUTS" --sanity_nsigma 20 \
    > "$OUT/report/train_e.log" 2>&1 &
PIDS+=($!); echo "  e-  PID $!"

for I in 1 2 3; do
    $PY train_single_particle.py "$OUT/dat/phi_tm_hadgated_train.dat" \
        -o "$OUT/models" --particle_index $I \
        --matching_cuts "$CUTS" --sanity_nsigma 10 \
        > "$OUT/report/train_${I}.log" 2>&1 &
    PIDS+=($!); echo "  idx=$I  PID $!"
done

echo "  waiting for: ${PIDS[*]}"
FAIL=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || { echo "  ** PID $pid FAILED **"; FAIL=1; }
done
echo "--- training done: $(date) ---"
ls -lh "$OUT/models/"*.pt 2>/dev/null || echo "  WARNING: no .pt files"
echo ""
[ "$FAIL" -eq 0 ] || { echo "===== FAILED: $(date) ====="; exit 1; }

# ── Step 4: Validate ──
echo "--- Step 4: validate ---"
touch /tmp/placeholder_grid.npz
$PY validate_event_full.py "$OUT/dat/phi_tm_val.dat" \
    --mlp_dir "$OUT/models" \
    --grid /tmp/placeholder_grid.npz \
    -o "$OUT/plots" \
    > "$OUT/report/validate.log" 2>&1 \
    || echo "  (validate non-zero exit — Grid PDF junk, check MLP PDF)"
echo "--- validate done: $(date) ---"
ls -lh "$OUT/plots/"*.pdf 2>/dev/null
echo ""

# ── Write params.json ──
cat > "$OUT/params.json" <<'PARAMS'
{
  "channel":              "phi",
  "period":               "rga_spring2019",
  "polarity":             "inb",
  "version":              "v11",
  "beam_energy":          10.2,
  "beam_pid":             11,
  "target_pid":           2212,
  "min_electron_theta":   6.0,
  "quality":              0.98,
  "val_fraction":         0.20,
  "seed":                 42,
  "n_sigma":              10,
  "sanity_nsigma_e":      20,
  "sanity_nsigma_h":      10,
  "note":                 "merged from rga_spring2019_1_inb + rga_spring2019_2_inb"
}
PARAMS

echo "===== merge rga_spring2019_inb COMPLETE: $(date) ====="
