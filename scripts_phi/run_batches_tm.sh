#!/bin/bash
# Process HIPO files in batches using the TruthMatch generator
# (make_training_data_truthmatch.py).  Robust against single-file crashes
# (bad HIPO file -> drop that batch's partial output, continue).
#
# Produces 4 final files in <output_dir>:
#   phi_tm_train.dat
#   phi_tm_val.dat
#   phi_tm_hadgated_train.dat
#   phi_tm_hadgated_val.dat
#
# Usage:
#   ./run_batches_tm.sh <hipo_dir> <output_dir> [batch_size]
#
# Examples:
#   outbending (default theta_e>=4):
#     ./run_batches_tm.sh rga_fall2018_outb_hipo rga_fall2018_outb_dat 5
#   inbending (theta_e>=6, fewer low-angle electrons):
#     ./run_batches_tm.sh rga_fall2018_inb_hipo  rga_fall2018_inb_dat  5  6.0

set -u

if [ $# -lt 2 ]; then
    echo "Usage: $0 <hipo_dir> <output_dir> [batch_size] [min_electron_theta]"
    echo "  batch_size:           number of files per batch (default: 400)"
    echo "  min_electron_theta:   gen e- theta cut, deg (default: 4.0)"
    exit 1
fi

HIPO_DIR="$1"
OUTPUT_DIR="$2"
BATCH_SIZE="${3:-400}"
MIN_E_THETA="${4:-4.0}"

# Count HIPO files (follows symlinks; -L on the dir)
N_FILES=$(find -L "$HIPO_DIR" -maxdepth 1 -name '*.hipo' -type f | wc -l)
if [ "$N_FILES" -eq 0 ]; then
    echo "Error: no *.hipo files in $HIPO_DIR"
    exit 1
fi
mkdir -p "$OUTPUT_DIR"

echo "Total HIPO files:        $N_FILES"
echo "Batch size:              $BATCH_SIZE"
echo "Output dir:              $OUTPUT_DIR"
echo "min gen e- theta (deg):  $MIN_E_THETA"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
N_BATCHES=$(( (N_FILES + BATCH_SIZE - 1) / BATCH_SIZE ))
echo "Number of batches: $N_BATCHES"
echo ""

# ── Step 1: run each batch ──────────────────────────────────────
for (( i=0; i<N_BATCHES; i++ )); do
    OFFSET=$(( i * BATCH_SIZE ))
    PART=$(printf "p%03d" $((i+1)))
    BASE="phi_tm_${PART}"
    echo "=== Batch $((i+1))/$N_BATCHES: files $((OFFSET+1))–$((OFFSET+BATCH_SIZE)),  output ${BASE} ==="

    python "$SCRIPT_DIR/make_training_data_truthmatch.py" "$HIPO_DIR" \
            --beam_id electron --beam_energy 10.6 --target_id proton \
            --min_electron_theta "$MIN_E_THETA" \
            --electron_match pidonly \
            -o "$BASE" --output_dir "$OUTPUT_DIR" \
            --max_files "$BATCH_SIZE" --file_offset "$OFFSET"
    RC=$?
    # Don't trust the exit code — HIPO C++ library has a teardown crash that
    # returns non-zero even after the python script successfully wrote files.
    # Use file existence as the truth: if _train.dat is present and has content
    # beyond the header, keep all 4; otherwise discard any partial siblings.
    TRAIN="$OUTPUT_DIR/${BASE}_train.dat"
    if [ -f "$TRAIN" ] && [ "$(grep -c '^[0-9]' "$TRAIN")" -gt 0 ]; then
        : # batch produced events, keep all 4 files
    else
        if [ $RC -ne 0 ]; then
            echo ">>> Batch $((i+1)) produced no events (exit $RC); cleaning up."
        else
            echo ">>> Batch $((i+1)) wrote no events (exit 0, empty file); cleaning up."
        fi
        rm -f "$OUTPUT_DIR/${BASE}_train.dat" \
              "$OUTPUT_DIR/${BASE}_val.dat" \
              "$OUTPUT_DIR/${BASE}_hadgated_train.dat" \
              "$OUTPUT_DIR/${BASE}_hadgated_val.dat"
    fi
    echo ""
done

# ── Step 2: merge each of the 4 file flavors ────────────────────
merge_flavor() {
    # $1 = suffix, e.g. "train" or "hadgated_train"
    local SUFFIX="$1"
    local OUT="$OUTPUT_DIR/phi_tm_${SUFFIX}.dat"
    echo "=== Merging ${SUFFIX} files -> ${OUT} ==="
    local FIRST=1
    for (( i=0; i<N_BATCHES; i++ )); do
        local PART; PART=$(printf "p%03d" $((i+1)))
        local F="$OUTPUT_DIR/phi_tm_${PART}_${SUFFIX}.dat"
        if [ ! -f "$F" ]; then
            echo "  (skip: $F not found)"
            continue
        fi
        if [ $FIRST -eq 1 ]; then
            cp "$F" "$OUT"
            FIRST=0
        else
            grep -v '^#!' "$F" >> "$OUT"        # strip per-batch header lines
        fi
    done
}

merge_flavor train
merge_flavor val
merge_flavor hadgated_train
merge_flavor hadgated_val

# ── Step 3: report ──────────────────────────────────────────────
echo ""
echo "=== Done ==="
for SUFFIX in train val hadgated_train hadgated_val; do
    F="$OUTPUT_DIR/phi_tm_${SUFFIX}.dat"
    if [ -f "$F" ]; then
        N=$(grep -c '^[0-9]' "$F" || true)
        SIZE=$(du -h "$F" | cut -f1)
        echo "  phi_tm_${SUFFIX}.dat:  ${N} events  (${SIZE})"
    fi
done
echo ""
echo "Partial per-batch files (phi_tm_pNNN_*.dat) left in place; delete with:"
echo "  rm $OUTPUT_DIR/phi_tm_p*_*.dat"
