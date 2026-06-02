#!/bin/bash
# run_batches_cpp.sh — batch-isolated wrapper around the C++ make_training_data
# binary.  HIPO C++ vector assertions occasionally abort() a process;  by
# running the binary per-batch we localize each crash to a small number of
# files instead of killing the whole run.
#
# Usage:
#   ./run_batches_cpp.sh <hipo_dir> <output_dir> [batch_size] [min_e_theta]
#
# Examples:
#   ./run_batches_cpp.sh rga_fall2018_inb_hipo  /work/clas12/vpk/cpp_dat/inb  5  6.0
#   ./run_batches_cpp.sh rga_fall2018_outb_hipo /work/clas12/vpk/cpp_dat/outb 5  4.0

set -u

if [ $# -lt 2 ]; then
    echo "Usage: $0 <hipo_dir> <output_dir> [batch_size] [min_e_theta]" >&2
    echo "  batch_size:    files per batch (default 5)" >&2
    echo "  min_e_theta:   gen e- theta cut, deg (default 6.0)" >&2
    exit 2
fi

HIPO_DIR="$1"
OUTPUT_DIR="$2"
BATCH_SIZE="${3:-5}"
MIN_E_THETA="${4:-6.0}"

CPP_BIN="${CPP_BIN:-/home/vpk/fastmc/cpp/build/make_training_data}"
REACTION="${REACTION:-epK+K-}"
BEAM_ENERGY="${BEAM_ENERGY:-10.6}"
BEAM_PID="${BEAM_PID:-11}"
TARGET_PID="${TARGET_PID:-2212}"
QUALITY="${QUALITY:-0.98}"
VAL_FRACTION="${VAL_FRACTION:-0.20}"
SEED="${SEED:-42}"

if [ ! -x "$CPP_BIN" ]; then
    echo "Error: C++ binary not executable: $CPP_BIN" >&2
    exit 2
fi

# Count HIPO files (follow symlinks)
N_FILES=$(find -L "$HIPO_DIR" -maxdepth 1 -name '*.hipo' -type f | wc -l)
if [ "$N_FILES" -eq 0 ]; then
    echo "Error: no *.hipo files in $HIPO_DIR" >&2
    exit 2
fi
mkdir -p "$OUTPUT_DIR"

N_BATCHES=$(( (N_FILES + BATCH_SIZE - 1) / BATCH_SIZE ))
echo "C++ binary:             $CPP_BIN"
echo "HIPO files total:       $N_FILES"
echo "Batch size:             $BATCH_SIZE"
echo "Number of batches:      $N_BATCHES"
echo "Output dir:             $OUTPUT_DIR"
echo "Reaction:               $REACTION"
echo "Beam energy:            $BEAM_ENERGY"
echo "min gen e- theta (deg): $MIN_E_THETA"
echo

# ─── Per-batch loop ──────────────────────────────────────────────────────
for (( i=0; i<N_BATCHES; i++ )); do
    OFFSET=$(( i * BATCH_SIZE ))
    PART=$(printf "p%04d" $((i+1)))
    BASE="phi_tm_${PART}"
    echo "=== Batch $((i+1))/$N_BATCHES: files $((OFFSET+1))–$((OFFSET+BATCH_SIZE)), output ${BASE} ==="

    "$CPP_BIN" "$HIPO_DIR" "$OUTPUT_DIR" \
        --reaction "$REACTION" \
        --beam_energy "$BEAM_ENERGY" \
        --beam_pid "$BEAM_PID" --target_pid "$TARGET_PID" \
        --min_electron_theta "$MIN_E_THETA" \
        --quality "$QUALITY" --val_fraction "$VAL_FRACTION" \
        --seed "$SEED" \
        --basename "$BASE" \
        --max_files "$BATCH_SIZE" --file_offset "$OFFSET"
    RC=$?

    TRAIN="$OUTPUT_DIR/${BASE}_train.dat"
    if [ -f "$TRAIN" ] && [ "$(grep -c '^[0-9]' "$TRAIN")" -gt 0 ]; then
        : # batch produced events, keep all 4 files
    else
        echo ">>> Batch $((i+1)) produced no events (exit $RC); cleaning up."
        rm -f "$OUTPUT_DIR/${BASE}_train.dat" \
              "$OUTPUT_DIR/${BASE}_val.dat" \
              "$OUTPUT_DIR/${BASE}_hadgated_train.dat" \
              "$OUTPUT_DIR/${BASE}_hadgated_val.dat"
    fi
done

# ─── Merge ───────────────────────────────────────────────────────────────
merge_flavor() {
    local SUFFIX="$1"
    local OUT="$OUTPUT_DIR/phi_tm_${SUFFIX}.dat"
    echo "=== Merging ${SUFFIX} -> ${OUT} ==="
    local FIRST=1
    for (( i=0; i<N_BATCHES; i++ )); do
        local PART; PART=$(printf "p%04d" $((i+1)))
        local F="$OUTPUT_DIR/phi_tm_${PART}_${SUFFIX}.dat"
        [ -f "$F" ] || continue
        if [ $FIRST -eq 1 ]; then cp "$F" "$OUT"; FIRST=0
        else grep -v '^#!' "$F" >> "$OUT"
        fi
    done
}

merge_flavor train
merge_flavor val
merge_flavor hadgated_train
merge_flavor hadgated_val

# ─── Summary ─────────────────────────────────────────────────────────────
echo
echo "=== Done ==="
for SUFFIX in train val hadgated_train hadgated_val; do
    F="$OUTPUT_DIR/phi_tm_${SUFFIX}.dat"
    if [ -f "$F" ]; then
        N=$(grep -c '^[0-9]' "$F" || true)
        SIZE=$(du -h "$F" | cut -f1)
        printf "  phi_tm_%-20s %10d events  (%s)\n" "${SUFFIX}.dat:" "$N" "$SIZE"
    fi
done
echo
echo "Cleanup per-batch files with:"
echo "  find $OUTPUT_DIR -maxdepth 1 -name 'phi_tm_p*_*.dat' -delete"
