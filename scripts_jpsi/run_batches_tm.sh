#!/bin/bash
# J/psi version — process HIPO files in batches using the TruthMatch generator
# (make_training_data_truthmatch.py).  Robust against single-file crashes
# (bad HIPO file -> drop that batch's partial output, continue).
#
# ep -> e' p J/psi, J/psi -> e+ e-
#   scattered e' : Forward Tagger (pid==11, status 1xxx), nearest in p
#   p, e+, e-    : p via TruthMatch; decay leptons pid==+-11 in FD, nearest in p
#   gen e- theta cut: NONE (FT electrons peak at theta ~ 3 deg)
#
# Produces 4 final files in <output_dir>:
#   jpsi_tm_train.dat
#   jpsi_tm_val.dat
#   jpsi_tm_hadgated_train.dat      (scattered e' found in FT)
#   jpsi_tm_hadgated_val.dat
#
# Usage:
#   ./run_batches_tm.sh <hipo_dir> <output_dir> [batch_size] [beam_energy]
#
# Examples:
#   F18/S18 (10.6 GeV):  ./run_batches_tm.sh /volatile/clas12/osg/marianat/769 out_dat 5
#   S19     (10.2 GeV):  ./run_batches_tm.sh /volatile/clas12/osg/marianat/10796 out_dat 5 10.2

set -u

if [ $# -lt 2 ]; then
    echo "Usage: $0 <hipo_dir> <output_dir> [batch_size] [beam_energy]"
    echo "  batch_size:   number of files per batch (default: 5)"
    echo "  beam_energy:  GeV (default: 10.6; use 10.2 for S19)"
    exit 1
fi

HIPO_DIR="$1"
OUTPUT_DIR="$2"
BATCH_SIZE="${3:-5}"
BEAM_ENERGY="${4:-10.6}"
MIN_E_THETA=0.0          # scattered e' is in the FT (~2.5-4.5 deg)
MAX_E_THETA=7.0          # FT electron: drop events with gen e' theta > 7 deg
PY="${PY:-/work/clas12/vpk/fast_MC/venv/bin/python}"

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
echo "Beam energy (GeV):       $BEAM_ENERGY"
echo "gen e- theta (deg):      [$MIN_E_THETA, $MAX_E_THETA]"
echo "Electron match:          pidonly_ft (e' in FT, decay e+/e- in FD)"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
N_BATCHES=$(( (N_FILES + BATCH_SIZE - 1) / BATCH_SIZE ))
echo "Number of batches: $N_BATCHES"
echo ""

# ── Step 1: run each batch ──────────────────────────────────────
for (( i=0; i<N_BATCHES; i++ )); do
    OFFSET=$(( i * BATCH_SIZE ))
    PART=$(printf "p%03d" $((i+1)))
    BASE="jpsi_tm_${PART}"
    echo "=== Batch $((i+1))/$N_BATCHES: files $((OFFSET+1))–$((OFFSET+BATCH_SIZE)),  output ${BASE} ==="

    "$PY" "$SCRIPT_DIR/make_training_data_truthmatch.py" "$HIPO_DIR" \
            --beam_id electron --beam_energy "$BEAM_ENERGY" --target_id proton \
            --min_electron_theta "$MIN_E_THETA" \
            --max_electron_theta "$MAX_E_THETA" \
            --electron_match pidonly_ft \
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
    local OUT="$OUTPUT_DIR/jpsi_tm_${SUFFIX}.dat"
    echo "=== Merging ${SUFFIX} files -> ${OUT} ==="
    local FIRST=1
    for (( i=0; i<N_BATCHES; i++ )); do
        local PART; PART=$(printf "p%03d" $((i+1)))
        local F="$OUTPUT_DIR/jpsi_tm_${PART}_${SUFFIX}.dat"
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
    F="$OUTPUT_DIR/jpsi_tm_${SUFFIX}.dat"
    if [ -f "$F" ]; then
        N=$(grep -c '^[0-9]' "$F" || true)
        SIZE=$(du -h "$F" | cut -f1)
        echo "  jpsi_tm_${SUFFIX}.dat:  ${N} events  (${SIZE})"
    fi
done
echo ""
echo "Partial per-batch files (jpsi_tm_pNNN_*.dat) left in place; delete with:"
echo "  rm $OUTPUT_DIR/jpsi_tm_p*_*.dat"
