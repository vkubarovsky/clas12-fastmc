#!/bin/bash
# fastmc.sh — wrapper for the Fast MC pipeline.
#
# Organizes the output dir tree per (channel, period, polarity, version), then
# invokes the right tool per sub-command:
#   build-dat   →  C++ binary make_training_data
#   build-cuts  →  Python build_matching_cuts.py
#   train       →  Python train_single_particle.py (× 4)
#   validate    →  Python validate_event_full.py
#   all         →  run all four in order
#
# Dir layout (per call):
#   <root>/<channel>/<period>_<polarity>/v<NN>/
#       dat/  cuts/  models/  plots/  report/  params.json
#
# Usage examples:
#   fastmc.sh build-dat  --channel phi --period rga_fall2018 --polarity inb \
#                        --version v11 --hipo-dir /volatile/.../inb \
#                        --beam-energy 10.6 --min-electron-theta 6.0
#
#   fastmc.sh all        --channel phi --period rga_fall2018 --polarity inb \
#                        --version v11 --hipo-dir /volatile/.../inb \
#                        --beam-energy 10.6 --min-electron-theta 6.0

set -euo pipefail

# ─── Defaults (tweak as needed) ──────────────────────────────────────────
ROOT_DEFAULT=/volatile/clas12/vpk/fastmc
CPP_BIN_DEFAULT=/home/vpk/fastmc/cpp/build/make_training_data
PY_SCRIPTS_DEFAULT=/home/vpk/2026_fastMC/train_fast_mc
PY_INTERP_DEFAULT=/work/clas12/vpk/fast_MC/venv/bin/python
PLACEHOLDER_GRID=/tmp/placeholder_grid.npz

# ─── Parse global args ───────────────────────────────────────────────────
CMD=""
CHANNEL=""
PERIOD=""
POLARITY=""
VERSION=""
HIPO_DIR=""
BEAM_ENERGY=10.6
BEAM_PID=11
TARGET_PID=2212
MIN_ETHETA=6.0
REACTION=epK+K-
QUALITY=0.98
VAL_FRACTION=0.20
SEED=42
BATCH_SIZE=5
MAX_FILES=0
FILE_OFFSET=0
N_SIGMA=10
SANITY_NSIGMA_E=20
SANITY_NSIGMA_H=10
ROOT="${ROOT_DEFAULT}"
CPP_BIN="${CPP_BIN_DEFAULT}"
PY_SCRIPTS="${PY_SCRIPTS_DEFAULT}"
PY="${PY_INTERP_DEFAULT}"

if [ $# -lt 1 ]; then
    echo "Usage: $0 {build-dat|build-cuts|train|validate|all} [options]" >&2
    exit 2
fi
CMD="$1"; shift

while [ $# -gt 0 ]; do
    case "$1" in
        --channel)              CHANNEL="$2"; shift 2 ;;
        --period)               PERIOD="$2"; shift 2 ;;
        --polarity)             POLARITY="$2"; shift 2 ;;
        --version)              VERSION="$2"; shift 2 ;;
        --hipo-dir)             HIPO_DIR="$2"; shift 2 ;;
        --beam-energy)          BEAM_ENERGY="$2"; shift 2 ;;
        --beam-pid)             BEAM_PID="$2"; shift 2 ;;
        --target-pid)           TARGET_PID="$2"; shift 2 ;;
        --min-electron-theta)   MIN_ETHETA="$2"; shift 2 ;;
        --reaction)             REACTION="$2"; shift 2 ;;
        --quality)              QUALITY="$2"; shift 2 ;;
        --val-fraction)         VAL_FRACTION="$2"; shift 2 ;;
        --seed)                 SEED="$2"; shift 2 ;;
        --batch-size)           BATCH_SIZE="$2"; shift 2 ;;
        --max-files)            MAX_FILES="$2"; shift 2 ;;
        --file-offset)          FILE_OFFSET="$2"; shift 2 ;;
        --n-sigma)              N_SIGMA="$2"; shift 2 ;;
        --sanity-nsigma-e)      SANITY_NSIGMA_E="$2"; shift 2 ;;
        --sanity-nsigma-h)      SANITY_NSIGMA_H="$2"; shift 2 ;;
        --root)                 ROOT="$2"; shift 2 ;;
        --cpp-bin)              CPP_BIN="$2"; shift 2 ;;
        --py-scripts)           PY_SCRIPTS="$2"; shift 2 ;;
        --python)               PY="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

# ─── Validate ────────────────────────────────────────────────────────────
[ -n "$CHANNEL"  ] || { echo "Need --channel"  >&2; exit 2; }
[ -n "$PERIOD"   ] || { echo "Need --period"   >&2; exit 2; }
[ -n "$POLARITY" ] || { echo "Need --polarity" >&2; exit 2; }
[ -n "$VERSION"  ] || { echo "Need --version"  >&2; exit 2; }

# ─── Compute paths ───────────────────────────────────────────────────────
OUT="${ROOT}/${CHANNEL}/${PERIOD}_${POLARITY}/${VERSION}"
DAT_DIR="${OUT}/dat"
CUTS_DIR="${OUT}/cuts"
MODELS_DIR="${OUT}/models"
PLOTS_DIR="${OUT}/plots"
REPORT_DIR="${OUT}/report"
PARAMS_FILE="${OUT}/params.json"

mkdir -p "$DAT_DIR" "$CUTS_DIR" "$MODELS_DIR" "$PLOTS_DIR" "$REPORT_DIR"

CUTS_JSON="${CUTS_DIR}/matching_cuts.json"
BASENAME="phi_tm"   # could be parameterized later per channel

# ─── Snapshot the call into params.json (best effort) ─────────────────────
write_params() {
    cat > "$PARAMS_FILE" <<EOF
{
  "channel":              "${CHANNEL}",
  "period":               "${PERIOD}",
  "polarity":             "${POLARITY}",
  "version":              "${VERSION}",
  "hipo_dir":             "${HIPO_DIR}",
  "reaction":             "${REACTION}",
  "beam_energy":          ${BEAM_ENERGY},
  "beam_pid":             ${BEAM_PID},
  "target_pid":           ${TARGET_PID},
  "min_electron_theta":   ${MIN_ETHETA},
  "quality":              ${QUALITY},
  "val_fraction":         ${VAL_FRACTION},
  "seed":                 ${SEED},
  "batch_size":           ${BATCH_SIZE},
  "n_sigma":              ${N_SIGMA},
  "sanity_nsigma_e":      ${SANITY_NSIGMA_E},
  "sanity_nsigma_h":      ${SANITY_NSIGMA_H}
}
EOF
}

# ─── Step implementations ────────────────────────────────────────────────
do_build_dat() {
    [ -n "$HIPO_DIR" ] || { echo "Need --hipo-dir for build-dat" >&2; exit 2; }
    [ -x "$CPP_BIN" ]  || { echo "C++ binary not found / not executable: $CPP_BIN" >&2; exit 2; }
    write_params
    echo "[build-dat] writing dat files to $DAT_DIR"
    "$CPP_BIN" "$HIPO_DIR" "$DAT_DIR" \
        --reaction "$REACTION" --beam_energy "$BEAM_ENERGY" \
        --beam_pid "$BEAM_PID" --target_pid "$TARGET_PID" \
        --min_electron_theta "$MIN_ETHETA" \
        --quality "$QUALITY" --val_fraction "$VAL_FRACTION" \
        --basename "$BASENAME" \
        --max_files "$MAX_FILES" --file_offset "$FILE_OFFSET" \
        --seed "$SEED" \
        > "$REPORT_DIR/build_dat.log" 2>&1
    echo "[build-dat] done. See $REPORT_DIR/build_dat.log"
}

do_build_cuts() {
    [ -f "$DAT_DIR/${BASENAME}_val.dat" ] || { echo "No val.dat — run build-dat first" >&2; exit 2; }
    write_params
    echo "[build-cuts] writing $CUTS_JSON"
    "$PY" "$PY_SCRIPTS/build_matching_cuts.py" \
        "$DAT_DIR/${BASENAME}_val.dat" \
        -o "$CUTS_JSON" --n_sigma "$N_SIGMA" \
        > "$REPORT_DIR/build_cuts.log" 2>&1
    echo "[build-cuts] done. See $REPORT_DIR/build_cuts.log"
}

do_train() {
    [ -f "$CUTS_JSON" ] || { echo "No cuts JSON — run build-cuts first" >&2; exit 2; }
    write_params
    echo "[train] training 4 MLPs into $MODELS_DIR"
    # electron — full file
    "$PY" "$PY_SCRIPTS/train_single_particle.py" \
        "$DAT_DIR/${BASENAME}_train.dat" \
        -o "$MODELS_DIR" --particle_index 0 \
        --matching_cuts "$CUTS_JSON" --sanity_nsigma "$SANITY_NSIGMA_E" \
        > "$REPORT_DIR/train_e.log" 2>&1 &
    PE=$!
    # hadrons — hadgated file
    for I in 1 2 3; do
        "$PY" "$PY_SCRIPTS/train_single_particle.py" \
            "$DAT_DIR/${BASENAME}_hadgated_train.dat" \
            -o "$MODELS_DIR" --particle_index "$I" \
            --matching_cuts "$CUTS_JSON" --sanity_nsigma "$SANITY_NSIGMA_H" \
            > "$REPORT_DIR/train_$I.log" 2>&1 &
    done
    wait
    echo "[train] done. See $REPORT_DIR/train_*.log"
}

do_validate() {
    [ -f "$MODELS_DIR/e-.pt" ] || { echo "No e-.pt — run train first" >&2; exit 2; }
    write_params
    touch "$PLACEHOLDER_GRID"
    echo "[validate] writing PDFs to $PLOTS_DIR"
    "$PY" "$PY_SCRIPTS/validate_event_full.py" \
        "$DAT_DIR/${BASENAME}_val.dat" \
        --mlp_dir "$MODELS_DIR" \
        --grid "$PLACEHOLDER_GRID" \
        -o "$PLOTS_DIR" \
        > "$REPORT_DIR/validate.log" 2>&1 || true
    echo "[validate] done. See $REPORT_DIR/validate.log"
}

do_all() {
    do_build_dat
    do_build_cuts
    do_train
    do_validate
}

# ─── Dispatch ────────────────────────────────────────────────────────────
case "$CMD" in
    build-dat)   do_build_dat ;;
    build-cuts)  do_build_cuts ;;
    train)       do_train ;;
    validate)    do_validate ;;
    all)         do_all ;;
    *)           echo "Unknown command: $CMD" >&2; exit 2 ;;
esac
