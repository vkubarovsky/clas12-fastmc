#!/bin/bash
# Check training status for any version — scans ALL period/polarity combinations
# Usage:  ./check_status.sh v10
#         ./check_status.sh v11
PY=/work/clas12/vpk/fast_MC/venv/bin/python
VER=${1:?Usage: $0 <version, e.g. v10 or v11>}
ROOT=/volatile/clas12/vpk/fastmc/phi

# Find all directories that have this version
DIRS=$(find "$ROOT" -maxdepth 2 -type d -name "$VER" 2>/dev/null | sort)
if [ -z "$DIRS" ]; then
    echo "No $VER directories found under $ROOT"
    exit 1
fi

for VDIR in $DIRS; do
    # Extract period_polarity from path
    LABEL=$(basename $(dirname "$VDIR"))
    echo "============================== $LABEL $VER =============================="

    echo "=== chain log ==="
    grep -E "start|done|FAIL|COMPLETE|WARNING|build-dat|build-cuts|train|validate" \
        "$VDIR/report/CHAIN.log" "$VDIR/report/build_dat.log" 2>/dev/null | tail -5 || echo "  (none)"

    echo "=== dat files ==="
    ls -lh "$VDIR/dat/"phi_tm_train.dat "$VDIR/dat/"phi_tm_hadgated_train.dat 2>/dev/null || echo "  (none)"

    echo "=== models ==="
    HAVE=0
    for n in e- p K- K+; do
        if [ -f "$VDIR/models/${n}.pt" ]; then
            ls -lh "$VDIR/models/${n}.pt" | awk '{printf "  %-5s %s  %s %s %s\n", "'$n'.pt", $5, $6, $7, $8}'
            HAVE=$((HAVE+1))
        else
            printf "  %-5s ** MISSING **\n" "$n.pt"
        fi
    done
    echo "  [$HAVE/4]"

    echo "=== training progress ==="
    for logf in "$VDIR/report/train_e.log" "$VDIR/report/train_1.log" \
                "$VDIR/report/train_2.log" "$VDIR/report/train_3.log"; do
        [ -f "$logf" ] || continue
        printf "  %-12s " "$(basename $logf)"
        tail -1 "$logf" 2>/dev/null
    done

    if [ "$HAVE" -eq 4 ]; then
        echo "=== std ==="
        $PY -c "
import torch
for n in ['e-','p','K-','K+']:
    c = torch.load('$VDIR/models/' + n + '.pt', map_location='cpu', weights_only=False)
    s = c['y_smear_std'].tolist()
    print(f'  {n:>3s}  dp={s[0]:.4f}  dth={s[1]:.4f}  dphi={s[2]:.4f}  dvz={s[3]:.4f}')
" 2>/dev/null || echo "  (torch load failed)"
    fi

    echo "=== validation ==="
    [ -f "$VDIR/plots/validation_MLP.pdf" ] \
        && ls -lh "$VDIR/plots/validation_MLP.pdf" \
        || echo "  not yet"
    echo
done

echo "=== summary ==="
for VDIR in $DIRS; do
    LABEL=$(basename $(dirname "$VDIR"))
    N=$(ls "$VDIR/models/"*.pt 2>/dev/null | wc -l)
    V=$([ -f "$VDIR/plots/validation_MLP.pdf" ] && echo "YES" || echo "no")
    D=$([ -f "$VDIR/dat/phi_tm_train.dat" ] && echo "YES" || echo "no")
    printf "  %-30s  dat: %s  models: %d/4  validation: %s\n" "$LABEL" "$D" "$N" "$V"
done
