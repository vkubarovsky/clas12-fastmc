#!/bin/bash
# Check v10 inb + outb status on ifarm
PY=/work/clas12/vpk/fast_MC/venv/bin/python

for tag in inb outb; do
    V10=/volatile/clas12/vpk/fastmc/phi/rga_fall2018_${tag}/v10
    echo "============================== $tag =============================="

    echo "=== chain log ==="
    grep -E "start|done|FAIL|COMPLETE|WARNING" "$V10/report/CHAIN.log" 2>/dev/null || echo "  (none)"

    echo "=== models ==="
    HAVE=0
    for n in e- p K- K+; do
        if [ -f "$V10/models/${n}.pt" ]; then
            ls -lh "$V10/models/${n}.pt" | awk '{printf "  %-5s %s  %s %s %s\n", "'$n'.pt", $5, $6, $7, $8}'
            HAVE=$((HAVE+1))
        else
            printf "  %-5s ** MISSING **\n" "$n.pt"
        fi
    done
    echo "  [$HAVE/4]"

    echo "=== training progress ==="
    for logf in "$V10/report/train_e.log" "$V10/report/train_1.log" \
                "$V10/report/train_2.log" "$V10/report/train_3.log"; do
        [ -f "$logf" ] || continue
        printf "  %-12s " "$(basename $logf)"
        tail -1 "$logf" 2>/dev/null
    done

    if [ "$HAVE" -eq 4 ]; then
        echo "=== std ==="
        $PY -c "
import torch
for n in ['e-','p','K-','K+']:
    c = torch.load('$V10/models/' + n + '.pt', map_location='cpu', weights_only=False)
    s = c['y_smear_std'].tolist()
    print(f'  {n:>3s}  dp={s[0]:.4f}  dth={s[1]:.4f}  dphi={s[2]:.4f}  dvz={s[3]:.4f}')
" 2>/dev/null || echo "  (torch load failed)"
    fi

    echo "=== validation ==="
    [ -f "$V10/plots/validation_MLP.pdf" ] \
        && ls -lh "$V10/plots/validation_MLP.pdf" \
        || echo "  not yet"
    echo
done

echo "=== summary ==="
for tag in inb outb; do
    V10=/volatile/clas12/vpk/fastmc/phi/rga_fall2018_${tag}/v10
    N=$(ls "$V10/models/"*.pt 2>/dev/null | wc -l)
    V=$([ -f "$V10/plots/validation_MLP.pdf" ] && echo "YES" || echo "no")
    printf "  %-5s  models %d/4   validation: %s\n" "$tag" "$N" "$V"
done
