#!/bin/bash
# Create v11 directory structure and link HIPO files
set -euo pipefail

INB_HIPO=/volatile/clas12/singh/phi_sims/acceptance_large/rgafall18/inb/fall2018_inb_55nA_Singh_10422
OUTB_HIPO=/volatile/clas12/singh/phi_sims/acceptance_large/rgafall18/outb/fall2018_outb_50nA_Singh_10423

for TAG in inb outb; do
    V11=/volatile/clas12/vpk/fastmc/phi/rga_fall2018_${TAG}/v11
    echo "=== Setting up $TAG ==="

    if [ "$TAG" == "inb" ]; then HDIR=$INB_HIPO; else HDIR=$OUTB_HIPO; fi

    # Check HIPO source exists
    [ -d "$HDIR" ] || { echo "ERROR: $HDIR not found"; exit 1; }

    mkdir -p "$V11"/{dat,cuts,models,plots,report}
    rm -rf "$V11/hipo"
    ln -sfn "$HDIR" "$V11/hipo"

    NHIPO=$(ls "$V11/hipo/"*.hipo 2>/dev/null | wc -l)
    echo "  hipo -> $HDIR"
    echo "  $NHIPO HIPO files"
    echo ""
done

echo "Done. Now launch:"
echo "  nohup ./run_v11_full_chain.sh inb  > /dev/null 2>&1 &"
echo "  nohup ./run_v11_full_chain.sh outb > /dev/null 2>&1 &"
