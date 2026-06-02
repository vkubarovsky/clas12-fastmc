# Fast MC scripts — ifarm2402

CLAS12 fast detector simulation for ep → e'p'φ, φ → K⁺K⁻.
Per-particle MLP acceptance + MDN resolution, trained on GEMC TruthMatch data.

## Directory layout

```
~/fastmc/scripts/                    ← this directory (code, persistent)
/volatile/clas12/vpk/fastmc/phi/     ← per-run output
    rga_fall2018_{inb|outb}/v10/
        hipo/                        symlink to HIPO files
        dat/                         .dat training data
        cuts/                        resolution JSON
        models/                      trained .pt files
        plots/                       validation PDFs
        report/                      chain + training logs
```

## Full chain (4 steps)

### Step 1 — HIPO → DAT

Converts GEMC HIPO files to compact .dat text files via TruthMatch
(MC::RecMatch quality > 0.98). Batched to survive occasional HIPO reader
crashes.

```
Scripts:  run_batches_tm.sh
          make_training_data_truthmatch.py
          make_training_data.py  (imported as mtd)
          matching_cuts.py       (imported by mtd)
```

Launch:
```bash
cd ~/fastmc/scripts
PY=/work/clas12/vpk/fast_MC/venv/bin/python

# Inbending (θ_e ≥ 6°)
nohup ./run_batches_tm.sh \
    /path/to/hipo_dir  /path/to/output_dat_dir  5  6.0 \
    > /tmp/batch_inb.log 2>&1 &

# Outbending (θ_e ≥ 4°)
nohup ./run_batches_tm.sh \
    /path/to/hipo_dir  /path/to/output_dat_dir  5  4.0 \
    > /tmp/batch_outb.log 2>&1 &
```

Outputs (4 merged .dat files per torus setting):
- `phi_tm_train.dat` / `phi_tm_val.dat` — all events → electron training
- `phi_tm_hadgated_train.dat` / `phi_tm_hadgated_val.dat` — electron-in-FD subset → hadron training

### Step 2 — Resolution JSON

Fits polynomial μ(p), σ(p) to residual distributions. Used as a sanity cut
during MDN training to reject reconstruction pathologies.

```
Script:  build_matching_cuts.py
```

### Step 3 — Train 4 MLPs

One MLP per particle (e⁻, p, K⁻, K⁺). Each has an acceptance classifier
and a 5-component Gaussian mixture smearing network.

- Electron trains on the full file (phi_tm_train.dat), 20σ sanity cut
- Hadrons train on the gated file (phi_tm_hadgated_train.dat), 10σ sanity cut

```
Script:  train_single_particle.py
```

Particle index order in .dat: 0 = e⁻, 1 = p, 2 = K⁻, 3 = K⁺

### Step 4 — Validate

Hierarchical FastMC sampling (electron first, hadrons conditional), compared
to GEMC truth. Produces a 13-page diagnostic PDF per model type.

```
Scripts:  validate_event_full.py
          validate_fast_mc_v2.py  (imported)
          fast_mc.py              (model architecture, imported)
          grid_fastmc.py          (grid model, imported)
          config.py               (paths, imported by fast_mc.py)
```

Key output: `validation_MLP.pdf` — page 13 has the summary table.
Target: all-4 ratio = 95–105%.

## Automated chain

`run_v10_chain.sh` runs steps 2–4 starting from existing .dat files:

```bash
cd ~/fastmc/scripts

# Single torus setting
nohup ./run_v10_chain.sh inb  > /dev/null 2>&1 &
nohup ./run_v10_chain.sh outb > /dev/null 2>&1 &
```

What it does:
1. Builds resolution JSON from val .dat
2. Trains 4 MLPs in parallel (OMP_NUM_THREADS=4 per process)
3. Runs validation, produces PDFs

Runtime: ~1 hour per torus setting.

## Monitor

```bash
./check_v10_both.sh
```

Shows: chain log, model count (N/4), training progress, smearing std values,
validation status. Green when both reach 4/4 models + validation PDF.

### Healthy std values (approximate)

```
  e-  dp≈0.05   dth≈0.15  dphi≈0.50  dvz≈0.50
   p  dp≈0.04   dth≈0.30  dphi≈0.60  dvz≈1.50
  K-  dp≈0.02   dth≈0.30  dphi≈0.65  dvz≈1.55
  K+  dp≈0.02   dth≈0.30  dphi≈0.65  dvz≈1.55
```

If dp_std ≥ 1 GeV → the sanity cut is missing or not working.

## Python environment

```
/work/clas12/vpk/fast_MC/venv/bin/python
```

Requires: numpy, matplotlib, torch, scikit-learn, scipy, hipopy (step 1 only).

## File dependency graph

```
run_batches_tm.sh
  └── make_training_data_truthmatch.py
        ├── make_training_data.py (as mtd)
        └── matching_cuts.py

build_matching_cuts.py  (standalone)

train_single_particle.py
  └── matching_cuts.py

validate_event_full.py
  ├── validate_fast_mc_v2.py
  │     └── fast_mc.py → config.py
  ├── fast_mc.py → config.py
  └── grid_fastmc.py
```

## Recovering the Python environment

The venv lives on `/work` which JLab can purge. A frozen package list is
saved in this directory for recovery.

### Save (do this once, already done)

```bash
/work/clas12/vpk/fast_MC/venv/bin/pip freeze > ~/fastmc/scripts/requirements.txt
```

### Restore (if /work is wiped)

```bash
# Use the system python3 to create a fresh venv
python3 -m venv /work/clas12/vpk/fast_MC/venv

# Install everything from the saved list (includes nvidia/CUDA libs
# bundled by PyTorch — large but required, CPU-only torch doesn't
# work on ifarm)
/work/clas12/vpk/fast_MC/venv/bin/pip install -r ~/fastmc/scripts/requirements.txt
```

If `requirements.txt` is also lost, install manually:

```bash
/work/clas12/vpk/fast_MC/venv/bin/pip install \
    numpy matplotlib torch scikit-learn scipy hipopy pypdf
```

This pulls the latest versions. The training code is not version-sensitive,
but for exact reproducibility keep `requirements.txt` current.
