# clas12-fastmc

Neural-network-based fast Monte Carlo for the CLAS12 detector.
Replaces the full GEMC (Geant4) + CLARA reconstruction chain with per-particle
acceptance and resolution models trained on simulated data.

Supported channels:

- **φ**: ep → e'p'φ, φ → K⁺K⁻ — scattered electron in the Forward Detector (`scripts_phi/`)
- **J/ψ**: ep → e'p'J/ψ, J/ψ → e⁺e⁻ — scattered electron in the **Forward Tagger**
  (θ < 7°, quasi-real photoproduction), decay leptons and proton FD-only (`scripts_jpsi/`)

## What it does

For each particle species (e⁻, p, K⁺, K⁻):

- **Acceptance model** — MLP that predicts whether a generated particle is reconstructed
- **Resolution model** — Mixture Density Network (MDN) that smears (p, θ, φ, vz) from generated to reconstructed

Input: generated 4-vectors from any event generator.
Output: reconstructed-like 4-vectors with realistic CLAS12 acceptance and resolution.

Speedup: ~10⁵× faster than GEMC + reconstruction.

## Pipeline

```
HIPO files ──→ .dat files ──→ resolution JSON ──→ train MLPs ──→ validate
   (C++)        (step 1)        (step 2)          (step 3)      (step 4)
```

### Single command (full chain)

```bash
./fastmc.sh all \
    --channel phi --period rga_fall2018 --polarity inb --version v11 \
    --hipo-dir /path/to/hipo/files \
    --beam-energy 10.6 --min-electron-theta 6.0
```

### Step by step

```bash
./fastmc.sh build-dat   --channel phi --period rga_fall2018 --polarity inb --version v11 \
                         --hipo-dir /path/to/hipo --beam-energy 10.6 --min-electron-theta 6.0
./fastmc.sh build-cuts  --channel phi --period rga_fall2018 --polarity inb --version v11
./fastmc.sh train       --channel phi --period rga_fall2018 --polarity inb --version v11
./fastmc.sh validate    --channel phi --period rga_fall2018 --polarity inb --version v11
./fastmc.sh clean       --channel phi --period rga_fall2018 --polarity inb --version v11
```

## Matching strategy

- **Hadrons (p, K⁺, K⁻):** hit-based TruthMatch via MC::RecMatch (quality > 0.98)
- **Electron:** reconstructed e⁻ (pid=11) in the forward detector, nearest in |p|

TruthMatch recovers ~25% of forward-detector tracks lost by kinematic windows (non-Gaussian
tails from energy loss and decay-in-flight), and keeps the ~95% of K⁺ that are
reconstructed as π⁺ by the Event Builder.

## Output directory layout

```
<root>/<channel>/<period>_<polarity>/<version>/
├── dat/          .dat training/validation data
├── cuts/         resolution JSON (polynomial fits)
├── models/       trained .pt files (e-.pt, p.pt, K-.pt, K+.pt)
├── plots/        validation PDFs
└── report/       logs, params.txt, launch.sh
```

## Directory structure

```
clas12-fastmc/
├── cpp/                              C++ HIPO→DAT converter
│   ├── make_training_data.cpp        reads MC::RecMatch, writes compact .dat
│   ├── CMakeLists.txt                builds against HIPO4
│   └── fastmc.sh                     pipeline wrapper (the main entry point)
├── scripts_phi/                      φ channel: Python training + validation
│   ├── train_single_particle.py      per-particle MLP + MDN training
│   ├── build_matching_cuts.py        fit resolution polynomials
│   ├── validate_event_full.py        13-page validation PDF
│   ├── fast_mc.py                    model architecture + inference
│   ├── grid_fastmc.py                grid-based acceptance model
│   ├── check_status.sh               monitor all running jobs
│   └── requirements.txt              Python dependencies
├── scripts_jpsi/                     J/ψ channel (FT electron), v1 derived from φ v11
│   ├── make_training_data_truthmatch.py  + pidonly_ft mode: e' matched in FT
│   │                                 (pid==11, status 1xxx), decay e+/e- in FD,
│   │                                 hadrons TruthMatch FD-only (no CD)
│   ├── run_batches_tm.sh             HIPO→DAT (jpsi_tm prefix, beam-energy arg)
│   ├── run_all_hipo2dat.sh           convert all 9 GEMC datasets in parallel
│   ├── run_v1_chain.sh               JSON → train 4 MLPs (e-, p, e+, e-d) → validate
│   ├── run_all_chains.sh             chains for all datasets, 3 at a time
│   ├── plot_gen_jpsi.py              4-page generated-kinematics plots (incl. t')
│   └── plot_training_data.py         5-page data plots (TruthMatch .dat format)
└── README.md
```

### J/ψ specifics (v1, June 2026)

- Scattered e′ is matched to reconstructed pid==11 in the **Forward Tagger**
  (|status| 1000–1999), nearest in |p|; generated θ(e′) < 7°, no minimum.
- The two e⁻ in the final state are separated by detector: index 0 = e′ (FT, model
  `e-.pt`), index 3 = decay (FD, model `e-d.pt`). Resolution JSON keys: `e-/FT`, `e-/FD`.
- The FT measures no vertex (vz_rec ≈ const) — Δvz is excluded from sanity cuts.
- Datasets: F18in/F18out/S18in/S18out (10.6 GeV) and S19in (10.2 GeV), GEMC by
  M. Tenorio, `/volatile/clas12/osg/marianat/{768-772,10796-10799}`.
- First validation (S18in_35nA): ALL-4 joint acceptance ratio FastMC/GEMC = 99.2%.

## Build (on JLab ifarm)

```bash
cd cpp
cmake -B build
cmake --build build -j
```

## Requirements

- HIPO4 library (for C++ builder)
- Python 3.10+ with: numpy, matplotlib, torch, scikit-learn, scipy
- hipopy (Python HIPO reader, only needed for legacy Python DAT generator)

## Supported run periods

| Period | Polarity | Beam (GeV) | θ_e cut |
|---|---|---|---|
| rga_fall2018 | inb, outb | 10.6 | 6°, 4° |
| rga_spring2018 | inb, outb | 10.6 | 6°, 4° |
| rga_spring2019 | inb | 10.2 | 6° |

## Authors

V. Kubarovsky (Jefferson Lab)
