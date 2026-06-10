# J/ψ Fast MC scripts — ifarm2402

CLAS12 fast detector simulation for **ep → e'p'J/ψ, J/ψ → e⁺e⁻**.
Per-particle MLP acceptance + MDN resolution, trained on GEMC TruthMatch data.

Derived from the φ chain (`../scripts_phi/`, v11) — version numbering restarts
at **v1** for J/ψ; the two chains evolve independently.

## J/ψ-specific conventions

- **Scattered electron in the Forward Tagger.** Matched to a reconstructed
  pid==11 track with |REC::Particle.status| in 1000–1999, nearest in |p|,
  no kinematic window (`--electron_match pidonly_ft`).
  Generated θ(e′) < 7°, no minimum (FT acceptance ≈ 2.5–4.5°).
- **Everything else FD-only, no CD.** Decay e⁺/e⁻: pid==±11 in the FD,
  nearest in |p|. Proton: hit-based TruthMatch (MC::RecMatch quality > 0.98);
  a match outside the FD counts as *not detected*.
- **Two e⁻ per event**, separated by detector:

  | index | particle | detector | model file | cuts-JSON key |
  |---|---|---|---|---|
  | 0 | e′ scattered | FT | `e-.pt`  | `e-`/`FT` |
  | 1 | p            | FD | `p.pt`   | `p`/`FD`  |
  | 2 | e⁺ decay     | FD | `e+.pt`  | `e+`/`FD` |
  | 3 | e⁻ decay     | FD | `e-d.pt` | `e-`/`FD` |

- The "hadron-gated" .dat files are gated on *scattered e′ found in the FT*.
- The FT measures no vertex (vz_rec ≈ −3 cm constant) — Δvz is not used in
  the resolution sanity cuts.

## Directory layout

```
~/fastmc/scripts_jpsi/                 ← this directory (code, persistent)
/volatile/clas12/vpk/fastmc/jpsi/<dataset>/v1/
    hipo/hipo_dir                      symlink to GEMC HIPO files
    dat/                               jpsi_tm_*.dat training data
    cuts/                              resolution JSON (FD + FT)
    models/                            e-.pt  p.pt  e+.pt  e-d.pt
    plots/                             validation + kinematics PDFs
    report/                            chain + training logs
```

## Datasets (GEMC by M. Tenorio, /volatile/clas12/osg/marianat/)

| Dataset | OSG dir | Beam (GeV) |
|---|---|---|
| F18in_45nA  | 769   | 10.6 |
| F18in_50nA  | 770   | 10.6 |
| F18in_55nA  | 768   | 10.6 |
| F18out_40nA | 771   | 10.6 |
| F18out_50nA | 772   | 10.6 |
| S18in_35nA  | 10797 | 10.6 |
| S18in_50nA  | 10798 | 10.6 |
| S18out_45nA | 10799 | 10.6 |
| S19in_50nA  | 10796 | **10.2** |

## Full chain

### Step 1 — HIPO → DAT (all datasets, parallel)

```bash
cd ~/fastmc/scripts_jpsi
./run_all_hipo2dat.sh                  # ~40 min for all 9
```

Single dataset (beam energy is the 4th argument, default 10.6):

```bash
./run_batches_tm.sh /volatile/clas12/vpk/fastmc/jpsi/S19in_50nA/v1/hipo/hipo_dir \
                    /volatile/clas12/vpk/fastmc/jpsi/S19in_50nA/v1/dat  5  10.2
```

Outputs per dataset: `jpsi_tm_{train,val}.dat` (all events → e′ training) and
`jpsi_tm_hadgated_{train,val}.dat` (e′-in-FT subset → p/e⁺/e⁻d training).

### Steps 2–4 — resolution JSON → train 4 MLPs → validate

```bash
./run_v1_chain.sh F18in_45nA           # one dataset (~15–60 min)
./run_all_chains.sh                    # all 9, three at a time
```

Key output: `plots/validation_MLP.pdf` — page 13 has the summary table.
Target: ALL-4 joint ratio FastMC/GEMC = 95–105%.
First result (S18in_35nA): **99.2%**.

## Monitoring

```bash
tail -f /volatile/clas12/vpk/fastmc/jpsi/*/v1/report/hipo2dat.log   # step 1
tail -f /volatile/clas12/vpk/fastmc/jpsi/<DS>/v1/report/CHAIN.log   # steps 2–4
```

## Plots

```bash
# 4 pages of generated kinematics: Q², xB, W, t, t′=|t−t_min|, M(e+e-);
# e′(FT), decay leptons, proton — E, θ, θ vs E
python plot_gen_jpsi.py <dat_file> -o gen.pdf

# 5 pages incl. rec−gen resolutions (TruthMatch .dat format)
python plot_training_data.py <dat_file> -o plots.pdf
```

## Python environment

```
source /work/clas12/vpk/fast_MC/venv/bin/activate
```

Restore after a /work purge: `pip install -r requirements.txt`
(needs numpy, matplotlib, torch, scikit-learn, scipy, hipopy).

## File dependency graph

```
run_all_hipo2dat.sh
  └── run_batches_tm.sh
        └── make_training_data_truthmatch.py   (pidonly_ft mode)
              ├── make_training_data.py  (as mtd; EXPECTED_PIDS epe+e- fixed)
              └── matching_cuts.py

run_all_chains.sh
  └── run_v1_chain.sh
        ├── build_matching_cuts.py   (FD + FT, no theta boundary)
        ├── train_single_particle.py (--model_name e-/p/e+/e-d, FT sanity cut)
        └── validate_event_full.py   (unique names, M(e+e-)/MM pages)
              ├── validate_fast_mc_v2.py → fast_mc.py → config.py
              └── grid_fastmc.py
```

## Known limitations (v1)

- The M(e⁺e⁻) radiative tail below ~2.95 GeV is under-populated by the
  5-Gaussian MDN (extreme bremsstrahlung tail of the decay leptons).
- The generator has a small empty gap at Q² ≈ 0.146–0.148 GeV² (harmless:
  the analysis uses Q² < ~0.12 GeV²).
