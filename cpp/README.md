# cpp/ — C++ rewrite of make_training_data + pipeline wrapper

## Files

| File | What |
|---|---|
| `make_training_data.cpp` | C++ port of `make_training_data_truthmatch.py`. Direct hipo4 calls. |
| `CMakeLists.txt` | Build recipe — finds HIPO via `HIPO_DIR` or `$HIPO`. |
| `fastmc.sh` | Top-level wrapper: `build-dat / build-cuts / train / validate / all`. Knows the directory convention. |

## Build (on ifarm)

```bash
ssh vpk@ifarm2402

# scp this dir from vpkmacmini first (or rsync from OneDrive)
mkdir -p /home/vpk/2026_fastMC/cpp
# (transfer make_training_data.cpp, CMakeLists.txt, fastmc.sh)

cd /home/vpk/2026_fastMC/cpp

# HIPO should auto-find at the standard JLab path; otherwise pass explicitly:
# cmake -B build -DHIPO_DIR=/u/scigroup/cvmfs/.../hipo/4.2.1
cmake -B build
cmake --build build -j
ls -la build/make_training_data
```

Result: `/home/vpk/2026_fastMC/cpp/build/make_training_data`

## Quick smoke test (5 hipo files)

```bash
./build/make_training_data \
    /home/vpk/2026_fastMC/train_fast_mc/rga_fall2018_inb_hipo \
    /tmp/cpp_smoke \
    --reaction epK+K- \
    --beam_energy 10.6 --beam_pid 11 --target_pid 2212 \
    --min_electron_theta 6.0 \
    --max_files 5

ls -la /tmp/cpp_smoke
head -10 /tmp/cpp_smoke/phi_tm_train.dat
```

Compare to Python output side-by-side:
```bash
diff <(head -50 /tmp/cpp_smoke/phi_tm_train.dat) \
     <(head -50 /old/python/output/phi_tm_train.dat)
```

Numbers may differ at the last decimal due to floating-point ordering — that's fine. Structure and event counts should match.

## fastmc.sh — full chain wrapper

The wrapper knows the dir convention `<root>/<channel>/<period>_<polarity>/<version>/` and dispatches each sub-command:

```bash
# All-in-one
./fastmc.sh all \
    --channel phi --period rga_fall2018 --polarity inb --version v11 \
    --hipo-dir /volatile/clas12/singh/phi_sims/.../rgafall18/inb \
    --beam-energy 10.6 \
    --min-electron-theta 6.0

# Or step-by-step
./fastmc.sh build-dat   --channel phi --period rga_fall2018 --polarity inb \
                        --version v11 --hipo-dir /path/to/hipo \
                        --beam-energy 10.6 --min-electron-theta 6.0
./fastmc.sh build-cuts  --channel phi --period rga_fall2018 --polarity inb --version v11 --n-sigma 10
./fastmc.sh train       --channel phi --period rga_fall2018 --polarity inb --version v11
./fastmc.sh validate    --channel phi --period rga_fall2018 --polarity inb --version v11
```

For outbending RGA fall 2018, just swap `--polarity outb --min-electron-theta 4.0`.
For RGB / RGC / RGK periods, just change `--period` and `--beam-energy`.

Each call writes `params.json` at the version-dir top — preserves exactly what was used.

## Output dir layout

```
<root=/volatile/clas12/vpk/fastmc>/<channel>/<period>_<polarity>/<version>/
├── params.json                    # exact params used
├── dat/
│   ├── phi_tm_train.dat
│   ├── phi_tm_val.dat
│   ├── phi_tm_hadgated_train.dat
│   └── phi_tm_hadgated_val.dat
├── cuts/
│   └── matching_cuts.json
├── models/
│   ├── e-.pt
│   ├── p.pt
│   ├── K-.pt
│   └── K+.pt
├── plots/
│   ├── validation_MLP.pdf
│   └── validation_Grid.pdf
└── report/
    ├── build_dat.log
    ├── build_cuts.log
    ├── train_e.log
    ├── train_1.log
    ├── train_2.log
    ├── train_3.log
    └── validate.log
```

## Status

- **C++ port: first draft, single-threaded.** Reads MC::Particle, REC::Particle, MC::RecMatch directly; writes the same new-compact dat format as the Python version.
- **Not yet:** batching/cluster wrappers (`run_batches_tm.sh` equivalent), parallel HIPO reads, hadron-greedy charge-blind matching (only "truthmatch" mode supported in this draft — same as the current Python flow uses).
- **TODO for v2 of the C++ port:**
  - SLURM submit script (when we go to multi-period production)
  - Add `match_hadrons_greedy` mode for legacy compat
  - Optional dump of run-period diagnostics to `report/`
