#!/usr/bin/env python3
"""Read GEMC HIPO files and write training/validation .dat files, using the
COATJAVA hit-based TruthMatch bank (MC::RecMatch) instead of kinematic windows.

This is the v8 generator.  It is a drop-in replacement for
make_training_data.py: the output .dat format and column semantics are
identical, so build_grid_efficiency.py / train_single_particle.py /
plot_training_data.py read it unchanged.  What differs:

  * Matching is hit-based: each MC particle is linked to the REC particle that
    MC::RecMatch flags for it, with quality > QMIN (default 0.98).  No kinematic
    windows, no charge-sign logic, no PID requirement.  Species comes from the
    MC particle (MC::Particle.pid).
  * Event-level cut: drop the event if the GENERATED electron theta < THETA_MIN
    (default 6 deg).  Inbending forward electrons are not cleanly reconstructed
    and dropping them shrinks the file substantially.
  * Two output file pairs are written:
        <out>_train.dat / <out>_val.dat            -> ALL surviving events
                                                       (electron training set)
        <out>_hadgated_train.dat / _val.dat        -> subset where the electron
                                                       is truth-matched in the FD
                                                       (p / K+ / K- training set,
                                                        the v6 electron gating)

Status word in each particle line (unchanged):
    0 = not detected
    1 = matched, but reconstructed PID != true PID
    2 = matched, reconstructed PID == true PID

Usage (test on one file):
  python make_training_data_truthmatch.py /path/to/hipo_dir \\
      --beam_id electron --beam_energy 10.6 --target_id proton \\
      -o phi_tm --max_files 1 --max_events 20000

Full run (background, all files):
  nohup python make_training_data_truthmatch.py /path/to/hipo_dir \\
      --beam_id electron --beam_energy 10.6 --target_id proton \\
      -o phi_tm > make_tm.log 2>&1 &
"""

import argparse
import glob
import os
import sys
import time
import numpy as np
from hipopy.hipopy import hipochain

# reuse the proven helpers / tables from the original generator
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import make_training_data as mtd


def kinematics_from_xyz(px, py, pz):
    p = float(np.sqrt(px * px + py * py + pz * pz))
    if p <= 0:
        return 0.0, 0.0, 0.0
    th = float(np.degrees(np.arccos(pz / p)))
    ph = float(np.degrees(np.arctan2(py, px)))
    return p, th, ph


def det_from_status(status):
    """0=none handled by caller; 1=FD, 2=CD, 3=FT."""
    s = abs(int(status))
    if s >= 4000:
        return 2          # CD
    if s >= 2000:
        return 1          # FD
    if s >= 1000:
        return 3          # FT
    return 1              # default to FD for low/odd status


def build_parser():
    p = argparse.ArgumentParser(
        description="Truth-match (MC::RecMatch) training-data generator (v8).")
    p.add_argument("hipo_dir", help="Directory containing *.hipo files")
    p.add_argument("--beam_id", required=True)
    p.add_argument("--beam_energy", required=True, type=float)
    p.add_argument("--target_id", required=True)
    p.add_argument("-o", "--output", required=True,
                   help="Output base name (produces _train.dat/_val.dat and "
                        "_hadgated_train.dat/_val.dat)")
    p.add_argument("--output_dir", default=".")
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--max_events", type=int, default=0)
    p.add_argument("--max_files", type=int, default=0)
    p.add_argument("--file_offset", type=int, default=0)
    p.add_argument("--quality", type=float, default=0.98,
                   help="Minimum MC::RecMatch quality for hadrons (default 0.98)")
    p.add_argument("--electron_quality", type=float, default=None,
                   help="Minimum MC::RecMatch quality for the electron when "
                        "--electron_match truth (default: same as --quality)")
    p.add_argument("--electron_match", choices=["pidonly", "exactpid", "truth"],
                   default="pidonly",
                   help="How to match the electron: 'pidonly' = reconstructed "
                        "e- (pid==11) in FD, nearest in p, NO kinematic window "
                        "(recommended, PCAL/HTCC PID reliable, max statistics); "
                        "'exactpid' = exact-PID + kinematic window (legacy); "
                        "'truth' = MC::RecMatch like the hadrons")
    p.add_argument("--matching_cuts", default=None,
                   help="JSON matching-cuts file (from build_matching_cuts.py) "
                        "used for the electron exact-PID window")
    p.add_argument("--min_electron_theta", type=float, default=6.0,
                   help="Drop events with generated electron theta below this (deg)")
    p.add_argument("--gated_suffix", default="hadgated",
                   help="Suffix for the electron-FD-gated hadron file")
    return p


def main():
    parser = build_parser()
    if len(sys.argv) == 1:
        parser.print_help(); sys.exit(0)
    args = parser.parse_args()

    beam_id = mtd.parse_particle_id(args.beam_id)
    target_id = mtd.parse_particle_id(args.target_id)
    beam_energy = args.beam_energy
    electron_quality = args.electron_quality if args.electron_quality is not None else args.quality
    beam_name = mtd.PDG_TO_NAME.get(beam_id, str(beam_id))
    target_name = mtd.PDG_TO_NAME.get(target_id, str(target_id))

    if not os.path.isdir(args.hipo_dir):
        sys.exit(f"Error: '{args.hipo_dir}' is not a directory")
    hipo_files = sorted(glob.glob(os.path.join(args.hipo_dir, "*.hipo")))
    if not hipo_files:
        sys.exit(f"Error: no *.hipo files in {args.hipo_dir}")
    n_total_files = len(hipo_files)
    hipo_files = hipo_files[args.file_offset:]
    if args.max_files > 0:
        hipo_files = hipo_files[:args.max_files]
    print(f"Found {n_total_files} HIPO files; processing {len(hipo_files)}")

    reaction = mtd.detect_reaction(hipo_files)
    labels = mtd.FINAL_STATE_LABELS[reaction]
    mass_label = mtd.MESON_MASS_LABEL[reaction]
    npart = len(labels)
    expected_key = tuple(sorted(mtd.EXPECTED_PIDS[reaction]))
    print(f"Reaction: {reaction} ({', '.join(labels)})")
    print(f"Beam {beam_name}({beam_id}) {beam_energy} GeV, target {target_name}({target_id})")
    # electron matching cuts (for exactpid mode)
    mc_cuts = None
    if args.matching_cuts:
        mc_cuts = mtd.MatchingCuts(args.matching_cuts)
        print(f"Electron exact-PID window from {args.matching_cuts} "
              f"({mc_cuts.n_sigma:.0f} sigma)")
    print(f"Hadron match: TruthMatch quality > {args.quality}")
    if args.electron_match == "pidonly":
        print("Electron match: reconstructed e- (pid==11) in FD, nearest in p, no window")
    elif args.electron_match == "exactpid":
        print(f"Electron match: exact-PID + kinematic window"
              f"{'' if mc_cuts else ' (flat MATCH_WINDOWS legacy)'}")
    else:
        print(f"Electron match: TruthMatch quality > {electron_quality}")
    print(f"Drop gen electron theta < {args.min_electron_theta} deg")

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    out_base = os.path.join(out_dir, args.output)

    header_lines = [
        f"#! reaction: {reaction}",
        f"#! beam: {beam_name} ({beam_id})",
        f"#! beam_energy: {beam_energy}",
        f"#! target: {target_name} ({target_id})",
        f"#! source: {os.path.abspath(args.hipo_dir)}",
        f"#! columns_event: event_num nrec Q2 xB W t {mass_label}",
        f"#! matching: hadrons hit-based TruthMatch via MC::RecMatch quality>{args.quality}; "
        f"electron {'PID==11 in FD nearest in p' if args.electron_match=='pidonly' else 'exact-PID+kinematic window' if args.electron_match=='exactpid' else 'TruthMatch quality>'+str(electron_quality)}; "
        f"species from MC::Particle.pid; gen electron theta>={args.min_electron_theta} deg",
        f"#! columns_particle: status(0=not_detected,1=matched_no_PID,2=matched_with_PID) "
        f"pid det(0=none,1=FD,2=CD,3=FT) p_gen theta_gen phi_gen vz_gen "
        f"[p_rec theta_rec phi_rec vz_rec]  "
        f"# rec columns OMITTED when status==0",
    ]

    banks = ["MC::Particle", "REC::Particle", "MC::RecMatch"]

    # open four output files
    f_train = open(f"{out_base}_train.dat", 'w')
    f_val = open(f"{out_base}_val.dat", 'w')
    f_gtrain = open(f"{out_base}_{args.gated_suffix}_train.dat", 'w')
    f_gval = open(f"{out_base}_{args.gated_suffix}_val.dat", 'w')
    for line in header_lines:
        for fh in (f_train, f_val, f_gtrain, f_gval):
            fh.write(line + '\n')

    rng = np.random.default_rng(seed=42)
    val_frac = args.val_fraction

    event_num = 0
    n_skipped_channel = 0
    n_skipped_theta = 0
    n_train = n_val = 0
    n_gtrain = n_gval = 0
    n_rec_per_particle = [0] * npart
    idx_pid = [None] * npart   # actual MC pid at each particle index (from first kept event)
    nrec_counts = [0] * (npart + 1)
    n_e_fd_gated = 0
    # PID-agreement tally (status 1 vs 2) per species, among matched
    n_matched = {}
    n_pid_agree = {}
    done = False
    t0 = time.time()

    try:
        for ifile, hfile in enumerate(hipo_files):
            if done:
                break
            print(f"  File {ifile+1}/{len(hipo_files)}: {os.path.basename(hfile)} "
                  f"[events so far: {event_num}]")
            chain = hipochain([hfile], banks=banks, step=5000, tags=[0])
            for batch in chain:
                mc_px = batch["MC::Particle_px"]; mc_py = batch["MC::Particle_py"]
                mc_pz = batch["MC::Particle_pz"]; mc_vz = batch["MC::Particle_vz"]
                mc_pid = batch["MC::Particle_pid"]
                r_px = batch["REC::Particle_px"]; r_py = batch["REC::Particle_py"]
                r_pz = batch["REC::Particle_pz"]; r_vz = batch["REC::Particle_vz"]
                r_pid = batch["REC::Particle_pid"]; r_st = batch["REC::Particle_status"]
                m_pind = batch["MC::RecMatch_pindex"]; m_mind = batch["MC::RecMatch_mcindex"]
                m_qual = batch["MC::RecMatch_quality"]

                n_ev = len(mc_px)
                for i in range(n_ev):
                    mc_pids_i = mc_pid[i]
                    if len(mc_pids_i) < npart:
                        continue
                    ev_key = tuple(sorted(int(mc_pids_i[k]) for k in range(npart)))
                    if ev_key != expected_key:
                        n_skipped_channel += 1
                        continue

                    # MC kinematics
                    mc_info = []
                    for k in range(npart):
                        p, th, ph = kinematics_from_xyz(mc_px[i][k], mc_py[i][k], mc_pz[i][k])
                        pid = int(mc_pids_i[k])
                        mc_info.append({'index': k, 'pid': pid, 'p': p, 'theta': th,
                                        'phi': ph, 'vz': float(mc_vz[i][k]),
                                        'mass': mtd.PDG_MASS.get(pid, 0.0)})

                    # gen electron theta cut (drop event)
                    e_theta = None
                    for mc in mc_info:
                        if abs(mc['pid']) == 11:
                            e_theta = mc['theta']
                            break
                    if e_theta is not None and e_theta < args.min_electron_theta:
                        n_skipped_theta += 1
                        continue

                    # truth map: mc_index -> (rec_index, best_quality) over ALL rows
                    # (species-dependent threshold applied later)
                    nmc = len(mc_pids_i)
                    nrec = len(r_pid[i])
                    truth = {}
                    pind_i = m_pind[i]; mind_i = m_mind[i]; qual_i = m_qual[i]
                    for r in range(len(pind_i)):
                        q = float(qual_i[r])
                        rec_idx = int(pind_i[r]); mc_idx = int(mind_i[r])
                        if not (0 <= mc_idx < nmc and 0 <= rec_idx < nrec):
                            continue
                        if (mc_idx not in truth) or (q > truth[mc_idx][1]):
                            truth[mc_idx] = (rec_idx, q)

                    # build particle records
                    particles = []
                    nrec_found = 0
                    for mc in mc_info:
                        k = mc['index']
                        is_electron = (abs(mc['pid']) == 11)
                        if is_electron and args.electron_match == "pidonly":
                            # reconstructed e- (exact pid) in FD, nearest in p, no window
                            best_j = -1; best_dp = 1e9
                            for j in range(nrec):
                                if int(r_pid[i][j]) != mc['pid']:
                                    continue
                                if det_from_status(r_st[i][j]) != 1:   # FD only
                                    continue
                                rpj, _, _ = kinematics_from_xyz(
                                    r_px[i][j], r_py[i][j], r_pz[i][j])
                                if abs(rpj - mc['p']) < best_dp:
                                    best_dp = abs(rpj - mc['p']); best_j = j
                            if best_j >= 0:
                                rp, rt, rph = kinematics_from_xyz(
                                    r_px[i][best_j], r_py[i][best_j], r_pz[i][best_j])
                                rvz = float(r_vz[i][best_j])
                                rec_pid = int(r_pid[i][best_j])
                                det = det_from_status(r_st[i][best_j])
                                found = True
                            else:
                                rp = rt = rph = rvz = -999.0
                                rec_pid = 0; det = 0; found = False
                        elif is_electron and args.electron_match == "exactpid":
                            # original exact-PID + kinematic window
                            found, rp, rt, rph, rvz, det, rec_pid = mtd.match_rec_particle(
                                mc['pid'], mc['p'], mc['theta'], mc['phi'],
                                r_pid[i], r_px[i], r_py[i], r_pz[i], r_vz[i], r_st[i],
                                mc_cuts=mc_cuts)
                        else:
                            qmin = electron_quality if is_electron else args.quality
                            if k in truth and truth[k][1] > qmin:
                                rec_idx = truth[k][0]
                                rp, rt, rph = kinematics_from_xyz(
                                    r_px[i][rec_idx], r_py[i][rec_idx], r_pz[i][rec_idx])
                                rvz = float(r_vz[i][rec_idx])
                                rec_pid = int(r_pid[i][rec_idx])
                                det = det_from_status(r_st[i][rec_idx])
                                found = True
                            else:
                                rp = rt = rph = rvz = -999.0
                                rec_pid = 0; det = 0; found = False

                        if not found:
                            status = 0
                        elif rec_pid == mc['pid']:
                            status = 2
                        else:
                            status = 1
                        if found:
                            nrec_found += 1
                            sp = mtd.PDG_TO_SHORT.get(mc['pid'], str(mc['pid']))
                            n_matched[sp] = n_matched.get(sp, 0) + 1
                            if status == 2:
                                n_pid_agree[sp] = n_pid_agree.get(sp, 0) + 1

                        particles.append({
                            'status': status, 'pid': mc['pid'],
                            'p_gen': mc['p'], 'theta_gen': mc['theta'],
                            'phi_gen': mc['phi'], 'vz_gen': mc['vz'],
                            'p_rec': rp, 'theta_rec': rt, 'phi_rec': rph, 'vz_rec': rvz,
                            'det': det, 'mass': mc['mass'],
                        })

                    event_num += 1
                    Q2, xB, W, t, Mh, y, nu = mtd.compute_kinematics(
                        beam_energy, beam_id, target_id,
                        [(p['pid'], p['p_gen'], p['theta_gen'], p['phi_gen'], p['mass'])
                         for p in particles])

                    event_line = (f"{event_num}  {nrec_found}  {Q2:.4f}  {xB:.4f}  "
                                  f"{W:.4f}  {t:.4f}  {Mh:.4f}\n")
                    plines = []
                    for p in particles:
                        if p['status'] > 0:
                            # detected: status pid det p_gen θ_gen φ_gen vz_gen p_rec θ_rec φ_rec vz_rec
                            plines.append(
                                f" {p['status']}  {p['pid']:>5d}  {p['det']:1d}  "
                                f"{p['p_gen']:8.4f}  {p['theta_gen']:8.3f}  {p['phi_gen']:8.3f}  {p['vz_gen']:7.3f}  "
                                f"{p['p_rec']:8.4f}  {p['theta_rec']:8.3f}  {p['phi_rec']:8.3f}  {p['vz_rec']:7.3f}\n")
                        else:
                            # not detected: status pid det p_gen θ_gen φ_gen vz_gen   (rec columns omitted)
                            plines.append(
                                f" {p['status']}  {p['pid']:>5d}  {p['det']:1d}  "
                                f"{p['p_gen']:8.4f}  {p['theta_gen']:8.3f}  {p['phi_gen']:8.3f}  {p['vz_gen']:7.3f}\n")

                    # electron-FD gating flag
                    e_gated = False
                    for p in particles:
                        if abs(p['pid']) == 11 and p['status'] > 0 and p['det'] == 1:
                            e_gated = True
                            break
                    if e_gated:
                        n_e_fd_gated += 1

                    is_val = (rng.random() < val_frac)
                    if is_val:
                        f_val.write(event_line); [f_val.write(pl) for pl in plines]; n_val += 1
                        if e_gated:
                            f_gval.write(event_line); [f_gval.write(pl) for pl in plines]; n_gval += 1
                    else:
                        f_train.write(event_line); [f_train.write(pl) for pl in plines]; n_train += 1
                        if e_gated:
                            f_gtrain.write(event_line); [f_gtrain.write(pl) for pl in plines]; n_gtrain += 1

                    nrec_counts[nrec_found] += 1
                    for k in range(npart):
                        if idx_pid[k] is None:
                            idx_pid[k] = particles[k]['pid']
                        n_rec_per_particle[k] += (1 if particles[k]['status'] > 0 else 0)

                    if args.max_events > 0 and event_num >= args.max_events:
                        done = True; break
                if done:
                    break
    finally:
        for fh in (f_train, f_val, f_gtrain, f_gval):
            fh.close()

    dt = time.time() - t0
    n_total = event_num
    print(f"\nProcessing time: {dt:.1f} s   ({n_total/max(dt,1e-9):.0f} ev/s)")
    print(f"Skipped wrong channel: {n_skipped_channel}")
    print(f"Skipped gen e- theta < {args.min_electron_theta}: {n_skipped_theta}")
    print(f"Kept events: {n_total}")
    print(f"Fully reconstructed: {nrec_counts[npart]} "
          f"({100*nrec_counts[npart]/max(n_total,1):.1f}%)")
    for k in range(npart):
        sp = mtd.PDG_TO_SHORT.get(idx_pid[k], str(idx_pid[k])) if idx_pid[k] is not None else labels[k]
        print(f"  index {k} [{sp:>4s}] matched: {n_rec_per_particle[k]} "
              f"({100*n_rec_per_particle[k]/max(n_total,1):.1f}%)")
    print(f"Electron truth-matched in FD (gating): {n_e_fd_gated} "
          f"({100*n_e_fd_gated/max(n_total,1):.1f}%)")
    print("\nPID agreement among matched (status 2 / matched):")
    for sp in sorted(n_matched):
        m = n_matched[sp]; a = n_pid_agree.get(sp, 0)
        print(f"  {sp:>4s}: {a}/{m}  ({100*a/max(m,1):.1f}% rec PID == true)")
    print(f"\nElectron file:  {out_base}_train.dat / _val.dat   "
          f"({n_train} / {n_val} events)")
    print(f"Hadron (e-FD-gated) file:  {out_base}_{args.gated_suffix}_train.dat / _val.dat   "
          f"({n_gtrain} / {n_gval} events)")
    print("\nDone.")


if __name__ == "__main__":
    main()
