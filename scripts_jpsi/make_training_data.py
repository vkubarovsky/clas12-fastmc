#!/usr/bin/env python3
"""Read GEMC HIPO files, match reconstructed to generated particles,
write training/validation text files and kinematic plots.

Usage:
  python make_training_data.py /path/to/hipo_dir \\
      --beam_id electron --beam_energy 10.6 --target_id proton -o output_base

  python make_training_data.py -h
"""

import argparse
import glob
import os
import sys
import time
import numpy as np
from hipopy.hipopy import hipochain
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from matching_cuts import MatchingCuts
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ── Particle name ↔ PDG lookup ────────────────────────────────────
PARTICLE_NAMES = {
    'electron': 11,  'e': 11,   '11': 11,
    'muon': 13,      'mu': 13,  '13': 13,
    'proton': 2212,  'p': 2212, '2212': 2212,
    'neutron': 2112, 'n': 2112, '2112': 2112,
}

PDG_TO_NAME = {
    11: 'electron', -11: 'positron', 13: 'muon', -13: 'antimuon',
    2212: 'proton', 2112: 'neutron',
    211: 'pi+', -211: 'pi-',
    321: 'K+', -321: 'K-',
    22: 'gamma',
}

PDG_TO_SHORT = {
    11: 'e-', -11: 'e+', 13: 'mu-', -13: 'mu+',
    2212: 'p', 2112: 'n',
    211: 'pi+', -211: 'pi-',
    321: 'K+', -321: 'K-',
    22: 'gamma',
}

# ── Matching windows: (dp_max GeV, dtheta_max deg, dphi_max deg, dvz_max cm) ─
# Edit these values based on resolution studies.
MATCH_WINDOWS = {
    11:   (0.25, 0.50, 2.5, 4.0),   # electron  (3σ from 500k events)
    2212: (0.10, 2.60, 2.0, 1.5),   # proton
    211:  (0.15, 0.70, 3.5, 5.0),   # pi+  (same as K+ for now)
    -211: (0.15, 0.90, 3.0, 5.0),   # pi-  (same as K- for now)
    321:  (0.15, 0.70, 3.5, 5.0),   # K+
    -321: (0.15, 0.90, 3.0, 5.0),   # K-
    -11:  (0.25, 0.50, 2.5, 4.0),   # positron  (same as electron)
}

# ── Reaction detection ────────────────────────────────────────────
KNOWN_REACTIONS = {
    (-321, 11, 321, 2212):   'epK+K-',
    (-211, 11, 211, 2212):   'eppi+pi-',
    (-11, 11, 2212):         'epe+e-',
}

# MC::Particle ordering: 0=scattered lepton, 1=recoil baryon, 2=h+, 3=h-
FINAL_STATE_LABELS = {
    'epK+K-':    ['e-', 'p', 'K+', 'K-'],
    'eppi+pi-':  ['e-', 'p', 'pi+', 'pi-'],
    'epe+e-':    ['e-', 'p', 'e+', 'e-'],
}

MESON_MASS_LABEL = {
    'epK+K-':    'M(K+K-)',
    'eppi+pi-':  'M(pi+pi-)',
    'epe+e-':    'M(e+e-)',
}

EXPECTED_PIDS = {
    'epK+K-':    [11, 2212, 321, -321],
    'eppi+pi-':  [11, 2212, 211, -211],
    'epe+e-':    [11, 2212, -11, 11],
}

MESON_NOMINAL_MASS = {
    'epK+K-':    1.019,   # phi
    'eppi+pi-':  0.775,   # rho
    'epe+e-':    3.097,   # J/psi
}

# Masses for known PIDs
PDG_MASS = {
    11: 0.000511, -11: 0.000511,
    13: 0.10566,  -13: 0.10566,
    2212: 0.938272, 2112: 0.939565,
    211: 0.13957,  -211: 0.13957,
    321: 0.49368,  -321: 0.49368,
    22: 0.0,
}


def parse_particle_id(s):
    """Accept PDG code (int) or name string."""
    s_lower = s.strip().lower()
    if s_lower in PARTICLE_NAMES:
        return PARTICLE_NAMES[s_lower]
    try:
        return int(s)
    except ValueError:
        print(f"Error: unknown particle '{s}'")
        print(f"  Accepted names: {', '.join(k for k in PARTICLE_NAMES if not k.isdigit())}")
        sys.exit(1)


def build_parser():
    p = argparse.ArgumentParser(
        description="Read GEMC HIPO files, match gen/rec, write training data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  CLAS12 phi:   %(prog)s ./rec/ --beam_id electron --beam_energy 10.6 --target_id proton -o phi
  CLAS12 rho:   %(prog)s ./rec/ --beam_id 11 --beam_energy 10.6 --target_id 2212 -o rho
  COMPASS:      %(prog)s ./rec/ --beam_id muon --beam_energy 160.0 --target_id proton -o compass

Accepted particle names:
  beam_id:   electron (e), muon (mu), or PDG code
  target_id: proton (p), neutron (n), or PDG code
""")
    p.add_argument("hipo_dir", help="Directory containing *.hipo files")
    p.add_argument("--beam_id", required=True,
                   help="Beam particle: electron (e), muon (mu), or PDG code")
    p.add_argument("--beam_energy", required=True, type=float,
                   help="Beam energy in GeV")
    p.add_argument("--target_id", required=True,
                   help="Target particle: proton (p), neutron (n), or PDG code")
    p.add_argument("-o", "--output", required=True,
                   help="Output base name (produces _train.dat and _val.dat)")
    p.add_argument("--output_dir", default=".",
                   help="Output directory (default: current directory)")
    p.add_argument("--val_fraction", type=float, default=0.2,
                   help="Validation fraction (default: 0.2)")
    p.add_argument("--max_events", type=int, default=0,
                   help="Maximum events to process (0 = all)")
    p.add_argument("--max_files", type=int, default=0,
                   help="Maximum HIPO files to process (0 = all)")
    p.add_argument("--file_offset", type=int, default=0,
                   help="Skip first N HIPO files")
    p.add_argument("--matching_cuts", default=None,
                   help="JSON file with momentum-dependent matching cuts (from build_matching_cuts.py)")
    return p


def detect_reaction(hipo_files, n_probe=100):
    """Read first events to determine the reaction from MC::Particle PIDs."""
    chain = hipochain(hipo_files[:1], banks=["MC::Particle"], step=n_probe, tags=[0])
    pid_sets = []
    for batch in chain:
        pid_list = batch["MC::Particle_pid"]
        for pids in pid_list:
            pid_sets.append(tuple(sorted(set(int(p) for p in pids))))
        break

    if not pid_sets:
        print("Error: no events found in HIPO files")
        sys.exit(1)

    most_common = max(set(pid_sets), key=pid_sets.count)
    key = tuple(sorted(most_common))
    if key in KNOWN_REACTIONS:
        return KNOWN_REACTIONS[key]

    print(f"Error: unknown reaction with PIDs {most_common}")
    print(f"  Known reactions: {list(KNOWN_REACTIONS.values())}")
    sys.exit(1)


def compute_kinematics(beam_energy, beam_pid, target_pid, particles):
    """Compute Q2, xB, W, t, Mh, y, nu from generated 4-vectors.
    particles: list of (pid, p, theta_deg, phi_deg, mass) for each final-state particle.
    Returns: Q2, xB, W, t, Mh, y, nu
    """
    beam_mass = 0.000511 if abs(beam_pid) == 11 else 0.10566
    target_mass = 0.938272 if target_pid == 2212 else 0.939565

    beam_E = beam_energy
    beam_pz = np.sqrt(beam_E**2 - beam_mass**2)

    e_p, e_theta, e_phi = particles[0][1], particles[0][2], particles[0][3]
    e_theta_r = np.radians(e_theta)
    e_phi_r = np.radians(e_phi)
    e_px = e_p * np.sin(e_theta_r) * np.cos(e_phi_r)
    e_py = e_p * np.sin(e_theta_r) * np.sin(e_phi_r)
    e_pz = e_p * np.cos(e_theta_r)
    e_E = np.sqrt(e_p**2 + beam_mass**2)

    nu = beam_E - e_E
    Q2 = 2.0 * beam_E * e_E * (1.0 - np.cos(e_theta_r))
    xB = Q2 / (2.0 * target_mass * nu) if nu > 0 else -999.0
    y = nu / beam_E if beam_E > 0 else -999.0
    W = np.sqrt(max(0, target_mass**2 + 2.0 * target_mass * nu - Q2))

    p_p, p_theta, p_phi = particles[1][1], particles[1][2], particles[1][3]
    p_theta_r = np.radians(p_theta)
    p_phi_r = np.radians(p_phi)
    p_px = p_p * np.sin(p_theta_r) * np.cos(p_phi_r)
    p_py = p_p * np.sin(p_theta_r) * np.sin(p_phi_r)
    p_pz = p_p * np.cos(p_theta_r)
    p_E = np.sqrt(p_p**2 + target_mass**2)

    dpx = p_px
    dpy = p_py
    dpz = p_pz
    dE = p_E - target_mass
    t = dE**2 - dpx**2 - dpy**2 - dpz**2

    hp_p, hp_theta, hp_phi, hp_mass = particles[2][1], particles[2][2], particles[2][3], particles[2][4]
    hm_p, hm_theta, hm_phi, hm_mass = particles[3][1], particles[3][2], particles[3][3], particles[3][4]

    hp_theta_r, hp_phi_r = np.radians(hp_theta), np.radians(hp_phi)
    hm_theta_r, hm_phi_r = np.radians(hm_theta), np.radians(hm_phi)

    hp_px = hp_p * np.sin(hp_theta_r) * np.cos(hp_phi_r)
    hp_py = hp_p * np.sin(hp_theta_r) * np.sin(hp_phi_r)
    hp_pz = hp_p * np.cos(hp_theta_r)
    hp_E = np.sqrt(hp_p**2 + hp_mass**2)

    hm_px = hm_p * np.sin(hm_theta_r) * np.cos(hm_phi_r)
    hm_py = hm_p * np.sin(hm_theta_r) * np.sin(hm_phi_r)
    hm_pz = hm_p * np.cos(hm_theta_r)
    hm_E = np.sqrt(hm_p**2 + hm_mass**2)

    Mh2 = (hp_E + hm_E)**2 - (hp_px + hm_px)**2 - (hp_py + hm_py)**2 - (hp_pz + hm_pz)**2
    Mh = np.sqrt(max(0, Mh2))

    return Q2, xB, W, t, Mh, y, nu


PDG_CHARGE = {
    11: -1, -11: +1,         # e-, e+
    13: -1, -13: +1,         # mu-, mu+
    211: +1, -211: -1,       # pi+, pi-
    321: +1, -321: -1,       # K+, K-
    2212: +1, -2212: -1,     # p, pbar
    22: 0, 111: 0, 2112: 0,  # gamma, pi0, n (neutrals — never match)
}


def charge_sign(pid):
    """Charge sign of a particle from its PDG code. 0 for neutrals/unknown."""
    return PDG_CHARGE.get(int(pid), 0)


def match_hadrons_greedy(mc_list, rec_pids, rec_px, rec_py, rec_pz, rec_vz, rec_status,
                         mc_cuts=None):
    """Greedy charge-blind matching of multiple MC hadrons to REC tracks.

    For each MC hadron, candidate REC particles must:
        - have the same CHARGE SIGN as the MC particle (PID-blind)
        - be within the kinematic window [μ - Nσ, μ + Nσ] (asymmetric)

    If mc_cuts (MatchingCuts) is provided, uses momentum-dependent windows.
    Otherwise falls back to flat MATCH_WINDOWS.

    Greedy assignment: sort all viable (MC, REC) pairs by normalized
    distance ascending; assign in order so each REC and each MC is used
    at most once.

    Returns:
        dict {mc_index: (found, rp, rtheta, rphi, rvz, det, rec_pid)}
        where keys are the MC particle indices passed in.
        Unmatched MC indices are absent from the dict.
    """
    # Pre-compute REC kinematics and detector type
    n_rec = len(rec_pids)
    rec_p = np.zeros(n_rec)
    rec_theta = np.zeros(n_rec)
    rec_phi = np.zeros(n_rec)
    rec_det = np.zeros(n_rec, dtype=int)
    for j in range(n_rec):
        p = np.sqrt(rec_px[j]**2 + rec_py[j]**2 + rec_pz[j]**2)
        rec_p[j] = p
        rec_theta[j] = np.degrees(np.arccos(rec_pz[j] / p)) if p > 0 else 0.0
        rec_phi[j] = np.degrees(np.arctan2(rec_py[j], rec_px[j]))
        abs_st = abs(int(rec_status[j]))
        rec_det[j] = 1 if abs_st < 4000 else 2   # 1=FD, 2=CD

    DET_LABEL = {1: "FD", 2: "CD"}

    # Build all viable pairs
    pairs = []   # (dist, k_mc_index, j_rec_index)
    for mc in mc_list:
        k = mc['index']
        mc_pid = int(mc['pid'])
        target_q = charge_sign(mc_pid)
        if target_q == 0:
            continue
        mc_name = PDG_TO_SHORT.get(mc_pid, f'pid{mc_pid}')

        for j in range(n_rec):
            if charge_sign(rec_pids[j]) != target_q:
                continue
            if abs(rec_status[j]) < 2000:
                continue

            dp = rec_p[j] - mc['p']
            dt = rec_theta[j] - mc['theta']
            dphi = rec_phi[j] - mc['phi']
            if dphi > 180.0:
                dphi -= 360.0
            elif dphi < -180.0:
                dphi += 360.0

            det_label = DET_LABEL.get(rec_det[j], "FD")

            if mc_cuts is not None:
                (dp_lo, dp_hi), (dt_lo, dt_hi), (dphi_lo, dphi_hi) = \
                    mc_cuts.window(mc_name, det_label, mc['p'], mc['theta'])
                if dp < dp_lo or dp > dp_hi:
                    continue
                if dt < dt_lo or dt > dt_hi:
                    continue
                if dphi < dphi_lo or dphi > dphi_hi:
                    continue
                dp_half = (dp_hi - dp_lo) / 2.0
                dt_half = (dt_hi - dt_lo) / 2.0
                dphi_half = (dphi_hi - dphi_lo) / 2.0
                dp_cen = dp - (dp_lo + dp_hi) / 2.0
                dt_cen = dt - (dt_lo + dt_hi) / 2.0
                dphi_cen = dphi - (dphi_lo + dphi_hi) / 2.0
                dist = np.sqrt((dp_cen/dp_half)**2 + (dt_cen/dt_half)**2 + (dphi_cen/dphi_half)**2)
            else:
                dp_max, dt_max, dphi_max, _ = MATCH_WINDOWS.get(mc_pid, (0.50, 4.0, 5.0, 5.0))
                if abs(dp) > dp_max or abs(dt) > dt_max or abs(dphi) > dphi_max:
                    continue
                dist = np.sqrt((dp/dp_max)**2 + (dt/dt_max)**2 + (dphi/dphi_max)**2)

            pairs.append((dist, k, j))

    if not pairs:
        return {}

    pairs.sort()   # by distance ascending
    used_mc = set()
    used_rec = set()
    matches = {}
    for dist, k, j in pairs:
        if k in used_mc or j in used_rec:
            continue
        rp = float(rec_p[j])
        rt = float(rec_theta[j])
        rph = float(rec_phi[j])
        rvz = float(rec_vz[j])
        det = int(rec_det[j])
        matches[k] = (True, rp, rt, rph, rvz, det, int(rec_pids[j]))
        used_mc.add(k)
        used_rec.add(j)
    return matches


def match_rec_particle(mc_pid, mc_p, mc_theta, mc_phi,
                       rec_pids, rec_px, rec_py, rec_pz, rec_vz, rec_status,
                       mc_cuts=None):
    """Find the reconstructed particle best matching the generated one (exact PID).
    Returns (found, rec_p, rec_theta, rec_phi, rec_vz, det, rec_pid) where det is
    1=FD (|status| 2000-3999), 2=CD (|status| >= 4000), 0=none.
    """
    mc_name = PDG_TO_SHORT.get(mc_pid, f'pid{mc_pid}')

    best_j = -1
    best_dist = 1e9

    for j in range(len(rec_pids)):
        if rec_pids[j] != mc_pid:
            continue
        if abs(rec_status[j]) < 2000:
            continue

        rp = np.sqrt(rec_px[j]**2 + rec_py[j]**2 + rec_pz[j]**2)
        rtheta = np.degrees(np.arccos(rec_pz[j] / rp)) if rp > 0 else 0
        rphi = np.degrees(np.arctan2(rec_py[j], rec_px[j]))

        dp = rp - mc_p
        dtheta = rtheta - mc_theta
        dphi = rphi - mc_phi
        if dphi > 180:
            dphi -= 360
        elif dphi < -180:
            dphi += 360

        abs_st = abs(int(rec_status[j]))
        det = 1 if abs_st < 4000 else 2
        det_label = "FD" if det == 1 else "CD"

        if mc_cuts is not None:
            (dp_lo, dp_hi), (dt_lo, dt_hi), (dphi_lo, dphi_hi) = \
                mc_cuts.window(mc_name, det_label, mc_p, mc_theta)
            if dp < dp_lo or dp > dp_hi:
                continue
            if dtheta < dt_lo or dtheta > dt_hi:
                continue
            if dphi < dphi_lo or dphi > dphi_hi:
                continue
            dp_half = (dp_hi - dp_lo) / 2.0
            dt_half = (dt_hi - dt_lo) / 2.0
            dphi_half = (dphi_hi - dphi_lo) / 2.0
            dist = np.sqrt(((dp-(dp_lo+dp_hi)/2)/dp_half)**2 +
                           ((dtheta-(dt_lo+dt_hi)/2)/dt_half)**2 +
                           ((dphi-(dphi_lo+dphi_hi)/2)/dphi_half)**2)
        else:
            dp_max, dtheta_max, dphi_max, _ = MATCH_WINDOWS.get(mc_pid, (0.50, 4.0, 5.0, 5.0))
            if abs(dp) > dp_max or abs(dtheta) > dtheta_max or abs(dphi) > dphi_max:
                continue
            dist = np.sqrt((dp/dp_max)**2 + (dtheta/dtheta_max)**2 + (dphi/dphi_max)**2)

        if dist < best_dist:
            best_dist = dist
            best_j = j

    if best_j < 0:
        return False, -999.0, -999.0, -999.0, -999.0, 0, 0

    rp = np.sqrt(rec_px[best_j]**2 + rec_py[best_j]**2 + rec_pz[best_j]**2)
    rtheta = np.degrees(np.arccos(rec_pz[best_j] / rp)) if rp > 0 else 0
    rphi = np.degrees(np.arctan2(rec_py[best_j], rec_px[best_j]))
    rvz = rec_vz[best_j]

    abs_status = abs(rec_status[best_j])
    det = 1 if abs_status < 4000 else 2

    return True, rp, rtheta, rphi, rvz, det, int(rec_pids[best_j])


def annotate(ax, data, loc="right"):
    """Add statistics box (N, Mean, Sigma) to an axes."""
    data = np.asarray(data)
    data = data[np.isfinite(data)]
    if len(data) == 0:
        return
    n, mean, sig = len(data), np.mean(data), np.std(data)
    txt = f"N     = {n}\nMean = {mean:.4f}\nSigma = {sig:.4f}"
    if loc == "right":
        x, ha = 0.97, "right"
    else:
        x, ha = 0.03, "left"
    ax.text(x, 0.95, txt, transform=ax.transAxes,
            ha=ha, va="top", fontsize=8, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow",
                      ec="gray", alpha=0.9))


def write_output(filename, header_lines, events):
    """Write events to text file."""
    with open(filename, 'w') as f:
        for line in header_lines:
            f.write(line + '\n')
        for ev in events:
            f.write(ev['event_line'] + '\n')
            for pline in ev['particle_lines']:
                f.write(pline + '\n')
    print(f"  Wrote {len(events)} events to {filename}")


def compute_missing_energy(beam_energy, beam_pid, target_pid, particles, use_rec=True):
    """Compute missing energy from 4-vectors.
    use_rec=True: from reconstructed (only if all particles reconstructed).
    use_rec=False: from generated (always valid).
    ME = (beam_E + target_M) - sum(E_i)
    """
    beam_mass = 0.000511 if abs(beam_pid) == 11 else 0.10566
    target_mass = 0.938272 if target_pid == 2212 else 0.939565

    total_E = 0.0
    for p in particles:
        mass = PDG_MASS.get(p['pid'], 0.0)
        if use_rec:
            if p['status'] == 0:
                return None
            total_E += np.sqrt(p['p_rec']**2 + mass**2)
        else:
            total_E += np.sqrt(p['p_gen']**2 + mass**2)

    return beam_energy + target_mass - total_E


def make_plots(events, reaction, beam_energy, beam_pid, target_pid, out_base):
    """Produce kinematic plots: gen distributions and gen-when-all-rec."""
    labels = FINAL_STATE_LABELS[reaction]
    mass_label = MESON_MASS_LABEL[reaction]
    npart = len(labels)

    gen_p = [[] for _ in range(npart)]
    gen_theta = [[] for _ in range(npart)]
    gen_phi = [[] for _ in range(npart)]
    gen_vz = [[] for _ in range(npart)]

    full_p = [[] for _ in range(npart)]
    full_theta = [[] for _ in range(npart)]
    full_phi = [[] for _ in range(npart)]
    full_vz = [[] for _ in range(npart)]

    Q2_all, xB_all, W_all, t_all, Mh_all = [], [], [], [], []
    y_all, nu_all = [], []
    Q2_full, xB_full, W_full, t_full, Mh_full = [], [], [], [], []
    y_full, nu_full = [], []
    y_rec_full, nu_rec_full, Mh_rec_full = [], [], []
    ME_gen_all = []
    ME_rec_full = []

    beam_mass = 0.000511 if abs(beam_pid) == 11 else 0.10566
    target_mass = 0.938272 if target_pid == 2212 else 0.939565

    for ev in events:
        is_full = (ev['nrec'] == npart)
        Q2_all.append(ev['Q2']); xB_all.append(ev['xB'])
        W_all.append(ev['W']); t_all.append(ev['t']); Mh_all.append(ev['Mh'])
        y_all.append(ev['y']); nu_all.append(ev['nu'])

        me_gen = compute_missing_energy(beam_energy, beam_pid, target_pid,
                                        ev['particles'], use_rec=False)
        if me_gen is not None:
            ME_gen_all.append(me_gen)

        if is_full:
            Q2_full.append(ev['Q2']); xB_full.append(ev['xB'])
            W_full.append(ev['W']); t_full.append(ev['t']); Mh_full.append(ev['Mh'])
            y_full.append(ev['y']); nu_full.append(ev['nu'])
            me_rec = compute_missing_energy(beam_energy, beam_pid, target_pid,
                                            ev['particles'], use_rec=True)
            if me_rec is not None:
                ME_rec_full.append(me_rec)

            # Reconstructed y, nu from rec electron
            ep = ev['particles'][0]
            if ep['status'] > 0:
                e_E_rec = np.sqrt(ep['p_rec']**2 + beam_mass**2)
                nu_rec = beam_energy - e_E_rec
                y_rec = nu_rec / beam_energy if beam_energy > 0 else -999
                nu_rec_full.append(nu_rec)
                y_rec_full.append(y_rec)

            # Reconstructed M(h+h-) from rec hadrons
            hp = ev['particles'][2]; hm = ev['particles'][3]
            if hp['status'] > 0 and hm['status'] > 0:
                hp_mass = PDG_MASS.get(hp['pid'], 0.0)
                hm_mass = PDG_MASS.get(hm['pid'], 0.0)
                hp_tr = np.radians(hp['theta_rec']); hp_pr = np.radians(hp['phi_rec'])
                hm_tr = np.radians(hm['theta_rec']); hm_pr = np.radians(hm['phi_rec'])
                hp_E = np.sqrt(hp['p_rec']**2 + hp_mass**2)
                hm_E = np.sqrt(hm['p_rec']**2 + hm_mass**2)
                hpx = hp['p_rec']*np.sin(hp_tr)*np.cos(hp_pr)
                hpy = hp['p_rec']*np.sin(hp_tr)*np.sin(hp_pr)
                hpz = hp['p_rec']*np.cos(hp_tr)
                hmx = hm['p_rec']*np.sin(hm_tr)*np.cos(hm_pr)
                hmy = hm['p_rec']*np.sin(hm_tr)*np.sin(hm_pr)
                hmz = hm['p_rec']*np.cos(hm_tr)
                Mh2_rec = (hp_E+hm_E)**2 - (hpx+hmx)**2 - (hpy+hmy)**2 - (hpz+hmz)**2
                Mh_rec_full.append(np.sqrt(max(0, Mh2_rec)))

        for k in range(npart):
            gen_p[k].append(ev['particles'][k]['p_gen'])
            gen_theta[k].append(ev['particles'][k]['theta_gen'])
            gen_phi[k].append(ev['particles'][k]['phi_gen'])
            gen_vz[k].append(ev['particles'][k]['vz_gen'])
            if is_full:
                full_p[k].append(ev['particles'][k]['p_gen'])
                full_theta[k].append(ev['particles'][k]['theta_gen'])
                full_phi[k].append(ev['particles'][k]['phi_gen'])
                full_vz[k].append(ev['particles'][k]['vz_gen'])

    GEN_KW  = dict(histtype="stepfilled", color="deepskyblue", edgecolor="navy",
                   alpha=0.7, label="Generated (all)")
    FULL_KW = dict(histtype="step", color="red", linewidth=1.5,
                   label="Generated (all rec)")

    # Stat box position per particle per variable
    # theta: top-left for proton (k=1), top-right for others
    def theta_loc(k):
        return "left" if k == 1 else "right"

    pdf_path = f"{out_base}_plots.pdf"
    with PdfPages(pdf_path) as pdf:
        # ── Page 1: per-particle p, theta, phi, vz ────────────────
        fig, axes = plt.subplots(npart, 4, figsize=(20, 4 * npart))
        if npart == 1:
            axes = axes[np.newaxis, :]
        for k in range(npart):
            gp = np.array(gen_p[k]); fp = np.array(full_p[k])
            gt = np.array(gen_theta[k]); ft = np.array(full_theta[k])
            gphi = np.array(gen_phi[k]); fphi = np.array(full_phi[k])
            gvz = np.array(gen_vz[k]); fvz = np.array(full_vz[k])

            axes[k, 0].hist(gp, bins=100, **GEN_KW)
            axes[k, 0].hist(fp, bins=100, **FULL_KW)
            axes[k, 0].set_xlabel("|p| (GeV)")
            axes[k, 0].set_ylabel(f"{labels[k]}")
            annotate(axes[k, 0], gp, loc="right")
            if k == 0:
                axes[k, 0].set_title("|p|")
                axes[k, 0].legend(fontsize=7)

            axes[k, 1].hist(gt, bins=100, **GEN_KW)
            axes[k, 1].hist(ft, bins=100, **FULL_KW)
            axes[k, 1].set_xlabel("θ (deg)")
            annotate(axes[k, 1], gt, loc=theta_loc(k))
            if k == 0:
                axes[k, 1].set_title("θ")

            axes[k, 2].hist(gphi, bins=100, **GEN_KW)
            axes[k, 2].hist(fphi, bins=100, **FULL_KW)
            axes[k, 2].set_xlabel("φ (deg)")
            annotate(axes[k, 2], gphi, loc="right")
            if k == 0:
                axes[k, 2].set_title("φ")

            axes[k, 3].hist(gvz, bins=100, **GEN_KW)
            axes[k, 3].hist(fvz, bins=100, **FULL_KW)
            axes[k, 3].set_xlabel("vz (cm)")
            annotate(axes[k, 3], gvz, loc="right")
            if k == 0:
                axes[k, 3].set_title("vz")

        fig.suptitle(f"{reaction}  —  Ebeam={beam_energy} GeV  —  "
                     f"{len(events)} events, {sum(1 for e in events if e['nrec']==npart)} fully reconstructed",
                     fontsize=13)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ── Page 2: event kinematics ──────────────────────────────
        fig2, axes2 = plt.subplots(2, 4, figsize=(22, 10))

        # Q2 — log scale
        va = np.array(Q2_all); vf = np.array(Q2_full)
        axes2[0, 0].hist(va, bins=100, **GEN_KW)
        axes2[0, 0].hist(vf, bins=100, **FULL_KW)
        axes2[0, 0].set_xlabel("Q² (GeV²)"); axes2[0, 0].set_title("Q²")
        axes2[0, 0].set_yscale("log")
        axes2[0, 0].legend(fontsize=7)
        annotate(axes2[0, 0], va, loc="right")

        # xB — log scale
        va = np.array(xB_all); vf = np.array(xB_full)
        axes2[0, 1].hist(va, bins=100, **GEN_KW)
        axes2[0, 1].hist(vf, bins=100, **FULL_KW)
        axes2[0, 1].set_xlabel("xB"); axes2[0, 1].set_title("xB")
        axes2[0, 1].set_yscale("log")
        annotate(axes2[0, 1], va, loc="right")

        # -t — log scale (swapped with W)
        va = -np.array(t_all); vf = -np.array(t_full)
        axes2[0, 2].hist(va, bins=100, **GEN_KW)
        axes2[0, 2].hist(vf, bins=100, **FULL_KW)
        axes2[0, 2].set_xlabel("-t (GeV²)"); axes2[0, 2].set_title("-t")
        axes2[0, 2].set_yscale("log")
        annotate(axes2[0, 2], va, loc="right")

        # W (swapped with -t)
        va = np.array(W_all); vf = np.array(W_full)
        axes2[0, 3].hist(va, bins=100, **GEN_KW)
        axes2[0, 3].hist(vf, bins=100, **FULL_KW)
        axes2[0, 3].set_xlabel("W (GeV)"); axes2[0, 3].set_title("W")
        annotate(axes2[0, 3], va, loc="left")

        # M(h+h-) — generated and reconstructed, narrow range around nominal mass
        REC_KW = dict(histtype="step", color="green", linewidth=1.5, label="Reconstructed")
        m_nom = MESON_NOMINAL_MASS[reaction]
        m_range = (m_nom - 0.04, m_nom + 0.04)
        va = np.array(Mh_all); vf = np.array(Mh_full)
        axes2[1, 0].hist(va, bins=100, range=m_range, **GEN_KW)
        axes2[1, 0].hist(vf, bins=100, range=m_range, **FULL_KW)
        if Mh_rec_full:
            axes2[1, 0].hist(np.array(Mh_rec_full), bins=100, range=m_range, **REC_KW)
        axes2[1, 0].set_xlabel(f"{mass_label} (GeV)"); axes2[1, 0].set_title(mass_label)
        axes2[1, 0].set_xlim(*m_range)
        axes2[1, 0].legend(fontsize=6)
        annotate(axes2[1, 0], va, loc="right")

        # y — generated and reconstructed
        va = np.array(y_all); vf = np.array(y_full)
        axes2[1, 1].hist(va, bins=100, range=(0, 1), **GEN_KW)
        axes2[1, 1].hist(vf, bins=100, range=(0, 1), **FULL_KW)
        if y_rec_full:
            axes2[1, 1].hist(np.array(y_rec_full), bins=100, range=(0, 1), **REC_KW)
        axes2[1, 1].set_xlabel("y"); axes2[1, 1].set_title("y")
        axes2[1, 1].set_xlim(0, 1)
        annotate(axes2[1, 1], va, loc="right")

        # nu — generated and reconstructed
        va = np.array(nu_all); vf = np.array(nu_full)
        axes2[1, 2].hist(va, bins=100, range=(0, 11), **GEN_KW)
        axes2[1, 2].hist(vf, bins=100, range=(0, 11), **FULL_KW)
        if nu_rec_full:
            axes2[1, 2].hist(np.array(nu_rec_full), bins=100, range=(0, 11), **REC_KW)
        axes2[1, 2].set_xlabel("ν (GeV)"); axes2[1, 2].set_title("ν")
        axes2[1, 2].set_xlim(0, 11)
        annotate(axes2[1, 2], va, loc="right")

        # Missing energy — generated (all events) and reconstructed (fully rec only)
        me_gen = np.array(ME_gen_all) if ME_gen_all else np.array([])
        me_rec = np.array(ME_rec_full) if ME_rec_full else np.array([])
        if len(me_gen) > 0:
            axes2[1, 3].hist(me_gen, bins=100, **GEN_KW)
        if len(me_rec) > 0:
            axes2[1, 3].hist(me_rec, bins=100, **REC_KW)
        axes2[1, 3].set_xlabel("Missing Energy (GeV)")
        axes2[1, 3].set_title("Missing Energy")
        if len(me_gen) > 0 or len(me_rec) > 0:
            axes2[1, 3].legend(fontsize=7)
            annotate(axes2[1, 3], me_gen if len(me_gen) > 0 else me_rec, loc="right")
        else:
            axes2[1, 3].axis('off')
            axes2[1, 3].text(0.5, 0.5, "No events",
                             transform=axes2[1, 3].transAxes,
                             ha="center", va="center", fontsize=12)

        for ax in axes2.flat:
            ax.set_ylabel("Counts")

        n_total = len(events)
        n_full = sum(1 for e in events if e['nrec'] == npart)
        fig2.suptitle(f"{reaction}  —  Ebeam={beam_energy} GeV  —  "
                      f"{n_total} events, {n_full} fully reconstructed "
                      f"({100*n_full/n_total:.1f}%)", fontsize=13)
        plt.tight_layout()
        pdf.savefig(fig2)
        plt.close(fig2)

        # ── Collect matched rec and delta arrays per particle ─────
        rec_data = []
        for k in range(npart):
            rp, rt, rphi, rvz = [], [], [], []
            mp, mt, mphi, mvz = [], [], [], []
            for ev in events:
                pp = ev['particles'][k]
                if pp['status'] > 0:
                    rp.append(pp['p_rec'])
                    rt.append(pp['theta_rec'])
                    rphi.append(pp['phi_rec'])
                    rvz.append(pp['vz_rec'])
                    mp.append(pp['p_gen'])
                    mt.append(pp['theta_gen'])
                    mphi.append(pp['phi_gen'])
                    mvz.append(pp['vz_gen'])
            rec_data.append({
                'rec_p': np.array(rp), 'rec_theta': np.array(rt),
                'rec_phi': np.array(rphi), 'rec_vz': np.array(rvz),
                'mc_p': np.array(mp), 'mc_theta': np.array(mt),
                'mc_phi': np.array(mphi), 'mc_vz': np.array(mvz),
            })

        # ── Page 3: 2D plots — theta vs p, theta vs phi ──────────
        # Theta ranges per particle: e-(0-35), p(0-70), h+(0-60), h-(0-60)
        THETA_MAX = [35, 70, 60, 60]

        fig3, axes3 = plt.subplots(npart, 4, figsize=(22, 4 * npart))
        if npart == 1:
            axes3 = axes3[np.newaxis, :]

        for k in range(npart):
            gp_k = np.array(gen_p[k])
            gt_k = np.array(gen_theta[k])
            gphi_k = np.array(gen_phi[k])
            rd = rec_data[k]
            th_max = THETA_MAX[k]

            axes3[k, 0].hist2d(gp_k, gt_k, bins=100,
                               range=[[0, gp_k.max()*1.1], [0, th_max]], cmap="Blues")
            axes3[k, 0].set_xlabel("|p| (GeV)")
            axes3[k, 0].set_ylabel(f"{labels[k]}  θ (deg)")
            axes3[k, 0].set_ylim(0, th_max)
            if k == 0:
                axes3[k, 0].set_title("θ vs |p|  (generated)")

            if len(rd['rec_p']) > 0:
                axes3[k, 1].hist2d(rd['rec_p'], rd['rec_theta'], bins=100,
                                   range=[[0, rd['rec_p'].max()*1.1], [0, th_max]], cmap="Reds")
            axes3[k, 1].set_xlabel("|p| (GeV)")
            axes3[k, 1].set_ylabel("θ (deg)")
            axes3[k, 1].set_ylim(0, th_max)
            if k == 0:
                axes3[k, 1].set_title("θ vs |p|  (reconstructed)")

            axes3[k, 2].hist2d(gphi_k, gt_k, bins=100,
                               range=[[-180, 180], [0, th_max]], cmap="Blues")
            axes3[k, 2].set_xlabel("φ (deg)")
            axes3[k, 2].set_ylabel("θ (deg)")
            axes3[k, 2].set_ylim(0, th_max)
            if k == 0:
                axes3[k, 2].set_title("θ vs φ  (generated)")

            if len(rd['rec_phi']) > 0:
                axes3[k, 3].hist2d(rd['rec_phi'], rd['rec_theta'], bins=100,
                                   range=[[-180, 180], [0, th_max]], cmap="Reds")
            axes3[k, 3].set_xlabel("φ (deg)")
            axes3[k, 3].set_ylabel("θ (deg)")
            axes3[k, 3].set_ylim(0, th_max)
            if k == 0:
                axes3[k, 3].set_title("θ vs φ  (reconstructed)")

        fig3.suptitle(f"{reaction}  —  2D kinematics  —  Ebeam={beam_energy} GeV", fontsize=13)
        plt.tight_layout()
        pdf.savefig(fig3)
        plt.close(fig3)

        # ── Page 4: 1D resolution — Δp, Δθ, Δφ, Δvz ─────────────
        DELTA_KW = dict(histtype="stepfilled", color="plum", edgecolor="purple", alpha=0.7)
        fig4, axes4 = plt.subplots(npart, 4, figsize=(22, 4 * npart))
        if npart == 1:
            axes4 = axes4[np.newaxis, :]

        for k in range(npart):
            rd = rec_data[k]
            if len(rd['rec_p']) == 0:
                for c in range(4):
                    axes4[k, c].text(0.5, 0.5, "No data", transform=axes4[k, c].transAxes,
                                     ha="center", va="center")
                    axes4[k, c].set_ylabel(f"{labels[k]}")
                continue

            dp = rd['rec_p'] - rd['mc_p']
            dt = rd['rec_theta'] - rd['mc_theta']
            dphi = rd['rec_phi'] - rd['mc_phi']
            dphi = np.where(dphi > 180, dphi - 360, dphi)
            dphi = np.where(dphi < -180, dphi + 360, dphi)
            dvz = rd['rec_vz'] - rd['mc_vz']

            axes4[k, 0].hist(dp, bins=100, range=(-0.3, 0.3), **DELTA_KW)
            axes4[k, 0].set_xlabel("Δ|p| (GeV)")
            axes4[k, 0].set_ylabel(f"{labels[k]}")
            annotate(axes4[k, 0], dp, loc="right")
            if k == 0:
                axes4[k, 0].set_title("Δ|p|")

            axes4[k, 1].hist(dt, bins=100, range=(-1, 1), **DELTA_KW)
            axes4[k, 1].set_xlabel("Δθ (deg)")
            annotate(axes4[k, 1], dt, loc="right")
            if k == 0:
                axes4[k, 1].set_title("Δθ")

            axes4[k, 2].hist(dphi, bins=100, range=(-2, 2), **DELTA_KW)
            axes4[k, 2].set_xlabel("Δφ (deg)")
            annotate(axes4[k, 2], dphi, loc="right")
            if k == 0:
                axes4[k, 2].set_title("Δφ")

            axes4[k, 3].hist(dvz, bins=100, range=(-5, 5), **DELTA_KW)
            axes4[k, 3].set_xlabel("Δvz (cm)")
            annotate(axes4[k, 3], dvz, loc="right")
            if k == 0:
                axes4[k, 3].set_title("Δvz")

        fig4.suptitle(f"{reaction}  —  Resolution: Δ = Reconstructed − Generated  —  "
                      f"Ebeam={beam_energy} GeV", fontsize=13)
        plt.tight_layout()
        pdf.savefig(fig4)
        plt.close(fig4)

        # ── Page 5: 2D resolution — Δp/p vs p, Δθ vs θ, Δφ vs φ, Δvz vs θ ──
        fig5, axes5 = plt.subplots(npart, 4, figsize=(22, 4 * npart))
        if npart == 1:
            axes5 = axes5[np.newaxis, :]

        for k in range(npart):
            rd = rec_data[k]
            if len(rd['rec_p']) == 0:
                for c in range(4):
                    axes5[k, c].text(0.5, 0.5, "No data", transform=axes5[k, c].transAxes,
                                     ha="center", va="center")
                    axes5[k, c].set_ylabel(f"{labels[k]}")
                continue

            dp = rd['rec_p'] - rd['mc_p']
            dt = rd['rec_theta'] - rd['mc_theta']
            dphi = rd['rec_phi'] - rd['mc_phi']
            dphi = np.where(dphi > 180, dphi - 360, dphi)
            dphi = np.where(dphi < -180, dphi + 360, dphi)
            dvz = rd['rec_vz'] - rd['mc_vz']
            dpp = dp / rd['mc_p']

            axes5[k, 0].hist2d(rd['mc_p'], dpp, bins=[50, 50],
                               range=[[0, rd['mc_p'].max()*1.1], [-0.05, 0.05]],
                               cmap="viridis")
            axes5[k, 0].axhline(y=0, color="red", linewidth=0.8)
            axes5[k, 0].set_xlabel("|p|_MC (GeV)")
            axes5[k, 0].set_ylabel(f"{labels[k]}  Δ|p|/|p|")
            if k == 0:
                axes5[k, 0].set_title("Δ|p|/|p| vs |p|")

            axes5[k, 1].hist2d(rd['mc_theta'], dt, bins=[50, 50],
                               range=[[0, rd['mc_theta'].max()*1.1], [-0.5, 0.5]],
                               cmap="viridis")
            axes5[k, 1].axhline(y=0, color="red", linewidth=0.8)
            axes5[k, 1].set_xlabel("θ_MC (deg)")
            axes5[k, 1].set_ylabel("Δθ (deg)")
            if k == 0:
                axes5[k, 1].set_title("Δθ vs θ")

            axes5[k, 2].hist2d(rd['mc_phi'], dphi, bins=[50, 50],
                               range=[[rd['mc_phi'].min()-5, rd['mc_phi'].max()+5], [-1, 1]],
                               cmap="viridis")
            axes5[k, 2].axhline(y=0, color="red", linewidth=0.8)
            axes5[k, 2].set_xlabel("φ_MC (deg)")
            axes5[k, 2].set_ylabel("Δφ (deg)")
            if k == 0:
                axes5[k, 2].set_title("Δφ vs φ")

            axes5[k, 3].hist2d(rd['mc_theta'], dvz, bins=[50, 50],
                               range=[[0, rd['mc_theta'].max()*1.1], [-2, 2]],
                               cmap="viridis")
            axes5[k, 3].axhline(y=0, color="red", linewidth=0.8)
            axes5[k, 3].set_xlabel("θ_MC (deg)")
            axes5[k, 3].set_ylabel("Δvz (cm)")
            if k == 0:
                axes5[k, 3].set_title("Δvz vs θ")

        fig5.suptitle(f"{reaction}  —  Resolution vs kinematics  —  "
                      f"Ebeam={beam_energy} GeV", fontsize=13)
        plt.tight_layout()
        pdf.savefig(fig5)
        plt.close(fig5)

    print(f"  Saved plots: {pdf_path}")


# ── Main ──────────────────────────────────────────────────────────
def main():
    parser = build_parser()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    beam_id = parse_particle_id(args.beam_id)
    target_id = parse_particle_id(args.target_id)
    beam_energy = args.beam_energy

    beam_name = PDG_TO_NAME.get(beam_id, str(beam_id))
    target_name = PDG_TO_NAME.get(target_id, str(target_id))

    hipo_dir = args.hipo_dir
    if not os.path.isdir(hipo_dir):
        print(f"Error: '{hipo_dir}' is not a directory")
        sys.exit(1)

    hipo_files = sorted(glob.glob(os.path.join(hipo_dir, "*.hipo")))
    if not hipo_files:
        print(f"Error: no *.hipo files found in {hipo_dir}")
        sys.exit(1)
    n_total_files = len(hipo_files)
    hipo_files = hipo_files[args.file_offset:]
    if args.max_files > 0:
        hipo_files = hipo_files[:args.max_files]
    print(f"Found {n_total_files} HIPO files in {hipo_dir}")
    if args.file_offset > 0 or args.max_files > 0:
        print(f"Processing files {args.file_offset+1}–{args.file_offset+len(hipo_files)} "
              f"({len(hipo_files)} files)")

    # Auto-detect reaction
    reaction = detect_reaction(hipo_files)
    labels = FINAL_STATE_LABELS[reaction]
    mass_label = MESON_MASS_LABEL[reaction]
    npart = len(labels)
    print(f"Detected reaction: {reaction}  ({', '.join(labels)})")
    print(f"Beam: {beam_name} ({beam_id}), "
          f"Energy: {beam_energy} GeV, "
          f"Target: {target_name} ({target_id})")

    # Matching cuts
    mc_cuts = None
    if args.matching_cuts:
        mc_cuts = MatchingCuts(args.matching_cuts)
        print(f"Matching cuts: {args.matching_cuts} ({mc_cuts.n_sigma:.0f}σ, "
              f"particles: {mc_cuts.particles})")
    else:
        print("Matching cuts: flat MATCH_WINDOWS (legacy)")

    # Output directory
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    out_base = os.path.join(out_dir, args.output)

    # Build header
    header_lines = [
        f"#! reaction: {reaction}",
        f"#! beam: {beam_name} ({beam_id})",
        f"#! beam_energy: {beam_energy}",
        f"#! target: {target_name} ({target_id})",
        f"#! source: {os.path.abspath(hipo_dir)}",
        f"#! columns_event: event_num nrec Q2 xB W t {mass_label}",
        f"#! matching: e+/e- by exact PID (PCAL/HTCC reliable); hadrons by CHARGE SIGN + greedy assignment (PID-blind); cuts={'momentum-dependent '+str(mc_cuts.n_sigma)+'sigma from '+mc_cuts.source if mc_cuts else 'flat MATCH_WINDOWS (legacy)'}",
        f"#! columns_particle: status(0=not_detected,1=matched_no_PID,2=matched_with_PID) pid det(0=none,1=FD,2=CD,3=FT) p_gen theta_gen phi_gen vz_gen [p_rec theta_rec phi_rec vz_rec]  # rec columns OMITTED when status==0",
    ]

    # Process events — stream to disk, keep compact arrays for plots
    banks = ["MC::Particle", "REC::Particle"]

    expected_key = tuple(sorted(EXPECTED_PIDS[reaction]))
    event_num = 0
    n_skipped = 0
    n_train = 0
    n_val = 0
    done = False

    # Train/val split: use RNG per event to decide on the fly
    rng = np.random.default_rng(seed=42)
    val_frac = args.val_fraction

    # Counters per particle
    n_rec_per_particle = [0] * npart
    nrec_counts = [0] * (npart + 1)  # nrec_counts[k] = events with exactly k particles reconstructed

    t0 = time.time()

    f_train = open(f"{out_base}_train.dat", 'w')
    f_val   = open(f"{out_base}_val.dat", 'w')
    for line in header_lines:
        f_train.write(line + '\n')
        f_val.write(line + '\n')

    try:
        for ifile, hfile in enumerate(hipo_files):
            if done:
                break
            print(f"  File {ifile+1}/{len(hipo_files)}: {os.path.basename(hfile)}  "
                  f"[events so far: {event_num}]")

            chain = hipochain([hfile], banks=banks, step=5000, tags=[0])

            for batch in chain:
                mc_px_list  = batch["MC::Particle_px"]
                mc_py_list  = batch["MC::Particle_py"]
                mc_pz_list  = batch["MC::Particle_pz"]
                mc_vz_list  = batch["MC::Particle_vz"]
                mc_pid_list = batch["MC::Particle_pid"]

                rec_px_list     = batch["REC::Particle_px"]
                rec_py_list     = batch["REC::Particle_py"]
                rec_pz_list     = batch["REC::Particle_pz"]
                rec_vz_list     = batch["REC::Particle_vz"]
                rec_pid_list    = batch["REC::Particle_pid"]
                rec_status_list = batch["REC::Particle_status"]

                n_ev = len(mc_px_list)

                for i in range(n_ev):
                    mc_pids = mc_pid_list[i]
                    if len(mc_pids) < npart:
                        continue

                    # Verify PIDs match the detected reaction (order may vary)
                    ev_key = tuple(sorted(int(mc_pids[k]) for k in range(npart)))
                    if ev_key != expected_key:
                        n_skipped += 1
                        continue

                    particles = []
                    nrec = 0

                    # First pass: compute MC kinematics for all particles
                    mc_info = []   # list of dicts {'index', 'pid', 'p', 'theta', 'phi', 'vz', 'mass'}
                    for k in range(npart):
                        mpx = mc_px_list[i][k]
                        mpy = mc_py_list[i][k]
                        mpz = mc_pz_list[i][k]
                        mvz = mc_vz_list[i][k]
                        mc_pid = int(mc_pids[k])
                        mc_p = np.sqrt(mpx**2 + mpy**2 + mpz**2)
                        mc_theta = np.degrees(np.arccos(mpz / mc_p)) if mc_p > 0 else 0.0
                        mc_phi = np.degrees(np.arctan2(mpy, mpx))
                        mc_info.append({
                            'index': k, 'pid': mc_pid, 'p': mc_p,
                            'theta': mc_theta, 'phi': mc_phi, 'vz': mvz,
                            'mass': PDG_MASS.get(mc_pid, 0.0),
                        })

                    # Electron(s): match by exact PID (PCAL/HTCC PID is reliable)
                    electron_results = {}   # index -> (found, rp, rtheta, rphi, rvz, det, rec_pid)
                    for mc in mc_info:
                        if abs(mc['pid']) == 11:    # e- or e+
                            found, rp, rtheta, rphi, rvz, det, rec_pid = match_rec_particle(
                                mc['pid'], mc['p'], mc['theta'], mc['phi'],
                                rec_pid_list[i], rec_px_list[i], rec_py_list[i],
                                rec_pz_list[i], rec_vz_list[i], rec_status_list[i],
                                mc_cuts=mc_cuts)
                            electron_results[mc['index']] = (found, rp, rtheta, rphi, rvz, det, rec_pid)

                    # Hadrons: charge-blind greedy assignment
                    hadron_list = [mc for mc in mc_info if abs(mc['pid']) != 11]
                    hadron_results = match_hadrons_greedy(
                        hadron_list,
                        rec_pid_list[i], rec_px_list[i], rec_py_list[i],
                        rec_pz_list[i], rec_vz_list[i], rec_status_list[i],
                        mc_cuts=mc_cuts)

                    # Combine results back per MC particle in input order
                    for mc in mc_info:
                        k = mc['index']
                        if k in electron_results:
                            found, rp, rtheta, rphi, rvz, det, rec_pid = electron_results[k]
                        elif k in hadron_results:
                            found, rp, rtheta, rphi, rvz, det, rec_pid = hadron_results[k]
                        else:
                            found, rp, rtheta, rphi, rvz, det, rec_pid = (False, -999.0, -999.0, -999.0, -999.0, 0, 0)

                        if not found:
                            status = 0
                        elif rec_pid == mc['pid']:
                            status = 2   # matched with PID
                        else:
                            status = 1   # matched without PID (charge-only)
                        nrec += (1 if found else 0)

                        particles.append({
                            'status': status,
                            'pid': mc['pid'],
                            'p_gen': mc['p'],
                            'theta_gen': mc['theta'],
                            'phi_gen': mc['phi'],
                            'vz_gen': mc['vz'],
                            'p_rec': rp,
                            'theta_rec': rtheta,
                            'phi_rec': rphi,
                            'vz_rec': rvz,
                            'det': det,        # 0=none, 1=FD, 2=CD
                            'mass': mc['mass'],
                        })

                    event_num += 1
                    Q2, xB, W, t, Mh, y, nu = compute_kinematics(
                        beam_energy, beam_id, target_id,
                        [(p['pid'], p['p_gen'], p['theta_gen'], p['phi_gen'], p['mass'])
                         for p in particles])

                    # Write to train or val file immediately
                    event_line = f"{event_num}  {nrec}  {Q2:.4f}  {xB:.4f}  {W:.4f}  {t:.4f}  {Mh:.4f}\n"
                    plines = []
                    for p in particles:
                        if p['status'] > 0:
                            # detected: status pid det p_gen θ_gen φ_gen vz_gen p_rec θ_rec φ_rec vz_rec
                            plines.append(f" {p['status']}  {p['pid']:>5d}  {p['det']:1d}  "
                                f"{p['p_gen']:8.4f}  {p['theta_gen']:8.3f}  {p['phi_gen']:8.3f}  {p['vz_gen']:7.3f}  "
                                f"{p['p_rec']:8.4f}  {p['theta_rec']:8.3f}  {p['phi_rec']:8.3f}  {p['vz_rec']:7.3f}\n")
                        else:
                            # not detected: status pid det p_gen θ_gen φ_gen vz_gen   (rec columns omitted)
                            plines.append(f" {p['status']}  {p['pid']:>5d}  {p['det']:1d}  "
                                f"{p['p_gen']:8.4f}  {p['theta_gen']:8.3f}  {p['phi_gen']:8.3f}  {p['vz_gen']:7.3f}\n")

                    if rng.random() < val_frac:
                        f_val.write(event_line)
                        for pl in plines:
                            f_val.write(pl)
                        n_val += 1
                    else:
                        f_train.write(event_line)
                        for pl in plines:
                            f_train.write(pl)
                        n_train += 1

                    # Update counters
                    nrec_counts[nrec] += 1
                    for k in range(npart):
                        n_rec_per_particle[k] += (1 if particles[k]['status'] > 0 else 0)

                    if args.max_events > 0 and event_num >= args.max_events:
                        done = True
                        break

                if done:
                    break

    finally:
        f_train.close()
        f_val.close()

    t_read = time.time() - t0

    n_total = event_num
    n_full = nrec_counts[npart]
    print(f"\nProcessing time: {t_read:.1f} s")
    if n_skipped > 0:
        print(f"Skipped (wrong channel): {n_skipped}")
    print(f"Total events:  {n_total}")
    print(f"Fully reconstructed: {n_full} ({100*n_full/n_total:.1f}%)")
    for k in range(npart):
        print(f"  {labels[k]:>4s} reconstructed: {n_rec_per_particle[k]} ({100*n_rec_per_particle[k]/n_total:.1f}%)")

    print(f"\nTrain/val split: {n_train} / {n_val}")
    print(f"  Wrote {n_train} events to {out_base}_train.dat")
    print(f"  Wrote {n_val} events to {out_base}_val.dat")
    print(f"\nTo produce plots run:")
    print(f"  python plot_training_data.py {out_base}_train.dat")

    print("\nDone.")


if __name__ == "__main__":
    main()
