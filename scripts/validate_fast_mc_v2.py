#!/usr/bin/env python3
"""Validate fast MC v2: compare FastMC vs GEMC using event-level .dat files.

Reads the validation file, runs FastMC on generated kinematics,
compares per-particle and full-event distributions.

Usage:
  python validate_fast_mc_v2.py phi_val.dat -m models/phi_v2 -o plots/phi_v2
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm

from fast_mc import FastMC, load_model_auto

PDG_TO_SHORT = {
    11: 'e-', -11: 'e+', 13: 'mu-', -13: 'mu+',
    2212: 'p', 2112: 'n',
    211: 'pi+', -211: 'pi-',
    321: 'K+', -321: 'K-',
    22: 'gamma',
}

PDG_MASS = {
    11: 0.000511, -11: 0.000511,
    2212: 0.938272, 2112: 0.939565,
    211: 0.13957, -211: 0.13957,
    321: 0.49368, -321: 0.49368,
}


def parse_args():
    p = argparse.ArgumentParser(description="Validate fast MC v2")
    p.add_argument("input", help="Validation data file (.dat)")
    p.add_argument("-m", "--model_dir", required=True, help="Model directory")
    p.add_argument("-o", "--output", default="plots/phi_v2", help="Output directory")
    p.add_argument("--max_events", type=int, default=0, help="Max events (0=all)")
    return p.parse_args()


def read_val_data(filename, max_events=0):
    """Read .dat validation file. Returns header, list of events."""
    header = {}
    events = []
    current_event = None
    current_particles = []

    with open(filename, 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('#!'):
                kv = line[2:].strip()
                if ':' in kv:
                    k, v = kv.split(':', 1)
                    header[k.strip()] = v.strip()
                continue
            if not line.strip():
                continue

            if line[0] == ' ':
                parts = line.split()
                status = int(parts[0])

                # Detect format: new compact format has det as 3rd column,
                # old format has det as 11th column (or missing).
                #
                # New format (compact):
                #   status pid det p_gen theta_gen phi_gen vz_gen [p_rec theta_rec phi_rec vz_rec]
                #   7 or 11 columns
                #
                # Old format:
                #   status pid p_gen theta_gen phi_gen vz_gen p_rec theta_rec phi_rec vz_rec [det]
                #   10 or 11 columns
                #
                # Distinguish: in new format, parts[2] is det (small int 0-3).
                # In old format, parts[2] is p_gen (float like "5.2804").
                # A '.' in parts[2] means old format.
                if '.' not in parts[2]:
                    # ── New compact format ──
                    det = int(parts[2])
                    p_gen     = float(parts[3])
                    theta_gen = float(parts[4])
                    phi_gen   = float(parts[5])
                    vz_gen    = float(parts[6])
                    if status > 0 and len(parts) >= 11:
                        p_rec     = float(parts[7])
                        theta_rec = float(parts[8])
                        phi_rec   = float(parts[9])
                        vz_rec    = float(parts[10])
                    else:
                        p_rec = theta_rec = phi_rec = vz_rec = -999.0
                else:
                    # ── Old format ──
                    p_gen     = float(parts[2])
                    theta_gen = float(parts[3])
                    phi_gen   = float(parts[4])
                    vz_gen    = float(parts[5])
                    p_rec     = float(parts[6])
                    theta_rec = float(parts[7])
                    phi_rec   = float(parts[8])
                    vz_rec    = float(parts[9])
                    if len(parts) >= 11:
                        det = int(parts[10])
                    else:
                        det = 1 if status >= 1 else 0

                current_particles.append({
                    'status': status,
                    'pid': int(parts[1]),
                    'p_gen': p_gen,
                    'theta_gen': theta_gen,
                    'phi_gen': phi_gen,
                    'vz_gen': vz_gen,
                    'p_rec': p_rec,
                    'theta_rec': theta_rec,
                    'phi_rec': phi_rec,
                    'vz_rec': vz_rec,
                    'det': det,
                })
                continue

            # Event line — store previous event
            if current_event is not None and current_particles:
                parts_ev = current_event
                events.append({
                    'event_num': parts_ev[0],
                    'nrec': parts_ev[1],
                    'Q2': parts_ev[2], 'xB': parts_ev[3],
                    'W': parts_ev[4], 't': parts_ev[5], 'Mh': parts_ev[6],
                    'particles': current_particles,
                })

            if max_events > 0 and len(events) >= max_events:
                break

            parts = line.split()
            current_event = (int(parts[0]), int(parts[1]),
                             float(parts[2]), float(parts[3]),
                             float(parts[4]), float(parts[5]), float(parts[6]))
            current_particles = []

    # Last event
    if current_event is not None and current_particles and (max_events == 0 or len(events) < max_events):
        parts_ev = current_event
        events.append({
            'event_num': parts_ev[0],
            'nrec': parts_ev[1],
            'Q2': parts_ev[2], 'xB': parts_ev[3],
            'W': parts_ev[4], 't': parts_ev[5], 'Mh': parts_ev[6],
            'particles': current_particles,
        })

    return header, events


def compute_invariant_mass(p1, theta1, phi1, mass1, p2, theta2, phi2, mass2):
    """Compute invariant mass of two particles."""
    t1, p1r = np.radians(theta1), np.radians(phi1)
    t2, p2r = np.radians(theta2), np.radians(phi2)
    E1 = np.sqrt(p1**2 + mass1**2)
    E2 = np.sqrt(p2**2 + mass2**2)
    px1, py1, pz1 = p1*np.sin(t1)*np.cos(p1r), p1*np.sin(t1)*np.sin(p1r), p1*np.cos(t1)
    px2, py2, pz2 = p2*np.sin(t2)*np.cos(p2r), p2*np.sin(t2)*np.sin(p2r), p2*np.cos(t2)
    M2 = (E1+E2)**2 - (px1+px2)**2 - (py1+py2)**2 - (pz1+pz2)**2
    return np.sqrt(np.maximum(0, M2))


def compute_Q2(beam_energy, p_e, theta_e):
    """Compute Q² from scattered electron."""
    return 2 * beam_energy * p_e * (1 - np.cos(np.radians(theta_e)))


def compute_W(beam_energy, p_e, theta_e, M_target=0.938272):
    """Compute W from scattered electron."""
    Q2 = compute_Q2(beam_energy, p_e, theta_e)
    nu = beam_energy - np.sqrt(p_e**2 + 0.000511**2)
    W2 = M_target**2 + 2*M_target*nu - Q2
    return np.sqrt(np.maximum(0, W2))


def annotate(ax, data, loc="right"):
    data = np.asarray(data)
    data = data[np.isfinite(data)]
    if len(data) == 0:
        return
    n, mean, sig = len(data), np.mean(data), np.std(data)
    txt = f"N={n}\nμ={mean:.4f}\nσ={sig:.4f}"
    x, ha = (0.97, "right") if loc == "right" else (0.03, "left")
    ax.text(x, 0.95, txt, transform=ax.transAxes,
            ha=ha, va="top", fontsize=8, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.9))


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    print(f"Reading {args.input} ...")
    header, events = read_val_data(args.input, max_events=args.max_events)
    reaction = header.get('reaction', 'unknown')
    beam_energy = float(header.get('beam_energy', '10.6'))
    npart = len(events[0]['particles'])
    print(f"  Reaction: {reaction}, beam: {beam_energy} GeV")
    print(f"  {len(events):,} validation events, {npart} particles/event")

    # Identify particles
    pids = [events[0]['particles'][k]['pid'] for k in range(npart)]
    names = [PDG_TO_SHORT.get(pid, str(pid)) for pid in pids]
    print(f"  Particles: {', '.join(names)}")

    # Load FastMC models — auto-detect single ({name}.pt) vs FD/CD pair
    # ({name}_FD.pt + {name}_CD.pt).
    models = {}
    for k in range(npart):
        m = load_model_auto(args.model_dir, names[k])
        if m is None:
            print(f"  WARNING: no model found for {names[k]} in {args.model_dir}")
            continue
        models[k] = m
        kind = type(m).__name__
        print(f"  Loaded {kind} for {names[k]}")

    # Build arrays from validation data
    gen_p = [np.array([ev['particles'][k]['p_gen'] for ev in events]) for k in range(npart)]
    gen_theta = [np.array([ev['particles'][k]['theta_gen'] for ev in events]) for k in range(npart)]
    gen_phi = [np.array([ev['particles'][k]['phi_gen'] for ev in events]) for k in range(npart)]
    gen_vz = [np.array([ev['particles'][k]['vz_gen'] for ev in events]) for k in range(npart)]
    gemc_status = [np.array([ev['particles'][k]['status'] for ev in events]) for k in range(npart)]
    gemc_p_rec = [np.array([ev['particles'][k]['p_rec'] for ev in events]) for k in range(npart)]
    gemc_theta_rec = [np.array([ev['particles'][k]['theta_rec'] for ev in events]) for k in range(npart)]
    gemc_phi_rec = [np.array([ev['particles'][k]['phi_rec'] for ev in events]) for k in range(npart)]
    gemc_vz_rec = [np.array([ev['particles'][k]['vz_rec'] for ev in events]) for k in range(npart)]
    gemc_det    = [np.array([ev['particles'][k]['det']    for ev in events]) for k in range(npart)]
    gemc_nrec = np.array([ev['nrec'] for ev in events])

    # Run FastMC on each particle
    fmc_results = {}
    fmc_accepted = np.ones(len(events), dtype=bool)  # all-4-accepted mask for FastMC
    gemc_all_accepted = (gemc_nrec == npart)

    for k in range(npart):
        if k not in models:
            continue
        result = models[k].simulate(gen_p[k], gen_theta[k], gen_phi[k], gen_vz[k])
        fmc_results[k] = result
        fmc_accepted &= result['accepted']

    n_gemc_full = gemc_all_accepted.sum()
    n_fmc_full = fmc_accepted.sum()
    print(f"\n  GEMC all-{npart} accepted: {n_gemc_full:,} ({100*n_gemc_full/len(events):.4f}%)")
    print(f"  FastMC all-{npart} accepted: {n_fmc_full:,} ({100*n_fmc_full/len(events):.4f}%)")

    # Per-particle stats
    for k in range(npart):
        if k not in fmc_results:
            continue
        n_gemc_k = gemc_status[k].sum()
        n_fmc_k = fmc_results[k]['accepted'].sum()
        print(f"    {names[k]:>4s}: GEMC {n_gemc_k:,} ({100*n_gemc_k/len(events):.4f}%) "
              f" FastMC {n_fmc_k:,} ({100*n_fmc_k/len(events):.4f}%)")

    # ── Plots ────────────────────────────────────────────────────────
    # Brighter, higher-contrast palette so overlap doesn't look grey.
    GEN_KW  = dict(histtype="stepfilled", color="#d3d3d3", edgecolor="#7a7a7a",
                   linewidth=1.0, label="Generated", alpha=0.55)
    GEMC_KW = dict(histtype="stepfilled", color="#ff7f50", edgecolor="#b22222",
                   linewidth=1.2, label="GEMC", alpha=0.65)
    FMC_KW  = dict(histtype="step",       color="#1f77b4", edgecolor="#1f77b4",
                   linewidth=1.6, label="FastMC")
    # Optional per-subsystem variants used by the new FD/CD pages.
    GEMC_FD_KW = dict(histtype="step", color="#b22222", linewidth=1.4, label="GEMC FD")
    GEMC_CD_KW = dict(histtype="step", color="#8b0000", linewidth=1.4, linestyle="--", label="GEMC CD")
    FMC_FD_KW  = dict(histtype="step", color="#1f77b4", linewidth=1.4, label="FastMC FD")
    FMC_CD_KW  = dict(histtype="step", color="#0a3d62", linewidth=1.4, linestyle="--", label="FastMC CD")

    pdf_path = os.path.join(args.output, "validation.pdf")
    with PdfPages(pdf_path) as pdf:

        # ── Page 1: Per-particle reconstructed distributions ─────────
        fig, axes = plt.subplots(npart, 4, figsize=(20, 4*npart))
        if npart == 1:
            axes = axes[np.newaxis, :]

        for k in range(npart):
            if k not in fmc_results:
                continue
            gemc_mask = gemc_status[k] == 1
            fmc_mask = fmc_results[k]['accepted']

            # p — generated (grey) underneath, then GEMC, then FastMC
            g_p = gemc_p_rec[k][gemc_mask]
            f_p = fmc_results[k]['p_rec']
            p_max = np.percentile(gen_p[k], 99.5) * 1.1
            axes[k, 0].hist(gen_p[k], bins=80, range=(0, p_max), **GEN_KW)
            axes[k, 0].hist(g_p,      bins=80, range=(0, p_max), **GEMC_KW)
            axes[k, 0].hist(f_p,      bins=80, range=(0, p_max), **FMC_KW)
            axes[k, 0].set_xlabel("|p| (GeV)")
            axes[k, 0].set_ylabel(names[k])
            if k == 0:
                axes[k, 0].set_title("|p|  (Generated / GEMC / FastMC)")
                axes[k, 0].legend(fontsize=7)

            # theta
            g_t = gemc_theta_rec[k][gemc_mask]
            f_t = fmc_results[k]['theta_rec']
            th_max = min(70, np.percentile(gen_theta[k], 99.5) * 1.2)
            axes[k, 1].hist(gen_theta[k], bins=80, range=(0, th_max), **GEN_KW)
            axes[k, 1].hist(g_t,          bins=80, range=(0, th_max), **GEMC_KW)
            axes[k, 1].hist(f_t,          bins=80, range=(0, th_max), **FMC_KW)
            axes[k, 1].set_xlabel("θ (deg)")
            if k == 0:
                axes[k, 1].set_title("θ  (Generated / GEMC / FastMC)")

            # phi
            g_phi = gemc_phi_rec[k][gemc_mask]
            f_phi = fmc_results[k]['phi_rec']
            axes[k, 2].hist(gen_phi[k], bins=80, range=(-180, 180), **GEN_KW)
            axes[k, 2].hist(g_phi,      bins=80, range=(-180, 180), **GEMC_KW)
            axes[k, 2].hist(f_phi,      bins=80, range=(-180, 180), **FMC_KW)
            axes[k, 2].set_xlabel("φ (deg)")
            if k == 0:
                axes[k, 2].set_title("φ  (Generated / GEMC / FastMC)")

            # vz
            g_vz = gemc_vz_rec[k][gemc_mask]
            f_vz = fmc_results[k]['vz_rec']
            axes[k, 3].hist(gen_vz[k], bins=80, range=(-8, 2), **GEN_KW)
            axes[k, 3].hist(g_vz,      bins=80, range=(-8, 2), **GEMC_KW)
            axes[k, 3].hist(f_vz,      bins=80, range=(-8, 2), **FMC_KW)
            axes[k, 3].set_xlabel("vz (cm)")
            if k == 0:
                axes[k, 3].set_title("vz  (Generated / GEMC / FastMC)")

        # All axes on page 1 → log Y so the generated curve doesn't crush
        # the GEMC/FastMC overlay.
        for ax in axes.flat:
            ax.set_yscale("log")

        fig.suptitle(f"Page 1: Per-particle GEN/GEMC/FastMC  (log Y)  — {reaction} — "
                     f"{len(events):,} events", fontsize=13)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ── Page 1a: same per-particle reconstructed distributions but
        # WITHOUT the generated overlay, linear Y — the clean GEMC vs
        # FastMC comparison.
        fig1a, axes1a = plt.subplots(npart, 4, figsize=(20, 4*npart))
        if npart == 1:
            axes1a = axes1a[np.newaxis, :]

        for k in range(npart):
            if k not in fmc_results:
                continue
            gemc_mask = gemc_status[k] == 1

            g_p   = gemc_p_rec[k][gemc_mask]
            g_t   = gemc_theta_rec[k][gemc_mask]
            g_phi = gemc_phi_rec[k][gemc_mask]
            g_vz  = gemc_vz_rec[k][gemc_mask]
            f_p   = fmc_results[k]['p_rec']
            f_t   = fmc_results[k]['theta_rec']
            f_phi = fmc_results[k]['phi_rec']
            f_vz  = fmc_results[k]['vz_rec']

            p_max  = np.percentile(gen_p[k],     99.5) * 1.1
            th_max = min(70, np.percentile(gen_theta[k], 99.5) * 1.2)

            axes1a[k, 0].hist(g_p, bins=80, range=(0, p_max), **GEMC_KW)
            axes1a[k, 0].hist(f_p, bins=80, range=(0, p_max), **FMC_KW)
            axes1a[k, 0].set_xlabel("|p| (GeV)"); axes1a[k, 0].set_ylabel(names[k])
            if k == 0:
                axes1a[k, 0].set_title("|p|  reconstructed  (GEMC / FastMC)")
                axes1a[k, 0].legend(fontsize=7)

            axes1a[k, 1].hist(g_t, bins=80, range=(0, th_max), **GEMC_KW)
            axes1a[k, 1].hist(f_t, bins=80, range=(0, th_max), **FMC_KW)
            axes1a[k, 1].set_xlabel("θ (deg)")
            if k == 0: axes1a[k, 1].set_title("θ  reconstructed")

            axes1a[k, 2].hist(g_phi, bins=80, range=(-180, 180), **GEMC_KW)
            axes1a[k, 2].hist(f_phi, bins=80, range=(-180, 180), **FMC_KW)
            axes1a[k, 2].set_xlabel("φ (deg)")
            if k == 0: axes1a[k, 2].set_title("φ  reconstructed")

            axes1a[k, 3].hist(g_vz, bins=80, range=(-8, 2), **GEMC_KW)
            axes1a[k, 3].hist(f_vz, bins=80, range=(-8, 2), **FMC_KW)
            axes1a[k, 3].set_xlabel("vz (cm)")
            if k == 0: axes1a[k, 3].set_title("vz  reconstructed")

        fig1a.suptitle(f"Page 1a: Per-particle GEMC vs FastMC (no GEN, linear Y) — "
                       f"{reaction} — {len(events):,} events", fontsize=13)
        plt.tight_layout()
        pdf.savefig(fig1a)
        plt.close(fig1a)

        # ── Page 1b: Per-particle reconstructed broken down by FD vs CD ──
        # Same 4-column layout as page 1, but each particle now shows 4
        # curves (GEMC FD, GEMC CD, FastMC FD, FastMC CD).  For FastMC, FD
        # vs CD is decided by reconstructed theta < 35° (proxy) since the
        # combined model doesn't tag subsystem.
        FD_CD_THRESH = 35.0
        fig_fc, axes_fc = plt.subplots(npart, 4, figsize=(20, 4*npart))
        if npart == 1:
            axes_fc = axes_fc[np.newaxis, :]

        for k in range(npart):
            if k not in fmc_results:
                continue
            gemc_mask = gemc_status[k] == 1
            gemc_fd   = gemc_mask & (gemc_det[k] == 1)
            gemc_cd   = gemc_mask & (gemc_det[k] == 2)

            fmc_p   = fmc_results[k]['p_rec']
            fmc_t   = fmc_results[k]['theta_rec']
            fmc_phi = fmc_results[k]['phi_rec']
            fmc_vz  = fmc_results[k]['vz_rec']
            fmc_fd_mask = fmc_t <  FD_CD_THRESH
            fmc_cd_mask = fmc_t >= FD_CD_THRESH

            p_max  = np.percentile(gen_p[k],     99.5) * 1.1
            th_max = min(70, np.percentile(gen_theta[k], 99.5) * 1.2)

            # |p|
            ax = axes_fc[k, 0]
            ax.hist(gemc_p_rec[k][gemc_fd],   bins=80, range=(0, p_max), **GEMC_FD_KW)
            ax.hist(gemc_p_rec[k][gemc_cd],   bins=80, range=(0, p_max), **GEMC_CD_KW)
            ax.hist(fmc_p[fmc_fd_mask],       bins=80, range=(0, p_max), **FMC_FD_KW)
            ax.hist(fmc_p[fmc_cd_mask],       bins=80, range=(0, p_max), **FMC_CD_KW)
            ax.set_xlabel("|p| (GeV)"); ax.set_ylabel(names[k])
            if k == 0:
                ax.set_title("|p| (FD/CD breakdown)")
                ax.legend(fontsize=6, loc="upper right")

            # theta
            ax = axes_fc[k, 1]
            ax.hist(gemc_theta_rec[k][gemc_fd], bins=80, range=(0, th_max), **GEMC_FD_KW)
            ax.hist(gemc_theta_rec[k][gemc_cd], bins=80, range=(0, th_max), **GEMC_CD_KW)
            ax.hist(fmc_t[fmc_fd_mask],         bins=80, range=(0, th_max), **FMC_FD_KW)
            ax.hist(fmc_t[fmc_cd_mask],         bins=80, range=(0, th_max), **FMC_CD_KW)
            ax.axvline(FD_CD_THRESH, color="gray", ls=":", lw=0.7)
            ax.set_xlabel("θ (deg)")
            if k == 0:
                ax.set_title("θ (FD/CD breakdown)")

            # phi
            ax = axes_fc[k, 2]
            ax.hist(gemc_phi_rec[k][gemc_fd], bins=80, range=(-180, 180), **GEMC_FD_KW)
            ax.hist(gemc_phi_rec[k][gemc_cd], bins=80, range=(-180, 180), **GEMC_CD_KW)
            ax.hist(fmc_phi[fmc_fd_mask],     bins=80, range=(-180, 180), **FMC_FD_KW)
            ax.hist(fmc_phi[fmc_cd_mask],     bins=80, range=(-180, 180), **FMC_CD_KW)
            ax.set_xlabel("φ (deg)")
            if k == 0:
                ax.set_title("φ (FD/CD breakdown)")

            # vz
            ax = axes_fc[k, 3]
            ax.hist(gemc_vz_rec[k][gemc_fd], bins=80, range=(-8, 2), **GEMC_FD_KW)
            ax.hist(gemc_vz_rec[k][gemc_cd], bins=80, range=(-8, 2), **GEMC_CD_KW)
            ax.hist(fmc_vz[fmc_fd_mask],     bins=80, range=(-8, 2), **FMC_FD_KW)
            ax.hist(fmc_vz[fmc_cd_mask],     bins=80, range=(-8, 2), **FMC_CD_KW)
            ax.set_xlabel("vz (cm)")
            if k == 0:
                ax.set_title("vz (FD/CD breakdown)")

        fig_fc.suptitle(f"Per-particle reconstructed — FD vs CD breakdown — "
                         f"{reaction}  (sum on page 1)", fontsize=13)
        plt.tight_layout()
        pdf.savefig(fig_fc)
        plt.close(fig_fc)

        # ── Page 2: Per-particle acceptance ratios ───────────────────
        fig2, axes2 = plt.subplots(npart, 4, figsize=(20, 4*npart))
        if npart == 1:
            axes2 = axes2[np.newaxis, :]

        for k in range(npart):
            if k not in fmc_results:
                continue
            gemc_mask = gemc_status[k] == 1
            fmc_mask = fmc_results[k]['accepted']

            for col, (var_g, var_f, vlabel, rng) in enumerate([
                (gemc_p_rec[k][gemc_mask], fmc_results[k]['p_rec'],
                 "|p| (GeV)", (0, np.percentile(gen_p[k], 99.5)*1.1)),
                (gemc_theta_rec[k][gemc_mask], fmc_results[k]['theta_rec'],
                 "θ (deg)", (0, min(70, np.percentile(gen_theta[k], 99.5)*1.2))),
                (gemc_phi_rec[k][gemc_mask], fmc_results[k]['phi_rec'],
                 "φ (deg)", (-180, 180)),
                (gemc_vz_rec[k][gemc_mask], fmc_results[k]['vz_rec'],
                 "vz (cm)", (-8, 2)),
            ]):
                h_g, edges = np.histogram(var_g, bins=50, range=rng)
                h_f, _ = np.histogram(var_f, bins=50, range=rng)
                centers = 0.5 * (edges[:-1] + edges[1:])
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.where(h_g > 0, h_f / h_g, np.nan)
                axes2[k, col].plot(centers, ratio, "k.", markersize=4)
                axes2[k, col].axhline(y=1, color="red", linewidth=1)
                axes2[k, col].set_xlabel(vlabel)
                axes2[k, col].set_ylabel("FastMC/GEMC")
                axes2[k, col].set_ylim(0.5, 1.5)
                if k == 0:
                    axes2[k, col].set_title(f"Ratio {vlabel}")
                if col == 0:
                    axes2[k, col].set_ylabel(f"{names[k]}  FastMC/GEMC")

        fig2.suptitle(f"Per-particle ratios FastMC/GEMC — {reaction}", fontsize=13)
        plt.tight_layout()
        pdf.savefig(fig2)
        plt.close(fig2)

        # ── Per-particle 2D acceptance efficiency: θ vs |p|, θ vs φ, φ vs |p|  ──
        # For each particle, three columns of 2D plots, three rows:
        #   row 0: GEMC efficiency     = N_GEMC_accepted / N_generated
        #   row 1: FastMC efficiency   = N_FastMC_accepted / N_generated
        #   row 2: FastMC / GEMC ratio (should be ~1 everywhere if model is good)
        # Binning uses generated kinematics (denominator).
        for k in range(npart):
            if k not in fmc_results:
                continue

            g_p   = gen_p[k]
            g_t   = gen_theta[k]
            g_phi = gen_phi[k]
            sel_gemc = gemc_status[k] == 1
            sel_fmc  = fmc_results[k]['accepted']

            p_max  = np.percentile(g_p, 99.5) * 1.1
            th_max = min(70, np.percentile(g_t, 99.5) * 1.2)

            kinds = [
                ("tp",  g_p,   g_t,   (0, p_max),   (0, th_max),  "|p| (GeV)", "θ (deg)"),
                ("tph", g_phi, g_t,   (-180, 180),  (0, th_max),  "φ (deg)",   "θ (deg)"),
                ("pph", g_p,   g_phi, (0, p_max),   (-180, 180),  "|p| (GeV)", "φ (deg)"),
            ]

            fig2d, axes2d = plt.subplots(3, 3, figsize=(18, 14))
            for col, (kind, xv, yv, xr, yr, xlab, ylab) in enumerate(kinds):
                bins = 60
                # Denominator: all generated (in this kinematic range)
                h_den, xe, ye = np.histogram2d(xv, yv, bins=bins, range=[xr, yr])
                # Numerator: GEMC-accepted
                h_g, _, _ = np.histogram2d(xv[sel_gemc], yv[sel_gemc], bins=bins, range=[xr, yr])
                # Numerator: FastMC-accepted
                h_f, _, _ = np.histogram2d(xv[sel_fmc], yv[sel_fmc], bins=bins, range=[xr, yr])

                # Efficiencies (avoid divide-by-zero)
                with np.errstate(divide="ignore", invalid="ignore"):
                    eff_g = np.where(h_den > 0, h_g / h_den, np.nan)
                    eff_f = np.where(h_den > 0, h_f / h_den, np.nan)
                    ratio = np.where(eff_g > 0, eff_f / eff_g, np.nan)

                # Use percentile-based vmax so colors aren't crushed by edge bins.
                vmax_eff = np.nanpercentile(np.concatenate([eff_g[~np.isnan(eff_g)],
                                                              eff_f[~np.isnan(eff_f)]]) , 99) \
                            if (np.isfinite(eff_g).any() or np.isfinite(eff_f).any()) else 1.0
                vmax_eff = max(vmax_eff, 1e-3)

                # Row 0 — GEMC efficiency
                im0 = axes2d[0, col].imshow(
                    eff_g.T, origin="lower", aspect="auto",
                    extent=[xe[0], xe[-1], ye[0], ye[-1]],
                    cmap="viridis", vmin=0, vmax=vmax_eff)
                axes2d[0, col].set_xlabel(xlab); axes2d[0, col].set_ylabel(ylab)
                axes2d[0, col].set_title(f"GEMC eff  {ylab} vs {xlab}")
                plt.colorbar(im0, ax=axes2d[0, col], pad=0.02)

                # Row 1 — FastMC efficiency
                im1 = axes2d[1, col].imshow(
                    eff_f.T, origin="lower", aspect="auto",
                    extent=[xe[0], xe[-1], ye[0], ye[-1]],
                    cmap="viridis", vmin=0, vmax=vmax_eff)
                axes2d[1, col].set_xlabel(xlab); axes2d[1, col].set_ylabel(ylab)
                axes2d[1, col].set_title(f"FastMC eff  {ylab} vs {xlab}")
                plt.colorbar(im1, ax=axes2d[1, col], pad=0.02)

                # Row 2 — ratio FastMC / GEMC (centered diverging colormap)
                im2 = axes2d[2, col].imshow(
                    ratio.T, origin="lower", aspect="auto",
                    extent=[xe[0], xe[-1], ye[0], ye[-1]],
                    cmap="RdBu_r", vmin=0.5, vmax=1.5)
                axes2d[2, col].set_xlabel(xlab); axes2d[2, col].set_ylabel(ylab)
                axes2d[2, col].set_title(f"FastMC/GEMC eff  {ylab} vs {xlab}")
                plt.colorbar(im2, ax=axes2d[2, col], pad=0.02)

            fig2d.suptitle(f"2-D acceptance efficiency — {names[k]}  ({reaction})\n"
                            "rows: GEMC ε,  FastMC ε,  ratio  (denominator = generated)",
                            fontsize=13)
            plt.tight_layout()
            pdf.savefig(fig2d)
            plt.close(fig2d)

        # ── Page 3: Full-event physics — Q2, xB, W, t, M(K+K-) ─────
        # Compute physics from reconstructed 4-vectors for fully-accepted events
        # GEMC: use events with nrec == npart
        # FastMC: use events where all particles accepted

        fig3, axes3 = plt.subplots(2, 5, figsize=(25, 10))

        # GEMC full events
        gemc_full_idx = np.where(gemc_all_accepted)[0]
        fmc_full_idx = np.where(fmc_accepted)[0]

        # Q2 from gen (already in event data)
        Q2_gen = np.array([ev['Q2'] for ev in events])
        xB_gen = np.array([ev['xB'] for ev in events])
        W_gen = np.array([ev['W'] for ev in events])
        t_gen = np.array([ev['t'] for ev in events])
        Mh_gen = np.array([ev['Mh'] for ev in events])

        # For GEMC-full and FMC-full, use generated kinematics of accepted events
        for col, (var, vlabel, rng, log_y) in enumerate([
            (Q2_gen, "Q²  (GeV²)", (0, 8), True),
            (xB_gen, "xB", (0, 0.8), True),
            (W_gen, "W (GeV)", (1, 5), True),
            (-t_gen, "-t (GeV²)", (0, 5), True),
            (Mh_gen, "M(K+K-) (GeV)", (0.98, 1.06), False),
        ]):
            # Generated (all events with nrec == npart from GEMC)
            axes3[0, col].hist(var[gemc_full_idx], bins=80, range=rng, **GEMC_KW)
            axes3[0, col].hist(var[fmc_full_idx], bins=80, range=rng, **FMC_KW)
            axes3[0, col].set_xlabel(vlabel)
            axes3[0, col].set_ylabel("Counts")
            axes3[0, col].set_title(vlabel)
            if log_y:
                axes3[0, col].set_yscale("log")
            if col == 0:
                axes3[0, col].legend(fontsize=7)

            # Ratio
            h_g, edges = np.histogram(var[gemc_full_idx], bins=50, range=rng)
            h_f, _ = np.histogram(var[fmc_full_idx], bins=50, range=rng)
            centers = 0.5 * (edges[:-1] + edges[1:])
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(h_g > 0, h_f / h_g, np.nan)
            axes3[1, col].plot(centers, ratio, "k.", markersize=4)
            axes3[1, col].axhline(y=1, color="red", linewidth=1)
            axes3[1, col].set_xlabel(vlabel)
            axes3[1, col].set_ylabel("FastMC/GEMC")
            axes3[1, col].set_ylim(0.5, 1.5)

        fig3.suptitle(f"Full-event kinematics (all {npart} particles accepted) — {reaction}\n"
                      f"GEMC: {n_gemc_full:,}  FastMC: {n_fmc_full:,}  "
                      f"({len(events):,} total events)", fontsize=13)
        plt.tight_layout()
        pdf.savefig(fig3)
        plt.close(fig3)

        # ── Page 4: Reconstructed M(K+K-) AND MM(ep->e'p'X) ─────────
        if npart == 4 and 2 in fmc_results and 3 in fmc_results:
            fig4, axes4 = plt.subplots(2, 3, figsize=(18, 11))

            # Indices in the .dat file: 0=e-, 1=p, 2=K-/K+, 3=K+/K-  (per PIDs)
            pid2, pid3 = pids[2], pids[3]
            mass2 = PDG_MASS.get(abs(pid2), 0.13957)
            mass3 = PDG_MASS.get(abs(pid3), 0.13957)

            gemc_full_mask = gemc_all_accepted
            fmc_full_mask  = fmc_accepted

            if gemc_full_mask.sum() > 0:
                # ---- 4-vectors for K-pair invariant mass (top row) ----
                gemc_Mh = compute_invariant_mass(
                    gemc_p_rec[2][gemc_full_mask], gemc_theta_rec[2][gemc_full_mask],
                    gemc_phi_rec[2][gemc_full_mask], mass2,
                    gemc_p_rec[3][gemc_full_mask], gemc_theta_rec[3][gemc_full_mask],
                    gemc_phi_rec[3][gemc_full_mask], mass3)

                # FastMC: smear ALL generated particles so we have rec for any subset
                fmc_psm  = [None]*npart
                fmc_tsm  = [None]*npart
                fmc_phism = [None]*npart
                for kk in range(npart):
                    if kk not in models:
                        continue
                    p_sm, t_sm, phi_sm, vz_sm = models[kk].smear(
                        gen_p[kk], gen_theta[kk], gen_phi[kk], gen_vz[kk])
                    fmc_psm[kk]  = p_sm
                    fmc_tsm[kk]  = t_sm
                    fmc_phism[kk] = phi_sm

                fmc_Mh = compute_invariant_mass(
                    fmc_psm[2][fmc_full_mask], fmc_tsm[2][fmc_full_mask],
                    fmc_phism[2][fmc_full_mask], mass2,
                    fmc_psm[3][fmc_full_mask], fmc_tsm[3][fmc_full_mask],
                    fmc_phism[3][fmc_full_mask], mass3)

                # ---- Missing mass MM(ep -> e'p'X) (bottom row) ----
                # Built from the SAME 4 accepted particles' e' and p',
                # ignoring K+ and K- in the kinematic balance.
                # Beam along +z (electron mass), target proton at rest.
                m_e  = PDG_MASS.get(11, 0.000511)
                m_pr = PDG_MASS.get(2212, 0.938272)
                p_beam = np.sqrt(beam_energy**2 - m_e**2)

                def MM_array(p_e, t_e, ph_e, p_p, t_p, ph_p):
                    te = np.radians(t_e);  pe = np.radians(ph_e)
                    tp = np.radians(t_p);  pp = np.radians(ph_p)
                    Ee = np.sqrt(p_e**2 + m_e**2)
                    Ep = np.sqrt(p_p**2 + m_pr**2)
                    pex = p_e * np.sin(te) * np.cos(pe)
                    pey = p_e * np.sin(te) * np.sin(pe)
                    pez = p_e * np.cos(te)
                    ppx = p_p * np.sin(tp) * np.cos(pp)
                    ppy = p_p * np.sin(tp) * np.sin(pp)
                    ppz = p_p * np.cos(tp)
                    E_X  = beam_energy + m_pr - Ee - Ep
                    px_X = -pex - ppx
                    py_X = -pey - ppy
                    pz_X = p_beam - pez - ppz
                    return np.sqrt(np.maximum(0.0, E_X**2 - px_X**2 - py_X**2 - pz_X**2))

                gemc_MM = MM_array(
                    gemc_p_rec[0][gemc_full_mask], gemc_theta_rec[0][gemc_full_mask],
                    gemc_phi_rec[0][gemc_full_mask],
                    gemc_p_rec[1][gemc_full_mask], gemc_theta_rec[1][gemc_full_mask],
                    gemc_phi_rec[1][gemc_full_mask])

                fmc_MM = MM_array(
                    fmc_psm[0][fmc_full_mask], fmc_tsm[0][fmc_full_mask],
                    fmc_phism[0][fmc_full_mask],
                    fmc_psm[1][fmc_full_mask], fmc_tsm[1][fmc_full_mask],
                    fmc_phism[1][fmc_full_mask])

                # ---- Top row: M(K+K-) ----
                m_range = (0.98, 1.10)
                axes4[0, 0].hist(gemc_Mh, bins=80, range=m_range, **GEMC_KW)
                axes4[0, 0].set_xlabel("M(K+K-) (GeV)")
                axes4[0, 0].set_title("GEMC reconstructed")
                annotate(axes4[0, 0], gemc_Mh)

                axes4[0, 1].hist(fmc_Mh, bins=80, range=m_range, **FMC_KW)
                axes4[0, 1].set_xlabel("M(K+K-) (GeV)")
                axes4[0, 1].set_title("FastMC reconstructed")
                annotate(axes4[0, 1], fmc_Mh)

                axes4[0, 2].hist(gemc_Mh, bins=80, range=m_range, **GEMC_KW)
                axes4[0, 2].hist(fmc_Mh, bins=80, range=m_range, **FMC_KW)
                axes4[0, 2].set_xlabel("M(K+K-) (GeV)")
                axes4[0, 2].set_title("Overlay  (log)")
                axes4[0, 2].set_yscale("log")
                axes4[0, 2].legend(fontsize=8)

                # ---- Bottom row: MM(ep -> e'p'X) ----
                mm_range = (0.6, 1.5)
                axes4[1, 0].hist(gemc_MM, bins=80, range=mm_range, **GEMC_KW)
                axes4[1, 0].axvline(1.019, color='red', ls='--', lw=1)
                axes4[1, 0].set_xlabel("MM(ep -> e'p'X) (GeV)")
                axes4[1, 0].set_title("GEMC")
                annotate(axes4[1, 0], gemc_MM)

                axes4[1, 1].hist(fmc_MM, bins=80, range=mm_range, **FMC_KW)
                axes4[1, 1].axvline(1.019, color='red', ls='--', lw=1)
                axes4[1, 1].set_xlabel("MM(ep -> e'p'X) (GeV)")
                axes4[1, 1].set_title("FastMC")
                annotate(axes4[1, 1], fmc_MM)

                axes4[1, 2].hist(gemc_MM, bins=80, range=mm_range, **GEMC_KW)
                axes4[1, 2].hist(fmc_MM, bins=80, range=mm_range, **FMC_KW)
                axes4[1, 2].axvline(1.019, color='red', ls='--', lw=1)
                axes4[1, 2].set_xlabel("MM(ep -> e'p'X) (GeV)")
                axes4[1, 2].set_title("Overlay  (log)")
                axes4[1, 2].set_yscale("log")
                axes4[1, 2].legend(fontsize=8)

            fig4.suptitle(f"All-4 accepted — top: M(K+K-),  bottom: MM(ep->e'p'X) — {reaction}",
                          fontsize=13)
            plt.tight_layout()
            pdf.savefig(fig4)
            plt.close(fig4)

        # ── Page 5: Summary table ────────────────────────────────────
        fig5, ax5 = plt.subplots(figsize=(12, 7))
        ax5.axis("off")

        rows = [
            ["Reaction", reaction, ""],
            ["Beam energy", f"{beam_energy} GeV", ""],
            ["Validation events", f"{len(events):,}", ""],
            ["", "", ""],
            ["", "GEMC", "FastMC"],
        ]

        for k in range(npart):
            if k not in fmc_results:
                continue
            n_g = int(gemc_status[k].sum())
            n_f = int(fmc_results[k]['accepted'].sum())
            rows.append([f"{names[k]} accepted",
                         f"{n_g:,} ({100*n_g/len(events):.4f}%)",
                         f"{n_f:,} ({100*n_f/len(events):.4f}%)"])

        rows.append(["", "", ""])
        rows.append([f"All {npart} accepted",
                     f"{n_gemc_full:,} ({100*n_gemc_full/len(events):.4f}%)",
                     f"{n_fmc_full:,} ({100*n_fmc_full/len(events):.4f}%)"])

        if n_gemc_full > 0:
            rows.append(["Full-event ratio", "",
                         f"{n_fmc_full/n_gemc_full:.3f}"])

        table = ax5.table(cellText=rows, colLabels=["Quantity", "Value", ""],
                          loc="center", cellLoc="left")
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 1.5)
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor("lightgray")
            if r == 0:
                cell.set_facecolor("lightyellow")
                cell.set_text_props(weight="bold")

        fig5.suptitle(f"Validation Summary — {reaction}", fontsize=14)
        plt.tight_layout()
        pdf.savefig(fig5)
        plt.close(fig5)

    print(f"\nSaved: {pdf_path}")
    import subprocess
    subprocess.Popen(["open", pdf_path])


if __name__ == "__main__":
    main()
