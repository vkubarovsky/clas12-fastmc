#!/usr/bin/env python3
"""Plot kinematic distributions from training .dat files produced by make_training_data.py.

Usage:
  python plot_training_data.py phi_train.dat
  python plot_training_data.py phi_train.dat -o my_plots.pdf
  python plot_training_data.py -h
"""

import argparse
import os
import re
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ── Constants (same as make_training_data.py) ─────────────────────
PDG_TO_SHORT = {
    11: 'e-', -11: 'e+', 13: 'mu-', -13: 'mu+',
    2212: 'p', 2112: 'n',
    211: 'pi+', -211: 'pi-',
    321: 'K+', -321: 'K-',
    22: 'gamma',
}

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

MESON_NOMINAL_MASS = {
    'epK+K-':    1.019,
    'eppi+pi-':  0.775,
    'epe+e-':    3.097,
}

PDG_MASS = {
    11: 0.000511, -11: 0.000511,
    13: 0.10566,  -13: 0.10566,
    2212: 0.938272, 2112: 0.939565,
    211: 0.13957,  -211: 0.13957,
    321: 0.49368,  -321: 0.49368,
    22: 0.0,
}

# axis limits per particle index — J/psi: e'(FT) <= 7 deg, p <= 25, e+/e- <= 40
THETA_MAX = [8, 28, 42, 42]


def annotate(ax, data, loc="right"):
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


def parse_dat_file(filepath):
    """Parse a .dat file. Returns (header_info, events)."""
    header = {}
    events = []

    with open(filepath) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("#!"):
            key_val = line[2:].strip()
            if ':' in key_val:
                key, val = key_val.split(':', 1)
                header[key.strip()] = val.strip()
            i += 1
            continue
        if not line or line.startswith('#'):
            i += 1
            continue

        parts = line.split()
        if len(parts) >= 7:
            ev = {
                'event_num': int(parts[0]),
                'nrec': int(parts[1]),
                'Q2': float(parts[2]),
                'xB': float(parts[3]),
                'W': float(parts[4]),
                't': float(parts[5]),
                'Mh': float(parts[6]),
                'particles': [],
            }
            i += 1
            while i < len(lines):
                pline = lines[i].strip()
                if not pline or pline.startswith('#'):
                    break
                # TruthMatch format:
                #   status pid det p_gen theta_gen phi_gen vz_gen [p_rec theta_rec phi_rec vz_rec]
                # rec columns are OMITTED when status==0 (7-column line)
                pp = pline.split()
                if len(pp) < 7:
                    break
                try:
                    status = int(pp[0])
                except ValueError:
                    break
                if status not in (0, 1, 2):
                    break

                pid = int(pp[1])
                det = int(pp[2])
                p_gen = float(pp[3])
                theta_gen = float(pp[4])
                phi_gen = float(pp[5])
                vz_gen = float(pp[6])

                if status > 0 and len(pp) >= 11:
                    p_rec = float(pp[7])
                    theta_rec = float(pp[8])
                    phi_rec = float(pp[9])
                    vz_rec = float(pp[10])
                else:
                    p_rec = -999.0
                    theta_rec = -999.0
                    phi_rec = -999.0
                    vz_rec = -999.0

                ev['particles'].append({
                    'status': status, 'pid': pid, 'det': det,
                    'p_gen': p_gen, 'theta_gen': theta_gen,
                    'phi_gen': phi_gen, 'vz_gen': vz_gen,
                    'p_rec': p_rec, 'theta_rec': theta_rec,
                    'phi_rec': phi_rec, 'vz_rec': vz_rec,
                })
                i += 1

            if ev['particles']:
                events.append(ev)
        else:
            i += 1

    # Extract beam/target info from header
    beam_pid = 11
    m = re.search(r'\((\-?\d+)\)', header.get('beam', ''))
    if m:
        beam_pid = int(m.group(1))

    target_pid = 2212
    m = re.search(r'\((\-?\d+)\)', header.get('target', ''))
    if m:
        target_pid = int(m.group(1))

    beam_energy = float(header.get('beam_energy', '10.6'))
    reaction = header.get('reaction', '')

    info = {
        'reaction': reaction,
        'beam_pid': beam_pid,
        'beam_energy': beam_energy,
        'target_pid': target_pid,
        'source': header.get('source', ''),
    }

    return info, events


def compute_derived(info, events):
    """Compute y, nu, rec y/nu, rec Mh, missing energy for each event."""
    beam_energy = info['beam_energy']
    beam_pid = info['beam_pid']
    target_pid = info['target_pid']
    beam_mass = 0.000511 if abs(beam_pid) == 11 else 0.10566
    target_mass = 0.938272 if target_pid == 2212 else 0.939565

    for ev in events:
        ep = ev['particles'][0]
        e_E_gen = np.sqrt(ep['p_gen']**2 + beam_mass**2)
        ev['nu'] = beam_energy - e_E_gen
        ev['y'] = ev['nu'] / beam_energy if beam_energy > 0 else -999

        # Reconstructed y, nu
        if ep['status'] > 0:
            e_E_rec = np.sqrt(ep['p_rec']**2 + beam_mass**2)
            ev['nu_rec'] = beam_energy - e_E_rec
            ev['y_rec'] = ev['nu_rec'] / beam_energy if beam_energy > 0 else -999
        else:
            ev['nu_rec'] = None
            ev['y_rec'] = None

        # Reconstructed M(h+h-)
        if len(ev['particles']) >= 4:
            hp = ev['particles'][2]
            hm = ev['particles'][3]
            if hp['status'] > 0 and hm['status'] > 0:
                hp_mass = PDG_MASS.get(hp['pid'], 0.0)
                hm_mass = PDG_MASS.get(hm['pid'], 0.0)
                hp_tr = np.radians(hp['theta_rec'])
                hp_pr = np.radians(hp['phi_rec'])
                hm_tr = np.radians(hm['theta_rec'])
                hm_pr = np.radians(hm['phi_rec'])
                hp_E = np.sqrt(hp['p_rec']**2 + hp_mass**2)
                hm_E = np.sqrt(hm['p_rec']**2 + hm_mass**2)
                hpx = hp['p_rec'] * np.sin(hp_tr) * np.cos(hp_pr)
                hpy = hp['p_rec'] * np.sin(hp_tr) * np.sin(hp_pr)
                hpz = hp['p_rec'] * np.cos(hp_tr)
                hmx = hm['p_rec'] * np.sin(hm_tr) * np.cos(hm_pr)
                hmy = hm['p_rec'] * np.sin(hm_tr) * np.sin(hm_pr)
                hmz = hm['p_rec'] * np.cos(hm_tr)
                Mh2 = (hp_E + hm_E)**2 - (hpx + hmx)**2 - (hpy + hmy)**2 - (hpz + hmz)**2
                ev['Mh_rec'] = np.sqrt(max(0, Mh2))
            else:
                ev['Mh_rec'] = None
        else:
            ev['Mh_rec'] = None

        # Missing energy — generated (all events)
        total_E_gen = 0.0
        for p in ev['particles']:
            mass = PDG_MASS.get(p['pid'], 0.0)
            total_E_gen += np.sqrt(p['p_gen']**2 + mass**2)
        ev['ME_gen'] = beam_energy + target_mass - total_E_gen

        # Missing energy — reconstructed (fully rec only)
        npart = len(ev['particles'])
        if ev['nrec'] == npart:
            total_E_rec = 0.0
            for p in ev['particles']:
                mass = PDG_MASS.get(p['pid'], 0.0)
                total_E_rec += np.sqrt(p['p_rec']**2 + mass**2)
            ev['ME_rec'] = beam_energy + target_mass - total_E_rec
        else:
            ev['ME_rec'] = None


def make_plots(info, events, pdf_path):
    reaction = info['reaction']
    beam_energy = info['beam_energy']

    if reaction not in FINAL_STATE_LABELS:
        print(f"Error: unknown reaction '{reaction}' for plotting")
        sys.exit(1)

    labels = FINAL_STATE_LABELS[reaction]
    mass_label = MESON_MASS_LABEL[reaction]
    npart = len(labels)

    # Collect arrays
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
    ME_gen_all, ME_rec_full = [], []

    for ev in events:
        is_full = (ev['nrec'] == npart)
        Q2_all.append(ev['Q2']); xB_all.append(ev['xB'])
        W_all.append(ev['W']); t_all.append(ev['t']); Mh_all.append(ev['Mh'])
        y_all.append(ev['y']); nu_all.append(ev['nu'])
        ME_gen_all.append(ev['ME_gen'])

        if is_full:
            Q2_full.append(ev['Q2']); xB_full.append(ev['xB'])
            W_full.append(ev['W']); t_full.append(ev['t']); Mh_full.append(ev['Mh'])
            y_full.append(ev['y']); nu_full.append(ev['nu'])
            if ev['y_rec'] is not None:
                y_rec_full.append(ev['y_rec'])
            if ev['nu_rec'] is not None:
                nu_rec_full.append(ev['nu_rec'])
            if ev['Mh_rec'] is not None:
                Mh_rec_full.append(ev['Mh_rec'])
            if ev['ME_rec'] is not None:
                ME_rec_full.append(ev['ME_rec'])

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
    REC_KW  = dict(histtype="step", color="green", linewidth=1.5, label="Reconstructed")

    def theta_loc(k):
        return "left" if k == 1 else "right"

    # Collect matched rec data per particle
    rec_data = []
    for k in range(npart):
        rp, rt, rphi, rvz = [], [], [], []
        mp, mt, mphi, mvz = [], [], [], []
        for ev in events:
            pp = ev['particles'][k]
            if pp['status'] > 0:
                rp.append(pp['p_rec']); rt.append(pp['theta_rec'])
                rphi.append(pp['phi_rec']); rvz.append(pp['vz_rec'])
                mp.append(pp['p_gen']); mt.append(pp['theta_gen'])
                mphi.append(pp['phi_gen']); mvz.append(pp['vz_gen'])
        rec_data.append({
            'rec_p': np.array(rp), 'rec_theta': np.array(rt),
            'rec_phi': np.array(rphi), 'rec_vz': np.array(rvz),
            'mc_p': np.array(mp), 'mc_theta': np.array(mt),
            'mc_phi': np.array(mphi), 'mc_vz': np.array(mvz),
        })

    n_total = len(events)
    n_full = sum(1 for e in events if e['nrec'] == npart)
    suptitle_base = (f"{reaction}  —  Ebeam={beam_energy} GeV  —  "
                     f"{n_total} events, {n_full} fully reconstructed")

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

        fig.suptitle(suptitle_base, fontsize=13)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ── Page 2: event kinematics ──────────────────────────────
        fig2, axes2 = plt.subplots(2, 4, figsize=(22, 10))

        va = np.array(Q2_all); vf = np.array(Q2_full)
        axes2[0, 0].hist(va, bins=100, **GEN_KW)
        axes2[0, 0].hist(vf, bins=100, **FULL_KW)
        axes2[0, 0].set_xlabel("Q² (GeV²)"); axes2[0, 0].set_title("Q²")
        axes2[0, 0].set_yscale("log")
        axes2[0, 0].legend(fontsize=7)
        annotate(axes2[0, 0], va, loc="right")

        va = np.array(xB_all); vf = np.array(xB_full)
        axes2[0, 1].hist(va, bins=100, **GEN_KW)
        axes2[0, 1].hist(vf, bins=100, **FULL_KW)
        axes2[0, 1].set_xlabel("xB"); axes2[0, 1].set_title("xB")
        axes2[0, 1].set_yscale("log")
        annotate(axes2[0, 1], va, loc="right")

        va = -np.array(t_all); vf = -np.array(t_full)
        axes2[0, 2].hist(va, bins=100, **GEN_KW)
        axes2[0, 2].hist(vf, bins=100, **FULL_KW)
        axes2[0, 2].set_xlabel("-t (GeV²)"); axes2[0, 2].set_title("-t")
        axes2[0, 2].set_yscale("log")
        annotate(axes2[0, 2], va, loc="right")

        va = np.array(W_all); vf = np.array(W_full)
        axes2[0, 3].hist(va, bins=100, **GEN_KW)
        axes2[0, 3].hist(vf, bins=100, **FULL_KW)
        axes2[0, 3].set_xlabel("W (GeV)"); axes2[0, 3].set_title("W")
        annotate(axes2[0, 3], va, loc="left")

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

        va = np.array(y_all); vf = np.array(y_full)
        axes2[1, 1].hist(va, bins=100, range=(0, 1), **GEN_KW)
        axes2[1, 1].hist(vf, bins=100, range=(0, 1), **FULL_KW)
        if y_rec_full:
            axes2[1, 1].hist(np.array(y_rec_full), bins=100, range=(0, 1), **REC_KW)
        axes2[1, 1].set_xlabel("y"); axes2[1, 1].set_title("y")
        axes2[1, 1].set_xlim(0, 1)
        annotate(axes2[1, 1], va, loc="right")

        va = np.array(nu_all); vf = np.array(nu_full)
        axes2[1, 2].hist(va, bins=100, range=(0, 11), **GEN_KW)
        axes2[1, 2].hist(vf, bins=100, range=(0, 11), **FULL_KW)
        if nu_rec_full:
            axes2[1, 2].hist(np.array(nu_rec_full), bins=100, range=(0, 11), **REC_KW)
        axes2[1, 2].set_xlabel("ν (GeV)"); axes2[1, 2].set_title("ν")
        axes2[1, 2].set_xlim(0, 11)
        annotate(axes2[1, 2], va, loc="right")

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

        fig2.suptitle(f"{suptitle_base} ({100*n_full/n_total:.1f}%)", fontsize=13)
        plt.tight_layout()
        pdf.savefig(fig2)
        plt.close(fig2)

        # ── Page 3: 2D plots — theta vs p, theta vs phi ──────────
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

    print(f"Saved plots: {pdf_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot distributions from training .dat files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  %(prog)s phi_train.dat
  %(prog)s phi_train.dat -o my_plots.pdf
""")
    parser.add_argument("dat_file", help="Input .dat file from make_training_data.py")
    parser.add_argument("-o", "--output", default=None,
                        help="Output PDF path (default: <input_base>_plots.pdf)")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    if not os.path.isfile(args.dat_file):
        print(f"Error: '{args.dat_file}' not found")
        sys.exit(1)

    print(f"Reading {args.dat_file} ...")
    info, events = parse_dat_file(args.dat_file)
    npart = len(FINAL_STATE_LABELS.get(info['reaction'], []))
    n_total = len(events)
    n_full = sum(1 for e in events if e['nrec'] == npart)
    print(f"Reaction: {info['reaction']}")
    print(f"Beam: {info['beam_pid']}, Energy: {info['beam_energy']} GeV, "
          f"Target: {info['target_pid']}")
    print(f"Events: {n_total}, fully reconstructed: {n_full} ({100*n_full/n_total:.1f}%)")

    compute_derived(info, events)

    if args.output:
        pdf_path = args.output
    else:
        base = os.path.splitext(args.dat_file)[0]
        pdf_path = f"{base}_plots.pdf"

    make_plots(info, events, pdf_path)
    print("Done.")


if __name__ == "__main__":
    main()
