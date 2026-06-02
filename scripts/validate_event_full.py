#!/usr/bin/env python3
"""Comprehensive validation: FastMC vs GEMC.

Supports both hierarchical (electron-gated) and independent sampling modes.
GEMC acceptance threshold is controlled by --min_status so the same cuts
(matching windows, PID) are applied to both GEMC reference and FastMC.

Produces TWO separate full PDF reports:
    validation_MLP.pdf   and   validation_Grid.pdf

Each PDF includes:
    Page 1: per-particle reconstructed distributions  (GEN / GEMC / FastMC, log Y)
    Page 2: per-particle GEMC vs FastMC (log Y, no GEN)
    Page 3: per-particle resolution  Δ = rec − gen
    Page 4: per-particle FD/CD breakdown
    Page 5: per-particle GEMC vs FastMC (linear)
    Page 6: per-particle ratios FastMC/GEMC
    Pages 7-10: 2-D acceptance efficiency per particle
    Page 11: full-event kinematics Q², xB, W, t, M(K+K-)
    Page 12: M(K+K-)  +  MM(ep→e'p'X)  overlays
    Page 13: summary table

Usage:
    # v5: electron-gated, PID-matched
    python validate_event_full.py phi_val.dat \\
        --mlp_dir models/phi_v5_electron_gated_5sig \\
        --grid models/grid_v5_5sig/grid_eff.npz \\
        --min_status 2 -o plots/phi_v5_electron_gated_5sig

    # v6: electron-gated, no PID
    python validate_event_full.py phi_val.dat \\
        --mlp_dir models/phi_v6_electron_gated_noPID_5sig \\
        --grid models/grid_v6_5sig/grid_eff.npz \\
        -o plots/phi_v6_electron_gated_noPID_5sig

    # v7: independent, no PID
    python validate_event_full.py phi_val.dat \\
        --mlp_dir models/phi_v7_noPID_5sig \\
        --grid models/grid_v7_5sig/grid_eff.npz \\
        --independent -o plots/phi_v7_noPID_5sig
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from validate_fast_mc_v2 import (read_val_data, PDG_TO_SHORT, PDG_MASS,
                                 compute_invariant_mass)
from fast_mc import FastMC
from grid_fastmc import GridFastMC


# ── Color scheme ────────────────────────────────────────────────────────
GEN_KW  = dict(histtype="stepfilled", color="#d3d3d3", edgecolor="#7a7a7a",
               linewidth=1.0, label="Generated", alpha=0.55)
GEMC_KW = dict(histtype="stepfilled", color="#ff7f50", edgecolor="#b22222",
               linewidth=1.2, label="GEMC", alpha=0.65)
FMC_KW  = dict(histtype="step",       color="#1f77b4", edgecolor="#1f77b4",
               linewidth=1.6, label="FastMC")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input")
    p.add_argument("--mlp_dir", default="models/phi_v5")
    p.add_argument("--grid",     default="models/grid_v2/grid_eff.npz")
    p.add_argument("-o", "--output", default="plots/event_fastmc")
    p.add_argument("--max_events", type=int, default=0)
    p.add_argument("--min_status", type=int, default=1,
                   help="Minimum GEMC status to count as accepted "
                        "(1=any match, 2=PID match). Applied equally to "
                        "GEMC reference and FastMC comparison.")
    p.add_argument("--independent", action="store_true",
                   help="Use independent (non-hierarchical) sampling: "
                        "each particle accepted independently, no electron gating.")
    return p.parse_args()


def hierarchical_simulate(electron_model, hadron_models, kin_e, kin_hads):
    """Run hierarchical sampling on a batch of events.

    Returns dict with per-particle:
        accept:     bool array of length N (hadrons are gated by e_acc)
        p_rec, theta_rec, phi_rec, vz_rec: arrays of length count_accepted

    kin_hads: dict {name -> (p, theta, phi, vz)}
    """
    # Electron: full simulate (accept + smear)
    e_full = electron_model.simulate(*kin_e)
    e_acc  = e_full['accepted']

    out = {'e-': {
        'accept':   e_acc,
        'p_rec':    e_full['p_rec'],
        'theta_rec': e_full['theta_rec'],
        'phi_rec':  e_full['phi_rec'],
        'vz_rec':   e_full['vz_rec'],
    }}

    for name, kin in kin_hads.items():
        # accept() returns full-length boolean
        had_acc_indiv = hadron_models[name].accept(*kin)
        # Hierarchical: final accept only if e also accepted
        had_acc = e_acc & had_acc_indiv
        # Smear only the finally-accepted events
        idx = np.where(had_acc)[0]
        if len(idx) > 0:
            p_in   = kin[0][idx]
            th_in  = kin[1][idx]
            phi_in = kin[2][idx]
            vz_in  = kin[3][idx]
            p_r, th_r, phi_r, vz_r = hadron_models[name].smear(p_in, th_in, phi_in, vz_in)
        else:
            p_r = th_r = phi_r = vz_r = np.array([])
        out[name] = {
            'accept':   had_acc,
            'p_rec':    p_r,
            'theta_rec': th_r,
            'phi_rec':  phi_r,
            'vz_rec':   vz_r,
        }
    return out


def independent_simulate(models, kin_all):
    """Run independent sampling: each particle accepted on its own.

    models: dict {name -> FastMC or GridFastMC}
    kin_all: dict {name -> (p, theta, phi, vz)}

    Returns dict with per-particle:
        accept, p_rec, theta_rec, phi_rec, vz_rec
    """
    out = {}
    for name, kin in kin_all.items():
        r = models[name].simulate(*kin)
        out[name] = {
            'accept':    r['accepted'],
            'p_rec':     r['p_rec'],
            'theta_rec': r['theta_rec'],
            'phi_rec':   r['phi_rec'],
            'vz_rec':    r['vz_rec'],
        }
    return out


def make_pdf(pdf_path, label, gemc, fmc, gen_arrays, npart, names, reaction,
             beam_energy, n_events, min_status=1, independent=False):
    """Make a comprehensive PDF for one FastMC implementation (label='MLP' or 'Grid')."""
    gen_p, gen_theta, gen_phi, gen_vz = gen_arrays
    (gemc_status, gemc_p_rec, gemc_theta_rec, gemc_phi_rec, gemc_vz_rec,
     gemc_all_idx, gemc_det) = gemc

    def _masks(k):
        """Return (gmask, gen_mask) for particle index k.

        Hierarchical: hadrons require electron detected in GEMC; gen_mask
            restricted to electron-detected events.
        Independent:  each particle stands alone; gen_mask = all events.
        """
        if independent:
            gm = gemc_status[k] >= min_status
            gn = np.ones(len(gen_p[k]), dtype=bool)
        else:
            if k == 0:
                gm = gemc_status[k] >= min_status
                gn = np.ones(len(gen_p[k]), dtype=bool)
            else:
                gm = (gemc_status[0] >= min_status) & (gemc_status[k] >= min_status)
                gn = gemc_status[0] >= min_status
        return gm, gn

    sampling_label = "independent" if independent else "hierarchical e⁻→hadrons"

    with PdfPages(pdf_path) as pdf:

        # ── Page 1: per-particle distributions (GEN/GEMC/FastMC, log Y) ──
        fig, axes = plt.subplots(npart, 4, figsize=(20, 4*npart))
        if npart == 1: axes = axes[np.newaxis, :]
        for k in range(npart):
            nm = names[k]
            gmask, gen_mask = _masks(k)
            fmask = fmc[nm]['accept']
            p_max = np.percentile(gen_p[k], 99.5) * 1.1
            th_max = min(70, np.percentile(gen_theta[k], 99.5) * 1.2)

            # |p|
            ax = axes[k, 0]
            ax.hist(gen_p[k][gen_mask],     bins=80, range=(0, p_max), **GEN_KW)
            ax.hist(gemc_p_rec[k][gmask],   bins=80, range=(0, p_max), **GEMC_KW)
            ax.hist(fmc[nm]['p_rec'],       bins=80, range=(0, p_max), **FMC_KW)
            ax.set_xlabel("|p| (GeV)"); ax.set_ylabel(nm)
            if k == 0:
                ax.set_title("|p|  GEN / GEMC / FastMC (log)"); ax.legend(fontsize=7)
            # theta
            ax = axes[k, 1]
            ax.hist(gen_theta[k][gen_mask],    bins=80, range=(0, th_max), **GEN_KW)
            ax.hist(gemc_theta_rec[k][gmask],  bins=80, range=(0, th_max), **GEMC_KW)
            ax.hist(fmc[nm]['theta_rec'],      bins=80, range=(0, th_max), **FMC_KW)
            ax.set_xlabel("θ (deg)")
            if k == 0: ax.set_title("θ  GEN / GEMC / FastMC (log)")
            # phi
            ax = axes[k, 2]
            ax.hist(gen_phi[k][gen_mask],    bins=80, range=(-180, 180), **GEN_KW)
            ax.hist(gemc_phi_rec[k][gmask],  bins=80, range=(-180, 180), **GEMC_KW)
            ax.hist(fmc[nm]['phi_rec'],      bins=80, range=(-180, 180), **FMC_KW)
            ax.set_xlabel("φ (deg)")
            if k == 0: ax.set_title("φ  GEN / GEMC / FastMC (log)")
            # vz
            ax = axes[k, 3]
            ax.hist(gen_vz[k][gen_mask],    bins=80, range=(-8, 2), **GEN_KW)
            ax.hist(gemc_vz_rec[k][gmask],  bins=80, range=(-8, 2), **GEMC_KW)
            ax.hist(fmc[nm]['vz_rec'],      bins=80, range=(-8, 2), **FMC_KW)
            ax.set_xlabel("vz (cm)")
            if k == 0: ax.set_title("vz  GEN / GEMC / FastMC (log)")
        for ax in axes.flat:
            ax.set_yscale("log")
        fig.suptitle(f"[{label}]  Per-particle distributions (log Y, GEN included) — "
                     f"{n_events:,} events  ({sampling_label})", fontsize=13)
        plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # ── Page 1.5: per-particle GEMC vs FastMC ONLY (no GEN), log Y ──
        # Same layout as page 1 but cleaner — easier to spot small differences
        # between GEMC and FastMC in the accepted distributions.
        fig, axes = plt.subplots(npart, 4, figsize=(20, 4*npart))
        if npart == 1: axes = axes[np.newaxis, :]
        for k in range(npart):
            nm = names[k]
            gmask, _ = _masks(k)
            p_max  = np.percentile(gen_p[k], 99.5) * 1.1
            th_max = min(70, np.percentile(gen_theta[k], 99.5) * 1.2)
            axes[k, 0].hist(gemc_p_rec[k][gmask],     bins=80, range=(0, p_max), **GEMC_KW)
            axes[k, 0].hist(fmc[nm]['p_rec'],         bins=80, range=(0, p_max), **FMC_KW)
            axes[k, 0].set_xlabel("|p| (GeV)"); axes[k, 0].set_ylabel(nm)
            axes[k, 1].hist(gemc_theta_rec[k][gmask], bins=80, range=(0, th_max), **GEMC_KW)
            axes[k, 1].hist(fmc[nm]['theta_rec'],     bins=80, range=(0, th_max), **FMC_KW)
            axes[k, 1].set_xlabel("θ (deg)")
            axes[k, 2].hist(gemc_phi_rec[k][gmask],   bins=80, range=(-180, 180), **GEMC_KW)
            axes[k, 2].hist(fmc[nm]['phi_rec'],       bins=80, range=(-180, 180), **FMC_KW)
            axes[k, 2].set_xlabel("φ (deg)")
            axes[k, 3].hist(gemc_vz_rec[k][gmask],    bins=80, range=(-8, 2), **GEMC_KW)
            axes[k, 3].hist(fmc[nm]['vz_rec'],        bins=80, range=(-8, 2), **FMC_KW)
            axes[k, 3].set_xlabel("vz (cm)")
            if k == 0:
                axes[k, 0].set_title("|p|  GEMC vs FastMC  (log Y)")
                axes[k, 0].legend(fontsize=7)
                axes[k, 1].set_title("θ  GEMC vs FastMC  (log Y)")
                axes[k, 2].set_title("φ  GEMC vs FastMC  (log Y)")
                axes[k, 3].set_title("vz  GEMC vs FastMC  (log Y)")
        for ax in axes.flat:
            ax.set_yscale("log")
        cond_note_p15 = "" if independent else " — hadrons conditional on e⁻ detected"
        fig.suptitle(f"[{label}]  Per-particle GEMC vs FastMC (log Y, no GEN)"
                     f"{cond_note_p15}", fontsize=13)
        plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # ── Page 1a: per-particle resolution Δ = rec − gen ──
        # Δp, Δθ, Δφ, Δvz histograms.  For each particle, GEMC and FastMC
        # overlaid; for hadrons the sample is restricted to events with
        # electron detected (both sides).
        DELTA_RANGES = {
            'p':   (-0.5, 0.5),
            'th':  (-1.0, 1.0),
            'phi': (-2.0, 2.0),
            'vz':  (-3.0, 3.0),
        }
        DELTA_LABELS = {
            'p':   "Δ|p| (GeV)",
            'th':  "Δθ (deg)",
            'phi': "Δφ (deg)",
            'vz':  "Δvz (cm)",
        }
        def wrap_phi(d):
            return ((d + 180.0) % 360.0) - 180.0

        fig, axes = plt.subplots(npart, 4, figsize=(20, 4*npart))
        if npart == 1: axes = axes[np.newaxis, :]
        for k in range(npart):
            nm = names[k]
            gmask, _ = _masks(k)

            # GEMC Δs
            g_dp   = gemc_p_rec[k][gmask]     - gen_p[k][gmask]
            g_dt   = gemc_theta_rec[k][gmask] - gen_theta[k][gmask]
            g_dphi = wrap_phi(gemc_phi_rec[k][gmask] - gen_phi[k][gmask])
            g_dvz  = gemc_vz_rec[k][gmask]    - gen_vz[k][gmask]

            # FastMC Δs (rec already only for accepted)
            fa = fmc[nm]['accept']
            f_dp   = fmc[nm]['p_rec']     - gen_p[k][fa]
            f_dt   = fmc[nm]['theta_rec'] - gen_theta[k][fa]
            f_dphi = wrap_phi(fmc[nm]['phi_rec'] - gen_phi[k][fa])
            f_dvz  = fmc[nm]['vz_rec']    - gen_vz[k][fa]

            for col, (key, gd, fd) in enumerate([
                ('p',   g_dp,   f_dp),
                ('th',  g_dt,   f_dt),
                ('phi', g_dphi, f_dphi),
                ('vz',  g_dvz,  f_dvz),
            ]):
                ax = axes[k, col]
                rng = DELTA_RANGES[key]
                ax.hist(gd, bins=80, range=rng, **GEMC_KW)
                ax.hist(fd, bins=80, range=rng, **FMC_KW)
                ax.set_xlabel(DELTA_LABELS[key])
                if col == 0: ax.set_ylabel(nm)
                if k == 0 and col == 0:
                    ax.set_title("Δ|p|  resolution"); ax.legend(fontsize=7)
                elif k == 0:
                    ax.set_title(f"{DELTA_LABELS[key].split()[0]}  resolution")

                # Add σ annotation
                if len(gd) > 10 and len(fd) > 10:
                    sg = np.std(gd[np.isfinite(gd)])
                    sf = np.std(fd[np.isfinite(fd)])
                    ax.text(0.02, 0.95, f"σ_GEMC = {sg:.3f}\nσ_FMC  = {sf:.3f}",
                            transform=ax.transAxes, fontsize=8, va="top", ha="left",
                            family="monospace",
                            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow",
                                     ec="gray", alpha=0.85))
        cond_note_res = "" if independent else ", hadrons conditional on e⁻"
        fig.suptitle(f"[{label}]  Per-particle resolution  Δ = rec − gen  "
                     f"(GEMC vs FastMC{cond_note_res})", fontsize=13)
        plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # ── Page 1b: per-particle FD/CD breakdown ──
        # GEMC FD/CD from .dat det column (already loaded as gemc_det if exists);
        # FastMC FD vs CD by theta_rec < 35 deg proxy.
        FD_CD_TH = 35.0
        GEMC_FD = dict(histtype="step", color="#b22222", linewidth=1.4, label="GEMC FD")
        GEMC_CD = dict(histtype="step", color="#8b0000", linewidth=1.4,
                       linestyle="--", label="GEMC CD")
        FMC_FD  = dict(histtype="step", color="#1f77b4", linewidth=1.4, label="FastMC FD")
        FMC_CD  = dict(histtype="step", color="#0a3d62", linewidth=1.4,
                       linestyle="--", label="FastMC CD")

        fig, axes = plt.subplots(npart, 4, figsize=(20, 4*npart))
        if npart == 1: axes = axes[np.newaxis, :]
        for k in range(npart):
            nm = names[k]
            gmask, _ = _masks(k)
            # GEMC FD/CD masks (use the det column written by make_training_data)
            if gemc_det is not None:
                gemc_fd_mask = gmask & (gemc_det[k] == 1)
                gemc_cd_mask = gmask & (gemc_det[k] == 2)
            else:
                gemc_fd_mask = gmask & (gemc_theta_rec[k] <  FD_CD_TH)
                gemc_cd_mask = gmask & (gemc_theta_rec[k] >= FD_CD_TH)
            # FastMC FD/CD by theta_rec proxy (no det info in FastMC sampling)
            fmc_t_rec = fmc[nm]['theta_rec']
            fmc_fd = fmc_t_rec <  FD_CD_TH
            fmc_cd = fmc_t_rec >= FD_CD_TH

            p_max  = np.percentile(gen_p[k], 99.5) * 1.1
            th_max = min(70, np.percentile(gen_theta[k], 99.5) * 1.2)

            # |p|
            ax = axes[k, 0]
            ax.hist(gemc_p_rec[k][gemc_fd_mask], bins=80, range=(0, p_max), **GEMC_FD)
            ax.hist(gemc_p_rec[k][gemc_cd_mask], bins=80, range=(0, p_max), **GEMC_CD)
            ax.hist(fmc[nm]['p_rec'][fmc_fd],    bins=80, range=(0, p_max), **FMC_FD)
            ax.hist(fmc[nm]['p_rec'][fmc_cd],    bins=80, range=(0, p_max), **FMC_CD)
            ax.set_xlabel("|p| (GeV)"); ax.set_ylabel(nm)
            if k == 0:
                ax.set_title("|p|  FD vs CD breakdown"); ax.legend(fontsize=6)
            # theta
            ax = axes[k, 1]
            ax.hist(gemc_theta_rec[k][gemc_fd_mask], bins=80, range=(0, th_max), **GEMC_FD)
            ax.hist(gemc_theta_rec[k][gemc_cd_mask], bins=80, range=(0, th_max), **GEMC_CD)
            ax.hist(fmc[nm]['theta_rec'][fmc_fd],    bins=80, range=(0, th_max), **FMC_FD)
            ax.hist(fmc[nm]['theta_rec'][fmc_cd],    bins=80, range=(0, th_max), **FMC_CD)
            ax.axvline(FD_CD_TH, color="gray", ls=":", lw=0.7)
            ax.set_xlabel("θ (deg)")
            if k == 0: ax.set_title("θ  FD vs CD breakdown")
            # phi
            ax = axes[k, 2]
            ax.hist(gemc_phi_rec[k][gemc_fd_mask], bins=80, range=(-180, 180), **GEMC_FD)
            ax.hist(gemc_phi_rec[k][gemc_cd_mask], bins=80, range=(-180, 180), **GEMC_CD)
            ax.hist(fmc[nm]['phi_rec'][fmc_fd],    bins=80, range=(-180, 180), **FMC_FD)
            ax.hist(fmc[nm]['phi_rec'][fmc_cd],    bins=80, range=(-180, 180), **FMC_CD)
            ax.set_xlabel("φ (deg)")
            if k == 0: ax.set_title("φ  FD vs CD breakdown")
            # vz
            ax = axes[k, 3]
            ax.hist(gemc_vz_rec[k][gemc_fd_mask], bins=80, range=(-8, 2), **GEMC_FD)
            ax.hist(gemc_vz_rec[k][gemc_cd_mask], bins=80, range=(-8, 2), **GEMC_CD)
            ax.hist(fmc[nm]['vz_rec'][fmc_fd],    bins=80, range=(-8, 2), **FMC_FD)
            ax.hist(fmc[nm]['vz_rec'][fmc_cd],    bins=80, range=(-8, 2), **FMC_CD)
            ax.set_xlabel("vz (cm)")
            if k == 0: ax.set_title("vz  FD vs CD breakdown")
        fig.suptitle(f"[{label}]  Per-particle FD vs CD breakdown  ({reaction})",
                     fontsize=13)
        plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # ── Page 2: per-particle (GEMC vs FastMC linear, clean) ──
        fig, axes = plt.subplots(npart, 4, figsize=(20, 4*npart))
        if npart == 1: axes = axes[np.newaxis, :]
        for k in range(npart):
            nm = names[k]
            gmask, _ = _masks(k)
            p_max  = np.percentile(gen_p[k], 99.5) * 1.1
            th_max = min(70, np.percentile(gen_theta[k], 99.5) * 1.2)
            axes[k, 0].hist(gemc_p_rec[k][gmask],     bins=80, range=(0, p_max), **GEMC_KW)
            axes[k, 0].hist(fmc[nm]['p_rec'],          bins=80, range=(0, p_max), **FMC_KW)
            axes[k, 0].set_xlabel("|p| (GeV)"); axes[k, 0].set_ylabel(nm)
            axes[k, 1].hist(gemc_theta_rec[k][gmask],  bins=80, range=(0, th_max), **GEMC_KW)
            axes[k, 1].hist(fmc[nm]['theta_rec'],      bins=80, range=(0, th_max), **FMC_KW)
            axes[k, 1].set_xlabel("θ (deg)")
            axes[k, 2].hist(gemc_phi_rec[k][gmask],    bins=80, range=(-180, 180), **GEMC_KW)
            axes[k, 2].hist(fmc[nm]['phi_rec'],        bins=80, range=(-180, 180), **FMC_KW)
            axes[k, 2].set_xlabel("φ (deg)")
            axes[k, 3].hist(gemc_vz_rec[k][gmask],     bins=80, range=(-8, 2), **GEMC_KW)
            axes[k, 3].hist(fmc[nm]['vz_rec'],         bins=80, range=(-8, 2), **FMC_KW)
            axes[k, 3].set_xlabel("vz (cm)")
            if k == 0:
                axes[k, 0].set_title("|p|"); axes[k, 1].set_title("θ")
                axes[k, 2].set_title("φ"); axes[k, 3].set_title("vz")
                axes[k, 0].legend(fontsize=7)
        fig.suptitle(f"[{label}]  Per-particle reconstructed (GEMC vs FastMC, linear)",
                     fontsize=13)
        plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # ── Page 3: per-particle ratios FastMC/GEMC ──
        fig, axes = plt.subplots(npart, 4, figsize=(20, 4*npart))
        if npart == 1: axes = axes[np.newaxis, :]
        for k in range(npart):
            nm = names[k]
            gmask, _ = _masks(k)
            p_max  = np.percentile(gen_p[k], 99.5) * 1.1
            th_max = min(70, np.percentile(gen_theta[k], 99.5) * 1.2)
            for col, (gemc_arr, fmc_arr, lab, rng) in enumerate([
                (gemc_p_rec[k][gmask],     fmc[nm]['p_rec'],     "|p| (GeV)", (0, p_max)),
                (gemc_theta_rec[k][gmask], fmc[nm]['theta_rec'], "θ (deg)",   (0, th_max)),
                (gemc_phi_rec[k][gmask],   fmc[nm]['phi_rec'],   "φ (deg)",   (-180, 180)),
                (gemc_vz_rec[k][gmask],    fmc[nm]['vz_rec'],    "vz (cm)",   (-8, 2)),
            ]):
                h_g, edges = np.histogram(gemc_arr, bins=50, range=rng)
                h_f, _     = np.histogram(fmc_arr,  bins=50, range=rng)
                centers = 0.5 * (edges[:-1] + edges[1:])
                with np.errstate(divide="ignore", invalid="ignore"):
                    r = np.where(h_g > 0, h_f / h_g, np.nan)
                axes[k, col].plot(centers, r, "k.", markersize=4)
                axes[k, col].axhline(1, color="red", lw=1)
                axes[k, col].set_xlabel(lab); axes[k, col].set_ylim(0.5, 1.5)
                if col == 0: axes[k, col].set_ylabel(f"{nm}  FastMC/GEMC")
                if k == 0:   axes[k, col].set_title(f"Ratio {lab}")
        fig.suptitle(f"[{label}]  FastMC/GEMC ratios", fontsize=13)
        plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # ── Pages 4..7: 2-D acceptance efficiency per particle ──
        # ε = (# accepted in 4-D bin) / (# generated in 4-D bin)
        # For hadrons, BOTH numerator and denominator are restricted to
        # events with detected electron (i.e., the analyzable sample).
        from matplotlib.colors import LogNorm  # local import
        for k in range(npart):
            nm = names[k]
            accepted_mask_gemc, gen_mask = _masks(k)
            accepted_mask_fmc = fmc[nm]['accept']

            p_max  = np.percentile(gen_p[k][gen_mask], 99.5) * 1.05 if gen_mask.any() else 8.0
            th_max = min(70.0, np.percentile(gen_theta[k][gen_mask], 99.5) * 1.15 if gen_mask.any() else 70.0)

            kinds = [
                ("tp",  gen_p[k],    gen_theta[k], (0, p_max),    (0, th_max),
                 "|p| (GeV)", "θ (deg)"),
                ("tph", gen_phi[k],  gen_theta[k], (-180, 180),   (0, th_max),
                 "φ (deg)",   "θ (deg)"),
                ("pph", gen_p[k],    gen_phi[k],   (0, p_max),    (-180, 180),
                 "|p| (GeV)", "φ (deg)"),
            ]

            fig, axes = plt.subplots(3, 3, figsize=(18, 14))
            for col, (kind, xv, yv, xr, yr, xlab, ylab) in enumerate(kinds):
                bins = 60
                # Apply gen_mask (full for e-, electron-detected for hadrons)
                xv_m = xv[gen_mask];  yv_m = yv[gen_mask]
                h_den, xe, ye = np.histogram2d(xv_m, yv_m, bins=bins, range=[xr, yr])

                # Numerator: GEMC accepted (in the gen_mask sample)
                m_g = accepted_mask_gemc & gen_mask
                h_g, _, _ = np.histogram2d(xv[m_g], yv[m_g], bins=bins, range=[xr, yr])
                # Numerator: FastMC accepted (already gated for hadrons in our
                # hierarchical sampling)
                h_f, _, _ = np.histogram2d(xv[accepted_mask_fmc], yv[accepted_mask_fmc],
                                            bins=bins, range=[xr, yr])
                with np.errstate(divide="ignore", invalid="ignore"):
                    eff_g = np.where(h_den > 0, h_g / h_den, np.nan)
                    eff_f = np.where(h_den > 0, h_f / h_den, np.nan)
                    ratio = np.where(eff_g > 0, eff_f / eff_g, np.nan)

                # Color scale: percentile-based to avoid edge crushing
                eff_max = np.nanpercentile(np.concatenate(
                    [eff_g[~np.isnan(eff_g)], eff_f[~np.isnan(eff_f)]]), 99) \
                    if (np.isfinite(eff_g).any() or np.isfinite(eff_f).any()) else 1.0
                eff_max = max(eff_max, 1e-3)

                im = axes[0, col].imshow(eff_g.T, origin="lower", aspect="auto",
                                          extent=[xe[0], xe[-1], ye[0], ye[-1]],
                                          cmap="viridis", vmin=0, vmax=eff_max)
                axes[0, col].set_xlabel(xlab); axes[0, col].set_ylabel(ylab)
                axes[0, col].set_title(f"GEMC eff  {ylab} vs {xlab}")
                plt.colorbar(im, ax=axes[0, col], pad=0.02)

                im = axes[1, col].imshow(eff_f.T, origin="lower", aspect="auto",
                                          extent=[xe[0], xe[-1], ye[0], ye[-1]],
                                          cmap="viridis", vmin=0, vmax=eff_max)
                axes[1, col].set_xlabel(xlab); axes[1, col].set_ylabel(ylab)
                axes[1, col].set_title(f"FastMC eff  {ylab} vs {xlab}")
                plt.colorbar(im, ax=axes[1, col], pad=0.02)

                im = axes[2, col].imshow(ratio.T, origin="lower", aspect="auto",
                                          extent=[xe[0], xe[-1], ye[0], ye[-1]],
                                          cmap="RdBu_r", vmin=0.5, vmax=1.5)
                axes[2, col].set_xlabel(xlab); axes[2, col].set_ylabel(ylab)
                axes[2, col].set_title(f"FastMC/GEMC eff  {ylab} vs {xlab}")
                plt.colorbar(im, ax=axes[2, col], pad=0.02)

            if independent:
                cond_note = ""
            else:
                cond_note = "" if k == 0 else " — events conditional on e⁻ detected"
            fig.suptitle(f"[{label}]  2-D acceptance efficiency — {nm}{cond_note}\n"
                          "rows: GEMC ε,  FastMC ε,  ratio (denominator = generated)",
                          fontsize=13)
            plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # ── Page 8: full-event kinematics ──
        fmc_all_mask = (fmc['e-']['accept'] & fmc['p']['accept'] &
                        fmc['K+']['accept'] & fmc['K-']['accept'])
        fmc_all_idx = np.where(fmc_all_mask)[0]
        Q2 = np.array([ev['Q2'] for ev in events])
        xB = np.array([ev['xB'] for ev in events])
        W  = np.array([ev['W']  for ev in events])
        t  = np.array([ev['t']  for ev in events])
        Mh = np.array([ev['Mh'] for ev in events])

        fig, axes = plt.subplots(2, 5, figsize=(25, 10))
        for col, (var, vlab, rng, log_y) in enumerate([
            (Q2, "Q² (GeV²)",     (0, 8),   True),
            (xB, "xB",            (0, 0.8), True),
            (W,  "W (GeV)",       (1, 5),   True),
            (-t, "-t (GeV²)",     (0, 5),   True),
            (Mh, "M(K+K-) (GeV)", (0.98, 1.06), False),
        ]):
            axes[0, col].hist(var[gemc_all_idx], bins=80, range=rng, **GEMC_KW)
            axes[0, col].hist(var[fmc_all_idx],  bins=80, range=rng, **FMC_KW)
            axes[0, col].set_xlabel(vlab); axes[0, col].set_title(vlab)
            if log_y: axes[0, col].set_yscale("log")
            if col == 0: axes[0, col].legend(fontsize=7)
            h_g, ed = np.histogram(var[gemc_all_idx], bins=50, range=rng)
            h_f, _  = np.histogram(var[fmc_all_idx],  bins=50, range=rng)
            cen = 0.5 * (ed[:-1] + ed[1:])
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.where(h_g > 0, h_f / h_g, np.nan)
            axes[1, col].plot(cen, r, "k.", markersize=4); axes[1, col].axhline(1, color="red", lw=1)
            axes[1, col].set_xlabel(vlab); axes[1, col].set_ylim(0.5, 1.5)
            axes[1, col].set_ylabel("FastMC/GEMC")
        fig.suptitle(f"[{label}]  Full-event kinematics (all-4 accepted)\n"
                     f"GEMC: {len(gemc_all_idx):,}   FastMC: {len(fmc_all_idx):,}",
                     fontsize=13)
        plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # ── Page 5: M(K+K-) + MM ──
        m_e = PDG_MASS[11]; m_pr = PDG_MASS[2212]; m_k = PDG_MASS[321]
        p_beam = np.sqrt(beam_energy**2 - m_e**2)

        def MM_arr(p_e, t_e, ph_e, p_p, t_p, ph_p):
            te = np.radians(t_e); pe = np.radians(ph_e)
            tp = np.radians(t_p); pp = np.radians(ph_p)
            Ee = np.sqrt(p_e**2 + m_e**2);  Ep = np.sqrt(p_p**2 + m_pr**2)
            return np.sqrt(np.maximum(0,
                (beam_energy + m_pr - Ee - Ep)**2
                - (-p_e*np.sin(te)*np.cos(pe) - p_p*np.sin(tp)*np.cos(pp))**2
                - (-p_e*np.sin(te)*np.sin(pe) - p_p*np.sin(tp)*np.sin(pp))**2
                - (p_beam - p_e*np.cos(te) - p_p*np.cos(tp))**2))

        # GEMC M(K+K-) and MM on all-4 events
        ig = gemc_all_idx
        gemc_Mh = compute_invariant_mass(
            gemc_p_rec[3][ig], gemc_theta_rec[3][ig], gemc_phi_rec[3][ig], m_k,
            gemc_p_rec[2][ig], gemc_theta_rec[2][ig], gemc_phi_rec[2][ig], m_k)
        gemc_MM = MM_arr(
            gemc_p_rec[0][ig], gemc_theta_rec[0][ig], gemc_phi_rec[0][ig],
            gemc_p_rec[1][ig], gemc_theta_rec[1][ig], gemc_phi_rec[1][ig])

        # FastMC: need rec arrays aligned to all-4-accepted events.
        # For each particle, fmc[nm]['p_rec'] etc. are only for accepted events.
        # We need to index into them: position in 'accept' array = position in p_rec.
        # Map fmc_all_idx → positions within each particle's accept-True list.
        def rec_at(nm, all_idx):
            mask = fmc[nm]['accept']
            # rank = cumulative count of True up to each position
            rank = np.cumsum(mask) - 1  # position within accepted list
            return (fmc[nm]['p_rec']   [rank[all_idx]],
                    fmc[nm]['theta_rec'][rank[all_idx]],
                    fmc[nm]['phi_rec'] [rank[all_idx]],
                    fmc[nm]['vz_rec']  [rank[all_idx]])

        f_pe, f_te, f_phie, f_vze = rec_at('e-', fmc_all_idx)
        f_pp, f_tp, f_phip, f_vzp = rec_at('p',  fmc_all_idx)
        f_pkp, f_tkp, f_phikp, _  = rec_at('K+', fmc_all_idx)
        f_pkm, f_tkm, f_phikm, _  = rec_at('K-', fmc_all_idx)

        fmc_Mh = compute_invariant_mass(
            f_pkp, f_tkp, f_phikp, m_k, f_pkm, f_tkm, f_phikm, m_k)
        fmc_MM = MM_arr(f_pe, f_te, f_phie, f_pp, f_tp, f_phip)

        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        m_range  = (0.98, 1.10)
        mm_range = (0.5, 1.5)
        axes[0, 0].hist(gemc_Mh, bins=80, range=m_range, **GEMC_KW)
        axes[0, 0].set_title(f"GEMC M(K+K-)  N={len(gemc_Mh):,}"); axes[0, 0].set_xlabel("GeV")
        axes[0, 1].hist(fmc_Mh,  bins=80, range=m_range, **FMC_KW)
        axes[0, 1].set_title(f"FastMC M(K+K-)  N={len(fmc_Mh):,}"); axes[0, 1].set_xlabel("GeV")
        axes[0, 2].hist(gemc_Mh, bins=80, range=m_range, **GEMC_KW)
        axes[0, 2].hist(fmc_Mh,  bins=80, range=m_range, **FMC_KW)
        axes[0, 2].set_yscale("log"); axes[0, 2].set_title("M(K+K-) overlay (log)")
        axes[0, 2].legend(fontsize=8); axes[0, 2].set_xlabel("GeV")
        axes[1, 0].hist(gemc_MM, bins=80, range=mm_range, **GEMC_KW)
        axes[1, 0].axvline(1.019, color="red", ls="--")
        axes[1, 0].set_title(f"GEMC MM(ep→e'p'X)  N={len(gemc_MM):,}"); axes[1, 0].set_xlabel("GeV")
        axes[1, 1].hist(fmc_MM,  bins=80, range=mm_range, **FMC_KW)
        axes[1, 1].axvline(1.019, color="red", ls="--")
        axes[1, 1].set_title(f"FastMC MM(ep→e'p'X)  N={len(fmc_MM):,}"); axes[1, 1].set_xlabel("GeV")
        axes[1, 2].hist(gemc_MM, bins=80, range=mm_range, **GEMC_KW)
        axes[1, 2].hist(fmc_MM,  bins=80, range=mm_range, **FMC_KW)
        axes[1, 2].axvline(1.019, color="red", ls="--")
        axes[1, 2].set_yscale("log"); axes[1, 2].set_title("MM overlay (log)")
        axes[1, 2].legend(fontsize=8); axes[1, 2].set_xlabel("GeV")
        fig.suptitle(f"[{label}]  M(K+K-) top, MM(ep→e'p'X) bottom", fontsize=13)
        plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # ── Page 9: summary table with percentage comparison ──
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis("off")
        rows = [["Quantity", "GEMC", f"FastMC ({label})", "FastMC / GEMC"]]

        def add_row(qname, n_g, n_f):
            pct = n_f / max(1, n_g) * 100
            rows.append([qname, f"{n_g:,}", f"{n_f:,}", f"{pct:6.2f}%"])

        n_g_e = int((gemc_status[0]>=min_status).sum())
        n_f_e = int(fmc['e-']['accept'].sum())
        add_row("e- detected", n_g_e, n_f_e)

        for nm in ["p", "K+", "K-"]:
            idx = names.index(nm)
            if independent:
                n_g = int((gemc_status[idx]>=min_status).sum())
                row_label = f"{nm} detected"
            else:
                n_g = int(((gemc_status[0]>=min_status) & (gemc_status[idx]>=min_status)).sum())
                row_label = f"e- AND {nm}"
            n_f = int(fmc[nm]['accept'].sum())
            add_row(row_label, n_g, n_f)

        rows.append(["", "", "", ""])
        n_g_all = len(gemc_all_idx)
        n_f_all = len(fmc_all_idx)
        add_row("ALL-4 (joint)", n_g_all, n_f_all)

        # Multiplicity: >=N particles detected per event
        gemc_ndet = np.zeros(n_events, dtype=int)
        fmc_ndet  = np.zeros(n_events, dtype=int)
        for k in range(npart):
            gemc_ndet += (gemc_status[k] >= min_status).astype(int)
            fmc_ndet  += fmc[names[k]]['accept'].astype(int)
        rows.append(["", "", "", ""])
        for m in range(1, npart + 1):
            n_g = int((gemc_ndet >= m).sum())
            n_f = int((fmc_ndet  >= m).sum())
            add_row(f">={m} particles", n_g, n_f)

        rows.append(["", "", "", ""])
        rows.append(["Reaction", reaction, "", ""])
        rows.append(["Beam energy", f"{beam_energy} GeV", "", ""])
        rows.append(["Total events", f"{n_events:,}", "", ""])

        table = ax.table(cellText=rows, loc="center", cellLoc="left",
                          colWidths=[0.28, 0.24, 0.24, 0.20])
        table.auto_set_font_size(False); table.set_fontsize(12); table.scale(1, 1.6)
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor("lightgray")
            if r == 0:
                cell.set_facecolor("lightyellow"); cell.set_text_props(weight="bold")
            if r > 0 and rows[r] and rows[r][0] == "ALL-4 (joint)":
                cell.set_facecolor("lightgreen")
                cell.set_text_props(weight="bold")
            if r > 0 and rows[r] and rows[r][0].startswith(">="):
                cell.set_facecolor("#e0f0ff")
        fig.suptitle(f"Summary [{label}]  vs GEMC  —  {sampling_label} sampling",
                     fontsize=14)
        plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    print(f"Reading {args.input} ...")
    header, events = read_val_data(args.input, max_events=args.max_events)
    N = len(events)
    npart = len(events[0]['particles'])
    names = [PDG_TO_SHORT.get(events[0]['particles'][k]['pid'], '?') for k in range(npart)]
    reaction = header.get('reaction', 'epK+K-')
    beam_energy = float(header.get('beam_energy', '10.6'))
    print(f"  N events: {N:,}    particles: {names}")

    # Build arrays
    gen_p     = [np.array([ev['particles'][k]['p_gen']     for ev in events]) for k in range(npart)]
    gen_theta = [np.array([ev['particles'][k]['theta_gen'] for ev in events]) for k in range(npart)]
    gen_phi   = [np.array([ev['particles'][k]['phi_gen']   for ev in events]) for k in range(npart)]
    gen_vz    = [np.array([ev['particles'][k]['vz_gen']    for ev in events]) for k in range(npart)]
    gemc_status = [np.array([ev['particles'][k]['status']  for ev in events]) for k in range(npart)]
    gemc_p_rec = [np.array([ev['particles'][k]['p_rec']    for ev in events]) for k in range(npart)]
    gemc_theta_rec = [np.array([ev['particles'][k]['theta_rec'] for ev in events]) for k in range(npart)]
    gemc_phi_rec = [np.array([ev['particles'][k]['phi_rec'] for ev in events]) for k in range(npart)]
    gemc_vz_rec = [np.array([ev['particles'][k]['vz_rec']  for ev in events]) for k in range(npart)]

    # det column (0=none, 1=FD, 2=CD) if available
    try:
        gemc_det = [np.array([ev['particles'][k].get('det', 0) for ev in events])
                    for k in range(npart)]
    except Exception:
        gemc_det = None

    min_status = args.min_status
    mode = "independent" if args.independent else "hierarchical"
    print(f"  min_status={min_status}  mode={mode}")

    gemc_all_mask = np.ones(N, dtype=bool)
    for k in range(npart):
        gemc_all_mask &= (gemc_status[k] >= min_status)
    gemc_all_idx  = np.where(gemc_all_mask)[0]

    gen_arrays  = (gen_p, gen_theta, gen_phi, gen_vz)
    gemc_bundle = (gemc_status, gemc_p_rec, gemc_theta_rec, gemc_phi_rec,
                   gemc_vz_rec, gemc_all_idx, gemc_det)

    # kin tuples by name
    kin_all = {names[k]: (gen_p[k], gen_theta[k], gen_phi[k], gen_vz[k])
               for k in range(npart)}
    kin_e    = kin_all['e-']
    kin_hads = {nm: kin_all[nm] for nm in kin_all if nm != 'e-'}

    # ── MLP ──
    print(f"\n=== MLP ({mode}) ===")
    mlp_models = {nm: FastMC(model_file=os.path.join(args.mlp_dir, f"{nm}.pt"))
                  for nm in names}
    if args.independent:
        fmc_mlp = independent_simulate(mlp_models, kin_all)
    else:
        fmc_mlp = hierarchical_simulate(mlp_models['e-'],
                    {nm: mlp_models[nm] for nm in kin_hads}, kin_e, kin_hads)
    make_pdf(os.path.join(args.output, "validation_MLP.pdf"), "MLP",
             gemc_bundle, fmc_mlp, gen_arrays, npart, names, reaction,
             beam_energy, N, min_status=min_status, independent=args.independent)

    # ── Grid ──
    print(f"\n=== Grid ({mode}) ===")
    grid_models = {nm: GridFastMC(args.grid, nm) for nm in names}
    if args.independent:
        fmc_grid = independent_simulate(grid_models, kin_all)
    else:
        fmc_grid = hierarchical_simulate(grid_models['e-'],
                    {nm: grid_models[nm] for nm in kin_hads}, kin_e, kin_hads)
    make_pdf(os.path.join(args.output, "validation_Grid.pdf"), "Grid",
             gemc_bundle, fmc_grid, gen_arrays, npart, names, reaction,
             beam_energy, N, min_status=min_status, independent=args.independent)

    print("\nDone.")
