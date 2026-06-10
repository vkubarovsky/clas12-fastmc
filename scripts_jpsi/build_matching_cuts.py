#!/usr/bin/env python3
"""Build momentum-dependent matching cuts from a GEMC .dat file.

Automated procedure:
  1. Read .dat, select detected particles (status>0)
  2. For each particle × detector (FD/CD):
     - Compute residuals ΔP, Δθ, Δφ
     - Fit Gaussians in slices → σ(ΔP) vs P, σ(Δθ) vs θ, σ(Δφ) vs φ
     - Fit smooth polynomial parametrizations
  3. Save to JSON (for use by matching_cuts.py / make_training_data.py)
  4. Produce validation PDF

Usage:
    python build_matching_cuts.py phi_val.dat -o matching_cuts_phi.json
    python build_matching_cuts.py phi_val.dat -o matching_cuts_phi.json --n_sigma 4

For outbending:
    python build_matching_cuts.py phi_val_outb.dat -o matching_cuts_phi_outb.json
"""

import argparse
import json
import os
import sys
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

PDG_TO_SHORT = {
    11: 'e-', -11: 'e+', 2212: 'p',
    211: 'pi+', -211: 'pi-', 321: 'K+', -321: 'K-',
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help=".dat validation/training file")
    p.add_argument("-o", "--output", default="matching_cuts.json",
                   help="Output JSON file")
    p.add_argument("--pdf", default=None,
                   help="Validation PDF (default: same name as JSON with .pdf)")
    p.add_argument("--n_sigma", type=float, default=4.0,
                   help="Number of sigma for cuts (default: 4)")
    p.add_argument("--nbins", type=int, default=40,
                   help="Number of slices for Gaussian fits")
    p.add_argument("--poly_deg_p", type=int, default=2,
                   help="Polynomial degree for σ(ΔP) vs P")
    p.add_argument("--poly_deg_theta", type=int, default=2,
                   help="Polynomial degree for σ(Δθ) vs θ")
    p.add_argument("--max_events", type=int, default=0)
    return p.parse_args()


# ── Fast .dat reader ──────────────────────────────────────────────

def read_particles_fast(filename, max_events=0):
    """Read .dat file, return per-particle-index arrays."""
    particles = {}
    npart = None
    n_events = 0
    k = -1

    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#!'):
                continue
            if not line.strip():
                continue
            if line[0] == ' ':
                k += 1
                parts = line.split()
                if k not in particles:
                    particles[k] = {col: [] for col in
                                    ['status', 'pid', 'p_gen', 'theta_gen', 'phi_gen',
                                     'vz_gen', 'p_rec', 'theta_rec', 'phi_rec',
                                     'vz_rec', 'det']}
                d = particles[k]
                status = int(parts[0])
                d['status'].append(status)
                d['pid'].append(int(parts[1]))
                # Detect format: new compact has det (int) as 3rd col
                if '.' not in parts[2]:
                    # New compact: status pid det p_gen ... [p_rec ...]
                    d['det'].append(int(parts[2]))
                    d['p_gen'].append(float(parts[3]))
                    d['theta_gen'].append(float(parts[4]))
                    d['phi_gen'].append(float(parts[5]))
                    d['vz_gen'].append(float(parts[6]))
                    if status > 0 and len(parts) >= 11:
                        d['p_rec'].append(float(parts[7]))
                        d['theta_rec'].append(float(parts[8]))
                        d['phi_rec'].append(float(parts[9]))
                        d['vz_rec'].append(float(parts[10]))
                    else:
                        d['p_rec'].append(-999.0)
                        d['theta_rec'].append(-999.0)
                        d['phi_rec'].append(-999.0)
                        d['vz_rec'].append(-999.0)
                else:
                    # Old format: status pid p_gen ... p_rec ... [det]
                    d['p_gen'].append(float(parts[2]))
                    d['theta_gen'].append(float(parts[3]))
                    d['phi_gen'].append(float(parts[4]))
                    d['vz_gen'].append(float(parts[5]))
                    d['p_rec'].append(float(parts[6]))
                    d['theta_rec'].append(float(parts[7]))
                    d['phi_rec'].append(float(parts[8]))
                    d['vz_rec'].append(float(parts[9]))
                    d['det'].append(int(parts[10]) if len(parts) >= 11 else
                                    (1 if status >= 1 else 0))
            else:
                if npart is None and k >= 0:
                    npart = k + 1
                k = -1
                n_events += 1
                if max_events > 0 and n_events >= max_events:
                    break

    for k in particles:
        for key in particles[k]:
            particles[k][key] = np.array(particles[k][key],
                                         dtype=int if key in ('status', 'pid', 'det')
                                         else float)

    names = []
    for k in sorted(particles.keys()):
        pid = int(particles[k]['pid'][0])
        names.append(PDG_TO_SHORT.get(pid, f'pid{pid}'))

    return particles, names, n_events


# ── Gaussian slice fitting ────────────────────────────────────────

def gauss(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma)**2)


def fit_slices(x, dx, x_range, nbins):
    """Fit Gaussian in slices of x. Returns centers, mu, sigma, sigma_err."""
    edges = np.linspace(x_range[0], x_range[1], nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mus = np.full(nbins, np.nan)
    sigmas = np.full(nbins, np.nan)
    sigma_errs = np.full(nbins, np.nan)

    for i in range(nbins):
        mask = (x >= edges[i]) & (x < edges[i + 1])
        dx_slice = dx[mask]
        if len(dx_slice) < 30:
            continue

        med = np.median(dx_slice)
        iqr = np.percentile(dx_slice, 75) - np.percentile(dx_slice, 25)
        sig_est = iqr / 1.349
        if sig_est < 1e-8:
            sig_est = np.std(dx_slice)
        if sig_est < 1e-8:
            continue

        lo, hi = med - 4 * sig_est, med + 4 * sig_est
        sel = dx_slice[(dx_slice >= lo) & (dx_slice <= hi)]
        if len(sel) < 20:
            continue

        h, be = np.histogram(sel, bins=60, range=(lo, hi))
        bc = 0.5 * (be[:-1] + be[1:])
        if h.max() < 5:
            continue

        try:
            p0 = [h.max(), med, sig_est]
            popt, pcov = curve_fit(gauss, bc, h, p0=p0,
                                   bounds=([0, lo, 1e-6], [h.max()*3, hi, (hi-lo)]))
            perr = np.sqrt(np.diag(pcov))
            if popt[2] > 0 and perr[2] < popt[2]:
                mus[i] = popt[1]
                sigmas[i] = abs(popt[2])
                sigma_errs[i] = perr[2]
        except (RuntimeError, ValueError):
            continue

    return centers, mus, sigmas, sigma_errs


def fit_polynomial(centers, values, errors, degree, min_points=5):
    """Weighted polynomial fit to slice results. Returns coefficients [c0, c1, ...]."""
    good = np.isfinite(values) & np.isfinite(errors) & (errors > 0)
    if good.sum() < min_points:
        avg = np.nanmean(values) if np.any(np.isfinite(values)) else 0.01
        return [float(avg)]

    x = centers[good]
    y = values[good]
    w = 1.0 / errors[good]

    coeffs = np.polyfit(x, y, degree, w=w)
    return list(reversed(coeffs.tolist()))


# ── Main pipeline ─────────────────────────────────────────────────

def build_cuts_for_particle(d, name, det_code, det_label, nbins, poly_deg_p, poly_deg_theta,
                             fd_cd_boundary=180.0):
    """Extract resolution and fit parametrization for one particle + detector.

    J/psi version: no CD, so the FD theta boundary is disabled (180 deg) —
    the decay leptons reach 40 deg in the FD.  FT entries have no boundary.
    """
    mask = d['status'] >= 1
    if det_code is not None:
        mask &= d['det'] == det_code
    # Physical theta boundary on top of det flag
    if det_label == "FD":
        mask &= d['theta_gen'] <  fd_cd_boundary
    elif det_label == "CD":
        mask &= d['theta_gen'] >= fd_cd_boundary

    n_det = int(mask.sum())
    if n_det < 200:
        return None, n_det

    p_gen = d['p_gen'][mask]
    p_rec = d['p_rec'][mask]
    th_gen = d['theta_gen'][mask]
    th_rec = d['theta_rec'][mask]
    phi_gen = d['phi_gen'][mask]
    phi_rec = d['phi_rec'][mask]

    dp = p_rec - p_gen
    dth = th_rec - th_gen
    dphi = (phi_rec - phi_gen + 180) % 360 - 180

    p_range = (max(0.1, np.percentile(p_gen, 0.5)),
               np.percentile(p_gen, 99.5) * 1.05)
    th_range = (max(0, np.percentile(th_gen, 0.5) - 1),
                min(70, np.percentile(th_gen, 99.5) * 1.05))

    # Fit slices
    c_p, mu_p, sig_p, sigerr_p = fit_slices(p_gen, dp, p_range, nbins)
    c_th, mu_th, sig_th, sigerr_th = fit_slices(th_gen, dth, th_range, nbins)
    c_phi, mu_phi, sig_phi, sigerr_phi = fit_slices(phi_gen, dphi, (-180, 180), nbins)

    # Fit polynomials
    mu_p_coeffs = fit_polynomial(c_p, mu_p, sigerr_p, poly_deg_p)
    sig_p_coeffs = fit_polynomial(c_p, sig_p, sigerr_p, poly_deg_p)

    mu_th_coeffs = fit_polynomial(c_th, mu_th, sigerr_th, poly_deg_theta)
    sig_th_coeffs = fit_polynomial(c_th, sig_th, sigerr_th, poly_deg_theta)

    # φ: approximately constant, just use average
    good_phi = np.isfinite(sig_phi)
    sig_phi_avg = float(np.mean(sig_phi[good_phi])) if good_phi.any() else 1.0
    mu_phi_avg = float(np.mean(mu_phi[good_phi])) if good_phi.any() else 0.0

    result = {
        "dp_vs_p": {
            "mu_coeffs": mu_p_coeffs,
            "sigma_coeffs": sig_p_coeffs,
            "p_range": [float(p_range[0]), float(p_range[1])],
        },
        "dtheta_vs_theta": {
            "mu_coeffs": mu_th_coeffs,
            "sigma_coeffs": sig_th_coeffs,
            "theta_range": [float(th_range[0]), float(th_range[1])],
        },
        "dphi_vs_phi": {
            "mu_avg": mu_phi_avg,
            "sigma_avg": sig_phi_avg,
        },
        "n_detected": n_det,
    }

    # Raw slice data for plotting
    plot_data = {
        'p': (c_p, mu_p, sig_p, sigerr_p, p_range, dp, p_gen),
        'th': (c_th, mu_th, sig_th, sigerr_th, th_range, dth, th_gen),
        'phi': (c_phi, mu_phi, sig_phi, sigerr_phi, (-180, 180), dphi, phi_gen),
    }

    return result, n_det, plot_data


def eval_poly(coeffs, x):
    return sum(c * x**i for i, c in enumerate(coeffs))


def make_validation_pdf(pdf, name, det_label, cut_data, plot_data, n_sigma):
    """Add validation pages for one particle+detector to the PDF."""
    for var, (centers, mus, sigmas, sigerrs, x_range, dx, x_gen) in plot_data.items():
        good = np.isfinite(sigmas)
        if not good.any():
            continue

        if var == 'p':
            x_label, dx_label = "|p| (GeV)", "Δ|p| (GeV)"
            cd = cut_data["dp_vs_p"]
            is_momentum = True
        elif var == 'th':
            x_label, dx_label = "θ (deg)", "Δθ (deg)"
            cd = cut_data["dtheta_vs_theta"]
            is_momentum = False
        else:
            x_label, dx_label = "φ (deg)", "Δφ (deg)"
            cd = cut_data["dphi_vs_phi"]
            is_momentum = False

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Left: 2D histogram with polynomial fit overlay
        ax = axes[0]
        p5, p95 = np.percentile(dx, [2, 98])
        dx_pad = (p95 - p5) * 0.3
        dx_range = (p5 - dx_pad, p95 + dx_pad)
        ax.hist2d(x_gen, dx, bins=[80, 100], range=[x_range, dx_range],
                  cmin=1, cmap="viridis")

        if var != 'phi':
            x_fine = np.linspace(x_range[0], x_range[1], 200)
            mu_fit = np.array([eval_poly(cd["mu_coeffs"], x) for x in x_fine])
            sig_fit = np.array([eval_poly(cd["sigma_coeffs"], x) for x in x_fine])
            ax.plot(x_fine, mu_fit + n_sigma * sig_fit, 'r-', lw=2,
                    label=f'±{n_sigma:.0f}σ poly')
            ax.plot(x_fine, mu_fit - n_sigma * sig_fit, 'r-', lw=2)
            ax.plot(x_fine, mu_fit, 'r--', lw=1, label='μ poly')
        else:
            mu_avg = cd["mu_avg"]
            sig_avg = cd["sigma_avg"]
            ax.axhline(mu_avg + n_sigma * sig_avg, color='r', lw=2,
                       label=f'±{n_sigma:.0f}σ = ±{n_sigma*sig_avg:.2f}°')
            ax.axhline(mu_avg - n_sigma * sig_avg, color='r', lw=2)
            ax.axhline(mu_avg, color='r', ls='--', lw=1)

        # Overlay slice fit points
        ax.plot(centers[good], mus[good] + n_sigma * sigmas[good],
                'w.', ms=3, alpha=0.6)
        ax.plot(centers[good], mus[good] - n_sigma * sigmas[good],
                'w.', ms=3, alpha=0.6)

        ax.set_xlabel(x_label)
        ax.set_ylabel(dx_label)
        ax.set_title(f"{name} [{det_label}]  {dx_label} vs {x_label}")
        ax.legend(fontsize=8)

        # Middle: σ vs x with polynomial fit
        ax = axes[1]
        ax.errorbar(centers[good], sigmas[good], yerr=sigerrs[good],
                    fmt='ko', ms=4, capsize=2, label='Gauss fits')
        if var != 'phi':
            ax.plot(x_fine, sig_fit, 'r-', lw=2, label=f'poly deg {len(cd["sigma_coeffs"])-1}')
        else:
            ax.axhline(sig_avg, color='r', lw=2, label=f'avg = {sig_avg:.3f}°')
        ax.set_xlabel(x_label)
        ax.set_ylabel(f"σ({dx_label})")
        ax.set_title(f"σ({dx_label}) vs {x_label}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x_range)

        # Right: σ/x or μ
        ax = axes[2]
        if is_momentum and good.any():
            ratio = sigmas[good] / centers[good] * 100
            ratio_err = sigerrs[good] / centers[good] * 100
            ax.errorbar(centers[good], ratio, yerr=ratio_err,
                        fmt='ko', ms=4, capsize=2)
            if var != 'phi':
                ax.plot(x_fine, sig_fit / x_fine * 100, 'r-', lw=2)
            ax.set_ylabel(f"σ({dx_label}) / |p|  (%)")
            ax.set_title(f"σ/|p| vs {x_label}")
        else:
            ax.errorbar(centers[good], mus[good], fmt='ko', ms=4, capsize=2)
            if var != 'phi':
                ax.plot(x_fine, mu_fit, 'r-', lw=2)
            else:
                ax.axhline(mu_avg, color='r', lw=2)
            ax.axhline(0, color='gray', lw=0.8)
            ax.set_ylabel(f"μ({dx_label})")
            ax.set_title(f"μ({dx_label}) vs {x_label}")
        ax.set_xlabel(x_label)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x_range)

        fig.suptitle(f"{name} [{det_label}]  N={cut_data['n_detected']:,}", fontsize=12)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def main():
    args = parse_args()

    print(f"Reading {args.input} ...")
    particles, names, n_events = read_particles_fast(args.input, args.max_events)
    print(f"  {n_events:,} events, {len(particles)} particles: {names}")

    # J/psi: scattered e' in FT, everything else in FD; no CD.
    # particles{} is keyed by particle INDEX, the output JSON by species x det,
    # so the two e- merge cleanly: 'e-'/'FT' = scattered, 'e-'/'FD' = decay.
    det_configs = [
        (1, "FD"),
        (3, "FT"),
    ]

    output = {
        "source": os.path.basename(args.input),
        "n_sigma": args.n_sigma,
        "n_events": n_events,
        "particles": {},
    }

    pdf_path = args.pdf or args.output.replace(".json", ".pdf")
    os.makedirs(os.path.dirname(pdf_path) or ".", exist_ok=True)

    with PdfPages(pdf_path) as pdf:
        for k in sorted(particles.keys()):
            d = particles[k]
            name = names[k]

            for det_code, det_label in det_configs:
                result = build_cuts_for_particle(
                    d, name, det_code, det_label,
                    args.nbins, args.poly_deg_p, args.poly_deg_theta)

                if result[0] is None:
                    n_det = result[1]
                    print(f"  {name} [{det_label}]: {n_det} detected, skipping")
                    continue

                cut_data, n_det, plot_data = result
                print(f"  {name} [{det_label}]: {n_det:,} detected")

                if name not in output["particles"]:
                    output["particles"][name] = {}
                output["particles"][name][det_label] = cut_data

                make_validation_pdf(pdf, name, det_label, cut_data, plot_data,
                                    args.n_sigma)

        # Summary page
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.axis("off")
        rows = [["Particle", "Det", "N detected",
                 "σ(ΔP) coeffs", "σ(Δθ) coeffs", "σ(Δφ) avg (deg)",
                 f"{args.n_sigma:.0f}σ ΔP @ 2 GeV", f"{args.n_sigma:.0f}σ Δθ @ 20°"]]

        for pname, pdata in output["particles"].items():
            for det, ddata in pdata.items():
                sp = ddata["dp_vs_p"]["sigma_coeffs"]
                st = ddata["dtheta_vs_theta"]["sigma_coeffs"]
                sphi = ddata["dphi_vs_phi"]["sigma_avg"]
                sig_p_at_2 = eval_poly(sp, 2.0)
                sig_th_at_20 = eval_poly(st, 20.0)
                cut_p = eval_poly(ddata["dp_vs_p"]["mu_coeffs"], 2.0) + args.n_sigma * sig_p_at_2
                cut_th = eval_poly(ddata["dtheta_vs_theta"]["mu_coeffs"], 20.0) + args.n_sigma * sig_th_at_20

                sp_str = ", ".join(f"{c:.5f}" for c in sp)
                st_str = ", ".join(f"{c:.5f}" for c in st)

                rows.append([pname, det, f"{ddata['n_detected']:,}",
                             sp_str, st_str, f"{sphi:.3f}",
                             f"±{abs(cut_p):.4f} GeV",
                             f"±{abs(cut_th):.3f}°"])

        table = ax.table(cellText=rows, loc="center", cellLoc="center",
                         colWidths=[0.08, 0.05, 0.10, 0.22, 0.22, 0.09, 0.12, 0.12])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.6)
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor("lightgray")
            if r == 0:
                cell.set_facecolor("lightyellow")
                cell.set_text_props(weight="bold")
        fig.suptitle(f"Matching cuts summary — {args.n_sigma:.0f}σ — {os.path.basename(args.input)}",
                     fontsize=14)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    # Save JSON
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved: {args.output}")
    print(f"Saved: {pdf_path}")

    # Print summary
    print(f"\n{'='*70}")
    print(f"MATCHING CUTS SUMMARY  ({args.n_sigma:.0f}σ)")
    print(f"{'='*70}")
    for pname, pdata in output["particles"].items():
        for det, ddata in pdata.items():
            sp = ddata["dp_vs_p"]["sigma_coeffs"]
            mp = ddata["dp_vs_p"]["mu_coeffs"]
            st = ddata["dtheta_vs_theta"]["sigma_coeffs"]
            mt = ddata["dtheta_vs_theta"]["mu_coeffs"]
            sphi = ddata["dphi_vs_phi"]["sigma_avg"]
            mphi = ddata["dphi_vs_phi"]["mu_avg"]

            print(f"\n  {pname} [{det}]  ({ddata['n_detected']:,} particles)")
            print(f"    σ(ΔP) = {' + '.join(f'{c:.6f}·p^{i}' for i,c in enumerate(sp))}")
            print(f"    μ(ΔP) = {' + '.join(f'{c:.6f}·p^{i}' for i,c in enumerate(mp))}")
            print(f"    σ(Δθ) = {' + '.join(f'{c:.6f}·θ^{i}' for i,c in enumerate(st))}")
            print(f"    μ(Δθ) = {' + '.join(f'{c:.6f}·θ^{i}' for i,c in enumerate(mt))}")
            print(f"    σ(Δφ) = {sphi:.4f}°  (constant),  μ(Δφ) = {mphi:.4f}°")

            # Sample cuts at a few momenta
            p_range = ddata["dp_vs_p"]["p_range"]
            ps = np.linspace(p_range[0], p_range[1], 5)
            print(f"    Sample {args.n_sigma:.0f}σ ΔP cuts:")
            for p in ps:
                mu = eval_poly(mp, p)
                sig = eval_poly(sp, p)
                print(f"      p={p:.1f} GeV: [{mu - args.n_sigma*sig:+.4f}, "
                      f"{mu + args.n_sigma*sig:+.4f}] GeV")


if __name__ == "__main__":
    main()
