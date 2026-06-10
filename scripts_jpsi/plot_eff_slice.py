#!/usr/bin/env python3
"""Plot GEMC vs FastMC efficiency vs |p| in a narrow (theta, phi) slice.

Usage:
  python plot_eff_slice.py phi_tm_val.dat --mlp_dir models/phi_v11 \
      --theta_min 15 --theta_max 25 --phi_center 0 --phi_half 1 \
      -o plots/eff_slice_v11.pdf
"""

import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

PDG_TO_SHORT = {11: 'e-', -11: 'e+', 2212: 'p', 321: 'K+', -321: 'K-',
                211: 'pi+', -211: 'pi-'}
PDG_MASS = {11: 0.000511, 2212: 0.938272, 321: 0.49368, -321: 0.49368,
            211: 0.13957, -211: 0.13957}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help=".dat val file")
    p.add_argument("--mlp_dir", required=True, help="directory with .pt models")
    p.add_argument("--theta_min", type=float, default=15.0)
    p.add_argument("--theta_max", type=float, default=25.0)
    p.add_argument("--phi_center", type=float, default=0.0)
    p.add_argument("--phi_half", type=float, default=1.0)
    p.add_argument("--nbins", type=int, default=50)
    p.add_argument("--max_events", type=int, default=0)
    p.add_argument("-o", "--output", default="plots/eff_slice.pdf")
    return p.parse_args()


def read_dat_particles(filename, max_events=0):
    """Read .dat and return per-particle arrays of (p_gen, theta_gen, phi_gen, status)."""
    header = {}
    npart = None
    data = {}  # k -> lists

    n_events = 0
    pidx = -1
    pids = []

    with open(filename) as f:
        for line in f:
            if line.startswith('#!'):
                kv = line[2:].strip()
                if ':' in kv:
                    k, v = kv.split(':', 1)
                    header[k.strip()] = v.strip()
                continue
            if not line.strip():
                continue

            if line[0] == ' ':
                pidx += 1
                parts = line.split()
                status = int(parts[0])
                pid = int(parts[1])

                # Detect format
                if '.' not in parts[2]:
                    # New compact: status pid det p_gen theta phi vz [rec...]
                    p_gen = float(parts[3])
                    theta_gen = float(parts[4])
                    phi_gen = float(parts[5])
                else:
                    # Old format: status pid p_gen theta phi vz rec...
                    p_gen = float(parts[2])
                    theta_gen = float(parts[3])
                    phi_gen = float(parts[4])

                if npart is None:
                    pids.append(pid)
                elif pidx < npart:
                    if pidx not in data:
                        data[pidx] = {'p': [], 'theta': [], 'phi': [], 'status': [], 'pid': pid}
                    data[pidx]['p'].append(p_gen)
                    data[pidx]['theta'].append(theta_gen)
                    data[pidx]['phi'].append(phi_gen)
                    data[pidx]['status'].append(status)
            else:
                if npart is None and pids:
                    npart = len(pids)
                    for k in range(npart):
                        data[k] = {'p': [], 'theta': [], 'phi': [], 'status': [], 'pid': pids[k]}
                pidx = -1
                n_events += 1
                if max_events > 0 and n_events >= max_events:
                    break

    # Convert to arrays
    for k in data:
        for key in ['p', 'theta', 'phi', 'status']:
            data[k][key] = np.array(data[k][key])

    return header, npart, data


def main():
    args = parse_args()

    print(f"Reading {args.input} ...")
    header, npart, data = read_dat_particles(args.input, args.max_events)
    print(f"  {len(data[0]['p']):,} events, {npart} particles")

    # Load MLP models
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from fast_mc import load_model_auto

    names = [PDG_TO_SHORT.get(data[k]['pid'], '?') for k in range(npart)]
    print(f"  Particles: {names}")

    models = {}
    for k in range(npart):
        m = load_model_auto(args.mlp_dir, names[k])
        if m:
            models[k] = m
            print(f"  Loaded MLP for {names[k]}")

    # Slice parameters
    th_lo, th_hi = args.theta_min, args.theta_max
    phi_lo = args.phi_center - args.phi_half
    phi_hi = args.phi_center + args.phi_half

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    plot_data = []

    with PdfPages(args.output) as pdf:
        for k in range(npart):
            p = data[k]['p']
            theta = data[k]['theta']
            phi = data[k]['phi']
            status = data[k]['status']

            # Apply slice
            mask = (theta >= th_lo) & (theta < th_hi) & (phi >= phi_lo) & (phi < phi_hi)
            p_slice = p[mask]
            status_slice = status[mask]

            if len(p_slice) == 0:
                print(f"  {names[k]}: no events in slice")
                continue

            # GEMC efficiency
            p_max = np.percentile(p_slice, 99.5)
            bins = np.linspace(0, p_max, args.nbins + 1)
            gen_hist, _ = np.histogram(p_slice, bins=bins)
            rec_hist, _ = np.histogram(p_slice[status_slice > 0], bins=bins)

            with np.errstate(divide='ignore', invalid='ignore'):
                eff_gemc = np.where(gen_hist > 0, rec_hist / gen_hist, 0)
                # Wilson 68% CL error
                n, k_rec = gen_hist, rec_hist
                z = 1.0
                denom = 1 + z**2 / n
                center = (k_rec + z**2 / 2) / (n + z**2)
                halfwidth = z * np.sqrt(k_rec * (n - k_rec) / n + z**2 / 4) / (n + z**2)
                eff_lo = np.where(n > 0, center - halfwidth, 0)
                eff_hi = np.where(n > 0, center + halfwidth, 0)

            # MLP efficiency — use predicted probability (smooth), not random sampling
            if k in models:
                import torch
                p_all = p[mask]
                theta_all = theta[mask]
                phi_all = phi[mask]
                vz = np.full_like(p_all, -3.0)  # typical vz

                model = models[k]
                X = model._normalize(p_all, theta_all, phi_all, vz)
                with torch.no_grad():
                    prob = torch.sigmoid(model.acc_model(X)).numpy().flatten()
                prob = model._calibrate(prob)

                # Average probability per bin = smooth efficiency
                prob_sum, _ = np.histogram(p_all, bins=bins, weights=prob)
                with np.errstate(divide='ignore', invalid='ignore'):
                    eff_mlp = np.where(gen_hist > 0, prob_sum / gen_hist, 0)
            else:
                eff_mlp = None

            # Store for combined plot
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            plot_data.append({
                'name': names[k], 'bin_centers': bin_centers,
                'eff_gemc': eff_gemc, 'eff_lo': eff_lo, 'eff_hi': eff_hi,
                'eff_mlp': eff_mlp, 'n_gen': len(p_slice),
                'gemc_total_eff': rec_hist.sum() / gen_hist.sum() if gen_hist.sum() > 0 else 0,
            })
            print(f"  {names[k]}: N_gen={len(p_slice):,}, "
                  f"GEMC eff={plot_data[-1]['gemc_total_eff']:.3f}")

        # 2x2 combined plot
        if plot_data:
            nplots = len(plot_data)
            ncols = 2
            nrows = (nplots + 1) // 2
            fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows))
            axes = axes.flatten() if nplots > 1 else [axes]

            for i, pd in enumerate(plot_data):
                ax = axes[i]
                ax.errorbar(pd['bin_centers'], pd['eff_gemc'],
                            yerr=[pd['eff_gemc'] - pd['eff_lo'], pd['eff_hi'] - pd['eff_gemc']],
                            fmt='o', color='red', markersize=3, label='GEMC')
                if pd['eff_mlp'] is not None:
                    ax.plot(pd['bin_centers'], pd['eff_mlp'], '-', color='blue',
                            linewidth=2, label='FastMC (MLP)')
                ax.set_xlabel('|p| (GeV)', fontsize=11)
                ax.set_ylabel('Efficiency', fontsize=11)
                ax.set_title(f"{pd['name']}   N_gen = {pd['n_gen']:,}", fontsize=12)
                ax.set_ylim(-0.05, 1.15)
                ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.5)
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)

            # Hide unused axes
            for i in range(nplots, len(axes)):
                axes[i].set_visible(False)

            fig.suptitle(f'Efficiency vs |p|  —  slice: {th_lo:.0f} < θ < {th_hi:.0f}°,  '
                         f'{phi_lo:.0f} < φ < {phi_hi:.0f}°', fontsize=14)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
