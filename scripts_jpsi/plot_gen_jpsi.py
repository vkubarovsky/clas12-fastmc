#!/usr/bin/env python3
"""Plot GENERATED kinematics from J/psi training .dat files
(produced by make_training_data_truthmatch.py, reaction epe+e-).

4 pages:
  1. Event:    Q2, xB, W, t, t' = |t - t_min|, M(e+e-)
  2. e' (FT):  E, theta, theta vs E
  3. Decay leptons: e+ and e-  E, theta, theta vs E
  4. Proton:   E, theta, theta vs E

Usage:
  python plot_gen_jpsi.py jpsi_tm_train.dat
  python plot_gen_jpsi.py jpsi_tm_train.dat -o gen_plots.pdf --max_events 200000
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

M_E = 0.000511
M_P = 0.938272

# particle order in the epe+e- .dat file
IDX_ESCAT, IDX_PROT, IDX_EPLUS, IDX_EMINUS = 0, 1, 2, 3
MASSES = [M_E, M_P, M_E, M_E]
NPART = 4


def t_min(Q2, W, M_V, M=M_P):
    """Kinematic t at theta_cm = 0 for gamma*(Q2) p -> V p at energy W (signed, ~small negative)."""
    W2 = W * W
    Ea = (W2 - Q2 - M * M) / (2.0 * W)          # virtual photon CM energy
    pa = np.sqrt(np.maximum(Ea * Ea + Q2, 0.0))  # photon CM momentum (m^2 = -Q2)
    Ec = (W2 + M_V * M_V - M * M) / (2.0 * W)    # meson CM energy
    pc2 = Ec * Ec - M_V * M_V
    pc = np.sqrt(np.maximum(pc2, 0.0))
    return -Q2 + M_V * M_V - 2.0 * (Ea * Ec - pa * pc)


def read_dat(filename, max_events=0):
    """Return dict of event arrays + per-particle gen arrays."""
    Q2 = []; xB = []; W = []; t = []; Mh = []
    p_gen = [[] for _ in range(NPART)]
    th_gen = [[] for _ in range(NPART)]
    n_ev = 0
    with open(filename) as f:
        need_particles = 0
        for line in f:
            if line.startswith('#'):
                continue
            if need_particles == 0:
                # event line: event_num nrec Q2 xB W t Mh
                w = line.split()
                if len(w) < 7:
                    continue
                Q2.append(float(w[2])); xB.append(float(w[3]))
                W.append(float(w[4])); t.append(float(w[5])); Mh.append(float(w[6]))
                need_particles = NPART
                k = 0
            else:
                # particle line: status pid det p_gen theta_gen phi_gen vz_gen [rec...]
                w = line.split()
                p_gen[k].append(float(w[3]))
                th_gen[k].append(float(w[4]))
                k += 1
                need_particles -= 1
                if need_particles == 0:
                    n_ev += 1
                    if max_events > 0 and n_ev >= max_events:
                        break
    out = {
        'Q2': np.array(Q2), 'xB': np.array(xB), 'W': np.array(W),
        't': np.array(t), 'Mh': np.array(Mh),
        'p': [np.array(a) for a in p_gen],
        'theta': [np.array(a) for a in th_gen],
        'E': [np.sqrt(np.array(a) ** 2 + m * m) for a, m in zip(p_gen, MASSES)],
    }
    print(f"Read {n_ev} events from {filename}")
    return out


def hist_panel(ax, x, label, bins=100, rng=None, log=False):
    ax.hist(x, bins=bins, range=rng, histtype='stepfilled',
            color='steelblue', edgecolor='black', linewidth=0.4)
    ax.set_xlabel(label)
    ax.set_ylabel('events')
    if log:
        ax.set_yscale('log')
    ax.grid(alpha=0.3)


def hist2d_panel(ax, x, y, xlabel, ylabel, bins=100, rng=None):
    h = ax.hist2d(x, y, bins=bins, range=rng, cmap='viridis', cmin=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.colorbar(h[3], ax=ax)


def particle_page(pdf, E, th, name, suptitle):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    hist_panel(axes[0], E, f"E({name}) gen  [GeV]")
    hist_panel(axes[1], th, f"theta({name}) gen  [deg]")
    hist2d_panel(axes[2], E, th, f"E({name})  [GeV]", f"theta({name})  [deg]")
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="jpsi .dat file (train or val)")
    ap.add_argument("-o", "--output", default=None, help="output PDF")
    ap.add_argument("--max_events", type=int, default=0)
    args = ap.parse_args()

    out_pdf = args.output or os.path.splitext(args.input)[0] + "_gen.pdf"
    d = read_dat(args.input, args.max_events)
    n = len(d['Q2'])
    if n == 0:
        sys.exit("No events read.")

    tmin = t_min(d['Q2'], d['W'], d['Mh'])
    tprime = np.abs(d['t'] - tmin)
    src = os.path.basename(args.input)

    with PdfPages(out_pdf) as pdf:
        # ── Page 1: event kinematics ──
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        hist_panel(axes[0, 0], d['Q2'], r"$Q^2$  [GeV$^2$]", log=True)
        hist_panel(axes[0, 1], d['xB'], r"$x_B$", log=True)
        hist_panel(axes[0, 2], d['W'], r"$W$  [GeV]")
        hist_panel(axes[1, 0], d['t'], r"$t$  [GeV$^2$]")
        hist_panel(axes[1, 1], tprime, r"$t' = |t - t_{min}|$  [GeV$^2$]", log=True)
        hist_panel(axes[1, 2], d['Mh'], r"$M(e^+e^-)$  [GeV]")
        fig.suptitle(f"Generated event kinematics — {src}  ({n} events)", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)

        # ── Page 2: scattered electron (FT) ──
        particle_page(pdf, d['E'][IDX_ESCAT], d['theta'][IDX_ESCAT], "e'",
                      f"Generated scattered electron (FT) — {src}")

        # ── Page 3: decay leptons ──
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        hist_panel(axes[0, 0], d['E'][IDX_EPLUS], "E(e+) gen  [GeV]")
        hist_panel(axes[0, 1], d['theta'][IDX_EPLUS], "theta(e+) gen  [deg]")
        hist2d_panel(axes[0, 2], d['E'][IDX_EPLUS], d['theta'][IDX_EPLUS],
                     "E(e+)  [GeV]", "theta(e+)  [deg]")
        hist_panel(axes[1, 0], d['E'][IDX_EMINUS], "E(e-) gen  [GeV]")
        hist_panel(axes[1, 1], d['theta'][IDX_EMINUS], "theta(e-) gen  [deg]")
        hist2d_panel(axes[1, 2], d['E'][IDX_EMINUS], d['theta'][IDX_EMINUS],
                     "E(e-)  [GeV]", "theta(e-)  [deg]")
        fig.suptitle(f"Generated J/psi decay leptons — {src}", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)

        # ── Page 4: proton ──
        particle_page(pdf, d['E'][IDX_PROT], d['theta'][IDX_PROT], "p",
                      f"Generated proton — {src}")

    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
