#!/usr/bin/env python3
"""Grid-based FastMC: per-particle 4-D efficiency and resolution lookup.

For each particle, loads grid_eff.npz built by build_grid_efficiency.py:
    eps   : 4-D array of acceptance probabilities (p, θ, φ, vz)
    mean  : 4-D × 4 array of Δ means (when resolution is stored)
    sigma : 4-D × 4 array of Δ widths
    edges : 4 bin-edge arrays

Inference:
    - Look up bin index per event (nearest neighbour OR quadrilinear interp)
    - Bernoulli on ε
    - Sample (Δp, Δθ, Δφ, Δvz) as independent Gaussians with stored μ, σ
    - kin_rec = kin_gen + Δ

Same external API as fast_mc.FastMC so it drops into validate_fast_mc_v2.

This module also provides:
    load_grid_model(model_dir, particle_name, *, interpolate=True)
        → returns a GridFastMC instance
"""

import os
import numpy as np


class GridFastMC:
    """Pure non-parametric per-particle FastMC backed by a 4-D grid."""

    def __init__(self, grid_npz_path, particle_name, interpolate=True,
                 sparse_threshold=10, eps_floor=0.0):
        """Load grid for one particle from a shared npz file.

        Args:
            grid_npz_path:    path to file built by build_grid_efficiency.py
            particle_name:    e.g. 'K+', 'p', 'e-'
            interpolate:      True → quadrilinear in-bin interp; False → nearest
            sparse_threshold: bins with denominator < this are flagged sparse
                              (caller may want to smooth offline)
            eps_floor:        clamp ε below this to 0 (drops sub-sample noise)
        """
        self.particle_name = particle_name
        self.interpolate   = interpolate
        self.sparse_threshold = sparse_threshold
        self.eps_floor     = eps_floor

        f = np.load(grid_npz_path, allow_pickle=True)
        if f"{particle_name}_eps" not in f.files:
            raise KeyError(f"No grid for {particle_name!r} in {grid_npz_path}")
        self.eps   = f[f"{particle_name}_eps"]
        self.H_den = f[f"{particle_name}_H_den"]
        self.edges = [f[f"{particle_name}_edges_{i}"] for i in range(4)]
        self.bin_widths = [e[1] - e[0] for e in self.edges]
        self.shape = self.eps.shape

        # Optional smearing arrays
        self.has_smear = f"{particle_name}_mean_d" in f.files
        if self.has_smear:
            self.mean_d  = f[f"{particle_name}_mean_d"]   # shape (n_p, n_th, n_phi, n_vz, 4)
            self.sigma_d = f[f"{particle_name}_sigma_d"]  # same

        # Apply sparse smoothing (in-place) and ε floor
        if sparse_threshold > 0:
            self._smooth_sparse_bins()
        if eps_floor > 0:
            self.eps = np.where(self.eps < eps_floor, 0.0, self.eps)

    def _smooth_sparse_bins(self):
        """Replace under-populated bins' ε with neighbour-average ε."""
        mask = self.H_den < self.sparse_threshold
        if not mask.any():
            return
        # Simple iterative neighbour fill: do up to 3 passes
        for _ in range(3):
            new_eps = self.eps.copy()
            # 4-D neighbours: shift by ±1 in each axis, average where the
            # shifted bin had enough events
            shifts = [(-1,0,0,0),(1,0,0,0),(0,-1,0,0),(0,1,0,0),
                      (0,0,-1,0),(0,0,1,0),(0,0,0,-1),(0,0,0,1)]
            acc = np.zeros_like(self.eps)
            cnt = np.zeros_like(self.eps)
            for dp, dt, df, dv in shifts:
                shifted_eps = np.roll(self.eps, shift=(dp, dt, df, dv), axis=(0,1,2,3))
                shifted_den = np.roll(self.H_den, shift=(dp, dt, df, dv), axis=(0,1,2,3))
                valid = shifted_den >= self.sparse_threshold
                acc += np.where(valid, shifted_eps, 0)
                cnt += valid.astype(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                neigh_mean = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0)
            # Only update the sparse bins
            new_eps[mask] = neigh_mean[mask]
            self.eps = new_eps

    # ── Bin index helpers ─────────────────────────────────────────────

    def _bin_index(self, p, theta, phi, vz):
        """Return (ip, it, iphi, ivz) clipped to valid range."""
        ip   = np.clip(np.digitize(p,     self.edges[0]) - 1, 0, self.shape[0]-1)
        it   = np.clip(np.digitize(theta, self.edges[1]) - 1, 0, self.shape[1]-1)
        iphi = np.clip(np.digitize(phi,   self.edges[2]) - 1, 0, self.shape[2]-1)
        ivz  = np.clip(np.digitize(vz,    self.edges[3]) - 1, 0, self.shape[3]-1)
        return ip, it, iphi, ivz

    def _bin_frac(self, p, theta, phi, vz):
        """Return (low_idx, frac) tuples per axis for quadrilinear interp."""
        def axfrac(x, edges, n):
            # convert to bin-center coordinate (centers at i + 0.5)
            x = np.asarray(x, dtype=np.float64)
            w = edges[1] - edges[0]
            # position in "center units": e.g. 0 = first center, n-1 = last center
            t = (x - (edges[0] + 0.5 * w)) / w
            i0 = np.clip(np.floor(t).astype(int), 0, n - 2)
            f  = np.clip(t - i0, 0.0, 1.0)
            return i0, f
        ip, fp   = axfrac(p,     self.edges[0], self.shape[0])
        it, ft   = axfrac(theta, self.edges[1], self.shape[1])
        iphi, fphi = axfrac(phi, self.edges[2], self.shape[2])
        ivz, fvz = axfrac(vz,    self.edges[3], self.shape[3])
        return (ip, it, iphi, ivz), (fp, ft, fphi, fvz)

    def _lookup_eps(self, p, theta, phi, vz):
        if not self.interpolate:
            ip, it, iphi, ivz = self._bin_index(p, theta, phi, vz)
            return self.eps[ip, it, iphi, ivz]

        (ip, it, iphi, ivz), (fp, ft, fphi, fvz) = self._bin_frac(p, theta, phi, vz)
        # 16 corner sum
        result = np.zeros_like(fp)
        for di_p in (0, 1):
            wp = fp if di_p else (1 - fp)
            for di_t in (0, 1):
                wt = ft if di_t else (1 - ft)
                for di_ph in (0, 1):
                    wph = fphi if di_ph else (1 - fphi)
                    for di_v in (0, 1):
                        wv = fvz if di_v else (1 - fvz)
                        w = wp * wt * wph * wv
                        result += w * self.eps[
                            np.minimum(ip + di_p,    self.shape[0]-1),
                            np.minimum(it + di_t,    self.shape[1]-1),
                            np.minimum(iphi + di_ph, self.shape[2]-1),
                            np.minimum(ivz + di_v,   self.shape[3]-1),
                        ]
        return result

    # ── Public API mimicking FastMC ──────────────────────────────────

    def accept(self, p, theta, phi, vz):
        p, theta, phi, vz = np.atleast_1d(p, theta, phi, vz)
        prob = self._lookup_eps(p, theta, phi, vz)
        return np.random.random(len(prob)) < prob

    def smear(self, p, theta, phi, vz):
        p, theta, phi, vz = np.atleast_1d(p, theta, phi, vz)
        if not self.has_smear:
            return p.copy(), theta.copy(), phi.copy(), vz.copy()
        ip, it, iphi, ivz = self._bin_index(p, theta, phi, vz)
        mu  = self.mean_d [ip, it, iphi, ivz]   # (N, 4)
        sig = self.sigma_d[ip, it, iphi, ivz]   # (N, 4)
        z = np.random.randn(len(p), 4)
        delta = mu + sig * z
        return (p     + delta[:, 0],
                theta + delta[:, 1],
                phi   + delta[:, 2],
                vz    + delta[:, 3])

    def simulate(self, p, theta, phi, vz):
        p, theta, phi, vz = np.atleast_1d(p, theta, phi, vz)
        mask = self.accept(p, theta, phi, vz)
        p_rec, t_rec, ph_rec, vz_rec = self.smear(p[mask], theta[mask], phi[mask], vz[mask])
        return {
            "accepted": mask,
            "p_mc":     p[mask],     "theta_mc": theta[mask],
            "phi_mc":   phi[mask],   "vz_mc":    vz[mask],
            "p_rec":    p_rec,       "theta_rec": t_rec,
            "phi_rec":  ph_rec,      "vz_rec":    vz_rec,
        }

    # For interop with load_model_auto pattern
    @property
    def n_features(self):
        return 4   # grid lookup uses (p, θ, φ, vz)
    @property
    def calibration_T(self):
        return 1.0
    @property
    def calibration_b(self):
        return 0.0


def load_grid_model(grid_npz_path, particle_name, **kwargs):
    """Convenience wrapper."""
    if not os.path.exists(grid_npz_path):
        return None
    try:
        return GridFastMC(grid_npz_path, particle_name, **kwargs)
    except KeyError:
        return None


class EventFastMC:
    """Hierarchical event-level FastMC.

    Reflects the CLAS12 reconstruction reality:
      1. e- is the trigger: its acceptance is marginal over all events.
      2. Hadrons (p, K+, K-, ...) PIDs are only reliable when an electron
         was reconstructed.  So hadron acceptance models are trained
         CONDITIONAL on electron detection, and at inference we sample
         them ONLY in events where the electron was first accepted.

    `simulate_event` returns the per-particle accept booleans honoring
    this hierarchy.  Joint acceptance = AND of all individual accepts.
    """
    def __init__(self, electron_model, hadron_models):
        """
        Args:
            electron_model: a FastMC/FastMCPair/GridFastMC for the electron.
                            Its accept() should give MARGINAL P(e- detected | kin_e).
            hadron_models:  dict {particle_name -> FastMC-like} for hadrons.
                            Their accept() should give CONDITIONAL
                            P(hadron detected | electron detected, kin_hadron).
        """
        self.electron = electron_model
        self.hadrons  = hadron_models    # name -> model

    def simulate_event(self, kin_e, kin_p, kin_kplus, kin_kminus):
        """Sample acceptance per particle for a batch of events.

        Args:
            kin_e:       tuple (p, theta, phi, vz)  for the electron
            kin_p, kin_kplus, kin_kminus: same for each hadron

        Returns:
            dict {'e-': bool array, 'p': ..., 'K+': ..., 'K-': ...}
            For events where the electron is rejected, all hadron accepts
            are set to False.
        """
        e_acc = self.electron.accept(*kin_e)
        out = {"e-": e_acc}
        kin_by_name = {"p": kin_p, "K+": kin_kplus, "K-": kin_kminus}
        for name, kin in kin_by_name.items():
            if name not in self.hadrons:
                out[name] = np.zeros_like(e_acc)
                continue
            had_acc = self.hadrons[name].accept(*kin)
            # Hadron is "accepted" only when e- also accepted in this event.
            out[name] = e_acc & had_acc
        return out

