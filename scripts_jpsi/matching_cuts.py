"""Momentum-dependent matching cuts from resolution parametrization.

Loads a JSON produced by build_matching_cuts.py and provides
    MatchingCuts.window(particle, det, p, theta, phi)
returning (dp_max, dtheta_max, dphi_max) at the given kinematics.

Usage:
    from matching_cuts import MatchingCuts
    mc = MatchingCuts("matching_cuts.json")
    (dp_lo, dp_hi), (dt_lo, dt_hi), (dphi_lo, dphi_hi) = mc.window("K+", "FD", p=2.5, theta=15.0)
    # A residual passes if:  dp_lo < dp < dp_hi  (asymmetric around μ)
"""

import json
import numpy as np


class MatchingCuts:
    def __init__(self, json_file):
        with open(json_file) as f:
            self.data = json.load(f)
        self.n_sigma = self.data["n_sigma"]

    def _eval_poly(self, coeffs, x):
        """Evaluate polynomial: coeffs[0] + coeffs[1]*x + coeffs[2]*x^2 + ..."""
        return sum(c * x**i for i, c in enumerate(coeffs))

    def _get_window(self, particle, det, variable, x):
        """Get (lo, hi) cut window for a variable at point x: [μ - Nσ, μ + Nσ]."""
        p = self.data["particles"].get(particle)
        if p is None:
            fb = self._fallback(variable)
            return -fb, fb
        d = p.get(det)
        if d is None:
            fb = self._fallback(variable)
            return -fb, fb
        v = d.get(variable)
        if v is None:
            fb = self._fallback(variable)
            return -fb, fb

        if "mu_coeffs" in v:
            mu = self._eval_poly(v["mu_coeffs"], x)
            sigma = self._eval_poly(v["sigma_coeffs"], x)
        else:
            mu = v.get("mu_avg", 0.0)
            sigma = v.get("sigma_avg", 1.0)
        sigma = max(sigma, 1e-6)
        return mu - self.n_sigma * sigma, mu + self.n_sigma * sigma

    def _fallback(self, variable):
        """Conservative fallback if particle/det not in JSON."""
        if variable == "dp_vs_p":
            return 0.50
        elif variable == "dtheta_vs_theta":
            return 4.0
        else:
            return 5.0

    def window(self, particle, det, p, theta, phi=0.0):
        """Return ((dp_lo, dp_hi), (dth_lo, dth_hi), (dphi_lo, dphi_hi))
        at given kinematics.  Each pair is [μ - Nσ, μ + Nσ]."""
        dp_win = self._get_window(particle, det, "dp_vs_p", p)
        dt_win = self._get_window(particle, det, "dtheta_vs_theta", theta)
        dphi_win = self._get_window(particle, det, "dphi_vs_phi", phi)
        return dp_win, dt_win, dphi_win

    def window_arrays(self, particle, det, p_arr, theta_arr):
        """Vectorized version for arrays. Returns (dp_lo, dp_hi), (dt_lo, dt_hi), (dphi_lo, dphi_hi)."""
        dp_lo = np.empty(len(p_arr))
        dp_hi = np.empty(len(p_arr))
        for i, p in enumerate(p_arr):
            dp_lo[i], dp_hi[i] = self._get_window(particle, det, "dp_vs_p", p)
        dt_lo = np.empty(len(theta_arr))
        dt_hi = np.empty(len(theta_arr))
        for i, t in enumerate(theta_arr):
            dt_lo[i], dt_hi[i] = self._get_window(particle, det, "dtheta_vs_theta", t)
        dphi_info = self.data["particles"].get(particle, {}).get(det, {}).get("dphi_vs_phi", {})
        mu_phi = dphi_info.get("mu_avg", 0.0)
        sig_phi = dphi_info.get("sigma_avg", 1.0)
        dphi_lo = mu_phi - self.n_sigma * sig_phi
        dphi_hi = mu_phi + self.n_sigma * sig_phi
        return (dp_lo, dp_hi), (dt_lo, dt_hi), (np.full(len(p_arr), dphi_lo), np.full(len(p_arr), dphi_hi))

    @property
    def particles(self):
        return list(self.data["particles"].keys())

    @property
    def source(self):
        return self.data.get("source", "unknown")

    def summary(self):
        """Print human-readable summary."""
        print(f"Matching cuts from: {self.source}")
        print(f"N sigma: {self.n_sigma}")
        for pname, pdata in self.data["particles"].items():
            for det, ddata in pdata.items():
                dp = ddata.get("dp_vs_p", {})
                dt = ddata.get("dtheta_vs_theta", {})
                dphi = ddata.get("dphi_vs_phi", {})
                print(f"  {pname:>3s} [{det}]:  "
                      f"σ(ΔP) poly deg {len(dp.get('sigma_coeffs', []))-1}  "
                      f"σ(Δθ) poly deg {len(dt.get('sigma_coeffs', []))-1}  "
                      f"σ(Δφ) avg={dphi.get('sigma_avg', 0):.3f}°")
