#!/usr/bin/env python3
"""Fast MC for CLAS12: acceptance + smearing using trained NN models."""

import os
import numpy as np
import torch
import torch.nn as nn

from config import BASE_DIR

N_GAUSS = 5
N_OUT = 4
N_FEATURES = 6   # p, theta, phi, vz, sin(6*phi), cos(6*phi)


def phi_features(phi_deg):
    """Sector-periodic Fourier features.

    Adds sin(6*phi) and cos(6*phi) which capture the 6-fold sector
    periodicity of CLAS12 forward-detector acceptance.  These let the
    network represent the sharp sector dips that a smooth 4-feature MLP
    averages away.

    Args:
        phi_deg: array-like of phi in degrees.

    Returns:
        (sin6phi, cos6phi) numpy float32 arrays of the same shape.
    """
    phi_rad = np.radians(np.asarray(phi_deg, dtype=np.float32))
    return np.sin(6 * phi_rad).astype(np.float32), np.cos(6 * phi_rad).astype(np.float32)


class AcceptanceNet(nn.Module):
    def __init__(self, n_in=N_FEATURES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class SmearingMDN(nn.Module):
    def __init__(self, n_gauss=N_GAUSS, n_out=N_OUT, n_in=N_FEATURES):
        super().__init__()
        self.n_gauss = n_gauss
        self.n_out = n_out
        self.backbone = nn.Sequential(
            nn.Linear(n_in, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.head = nn.Linear(128, n_gauss * (1 + 2 * n_out))

    def forward(self, x):
        h = self.backbone(x)
        params = self.head(h)
        pi = torch.softmax(params[:, :self.n_gauss], dim=1)
        mu = params[:, self.n_gauss:self.n_gauss*(1+self.n_out)].view(-1, self.n_gauss, self.n_out)
        sigma = torch.exp(
            params[:, self.n_gauss*(1+self.n_out):].clamp(-5, 2)
        ).view(-1, self.n_gauss, self.n_out)
        return pi, mu, sigma

    def sample(self, x):
        pi, mu, sigma = self.forward(x)
        comp = torch.multinomial(pi, 1).squeeze(-1)
        batch_idx = torch.arange(len(x), device=x.device)
        chosen_mu = mu[batch_idx, comp]
        chosen_sigma = sigma[batch_idx, comp]
        return chosen_mu + chosen_sigma * torch.randn_like(chosen_mu)


class FastMC:
    def __init__(self, particle="electron", period="rga_fall_2018_inbending", detector="FD", model_file=None):
        if model_file is None:
            model_file = f"{BASE_DIR}/models/{period}/{particle}_{detector}.pt"
        checkpoint = torch.load(model_file, map_location="cpu", weights_only=True)
        self.X_mean = checkpoint["X_mean"].numpy()
        self.X_std  = checkpoint["X_std"].numpy()
        self.y_smear_mean = checkpoint["y_smear_mean"].numpy()
        self.y_smear_std  = checkpoint["y_smear_std"].numpy()

        # Detect feature count from the saved normalization tensor so we stay
        # backward-compatible with 4-feature checkpoints.  New checkpoints have
        # 6 features (the extra two are sin(6phi) and cos(6phi)).
        self.n_features = len(self.X_mean)
        assert self.n_features in (4, N_FEATURES), \
            f"unexpected n_features={self.n_features} in checkpoint"

        self.acc_model = AcceptanceNet(n_in=self.n_features)
        self.acc_model.load_state_dict(checkpoint["acc_model"])
        self.acc_model.eval()

        self.mdn_model = SmearingMDN(n_in=self.n_features)
        self.mdn_model.load_state_dict(checkpoint["mdn_model"])
        self.mdn_model.eval()

        # Optional temperature+bias calibration: P_new = sigmoid(T*logit(P) + b).
        # Set by load_model_auto if a calibration.json is found alongside.
        self.calibration_T = 1.0
        self.calibration_b = 0.0

    def _calibrate(self, prob):
        """Apply temperature+bias if a calibration was attached."""
        if self.calibration_T == 1.0 and self.calibration_b == 0.0:
            return prob
        p = np.clip(prob, 1e-7, 1.0 - 1e-7)
        L = np.log(p / (1.0 - p))
        z = np.clip(self.calibration_T * L + self.calibration_b, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-z))

    def _normalize(self, p, theta, phi, vz):
        cols = [p, theta, phi, vz]
        if self.n_features == N_FEATURES:
            s, c = phi_features(phi)
            cols += [s, c]
        X = np.column_stack(cols).astype(np.float32)
        return torch.tensor((X - self.X_mean) / self.X_std)

    @torch.no_grad()
    def accept(self, p, theta, phi, vz):
        """Return boolean array: True if particle is reconstructed."""
        p, theta, phi, vz = np.atleast_1d(p, theta, phi, vz)
        X = self._normalize(p, theta, phi, vz)
        prob = torch.sigmoid(self.acc_model(X)).numpy()
        prob = self._calibrate(prob)
        return np.random.random(len(prob)) < prob

    @torch.no_grad()
    def smear(self, p, theta, phi, vz):
        """Return (p_rec, theta_rec, phi_rec, vz_rec) with detector smearing."""
        p, theta, phi, vz = np.atleast_1d(p, theta, phi, vz)
        X = self._normalize(p, theta, phi, vz)
        delta_norm = self.mdn_model.sample(X).numpy()
        delta = delta_norm * self.y_smear_std + self.y_smear_mean
        return p + delta[:, 0], theta + delta[:, 1], phi + delta[:, 2], vz + delta[:, 3]

    def simulate(self, p, theta, phi, vz):
        """Full fast MC: accept/reject + smear. Returns dict with results."""
        p, theta, phi, vz = np.atleast_1d(p, theta, phi, vz)
        mask = self.accept(p, theta, phi, vz)
        p_rec, theta_rec, phi_rec, vz_rec = self.smear(p[mask], theta[mask], phi[mask], vz[mask])
        return {
            "accepted": mask,
            "p_mc": p[mask], "theta_mc": theta[mask], "phi_mc": phi[mask], "vz_mc": vz[mask],
            "p_rec": p_rec, "theta_rec": theta_rec, "phi_rec": phi_rec, "vz_rec": vz_rec,
        }


class FastMCPair:
    """Combined FD + CD model for one hadron.

    Loads two FastMC instances (one trained on FD-only events, one on CD-only)
    and combines them at inference with:

        P_accept = 1 - (1 - P_FD) * (1 - P_CD)

    Smearing is delegated to whichever sub-model fired for each particle.  If
    both fire (rare in practice since FD/CD are nearly disjoint in theta), we
    weight by the per-event probabilities.

    Usage:
        pair = FastMCPair(model_fd="models/phi_v4/K+_FD.pt",
                          model_cd="models/phi_v4/K+_CD.pt")
        result = pair.simulate(p, theta, phi, vz)   # same API as FastMC
    """
    def __init__(self, model_fd, model_cd):
        self.fd = FastMC(model_file=model_fd)
        self.cd = FastMC(model_file=model_cd)
        # Calibration applies to the *combined* probability (after FD/CD merge).
        self.calibration_T = 1.0
        self.calibration_b = 0.0

    # Expose a few attributes for compatibility with code that pokes at FastMC
    @property
    def n_features(self):
        return self.fd.n_features

    def _calibrate(self, prob):
        if self.calibration_T == 1.0 and self.calibration_b == 0.0:
            return prob
        p = np.clip(prob, 1e-7, 1.0 - 1e-7)
        L = np.log(p / (1.0 - p))
        z = np.clip(self.calibration_T * L + self.calibration_b, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-z))

    @torch.no_grad()
    def accept(self, p, theta, phi, vz):
        """Bernoulli on combined acceptance probability."""
        p, theta, phi, vz = np.atleast_1d(p, theta, phi, vz)
        X_fd = self.fd._normalize(p, theta, phi, vz)
        X_cd = self.cd._normalize(p, theta, phi, vz)
        p_fd = torch.sigmoid(self.fd.acc_model(X_fd)).numpy()
        p_cd = torch.sigmoid(self.cd.acc_model(X_cd)).numpy()
        p_combined = 1.0 - (1.0 - p_fd) * (1.0 - p_cd)
        p_combined = self._calibrate(p_combined)
        return np.random.random(len(p_combined)) < p_combined

    @torch.no_grad()
    def smear(self, p, theta, phi, vz):
        """Smear using whichever subsystem 'wins' per event.

        We sample which subsystem (FD or CD) reconstructed each particle from
        P_fd / (P_fd + P_cd); then call that subsystem's MDN.  When the sum is
        zero (both predict 0), we fall back to FD by default — those events
        wouldn't pass accept() anyway, so it's harmless.
        """
        p, theta, phi, vz = np.atleast_1d(p, theta, phi, vz)
        X_fd = self.fd._normalize(p, theta, phi, vz)
        X_cd = self.cd._normalize(p, theta, phi, vz)
        p_fd = torch.sigmoid(self.fd.acc_model(X_fd)).numpy()
        p_cd = torch.sigmoid(self.cd.acc_model(X_cd)).numpy()

        denom = p_fd + p_cd
        weight_fd = np.where(denom > 0, p_fd / np.maximum(denom, 1e-9), 1.0)
        use_fd = np.random.random(len(weight_fd)) < weight_fd

        # Smear with each model, then pick per-event
        d_fd_norm = self.fd.mdn_model.sample(X_fd).numpy()
        d_cd_norm = self.cd.mdn_model.sample(X_cd).numpy()
        d_fd = d_fd_norm * self.fd.y_smear_std + self.fd.y_smear_mean
        d_cd = d_cd_norm * self.cd.y_smear_std + self.cd.y_smear_mean
        delta = np.where(use_fd[:, None], d_fd, d_cd)

        return (p + delta[:, 0], theta + delta[:, 1],
                phi + delta[:, 2], vz + delta[:, 3])

    def simulate(self, p, theta, phi, vz):
        p, theta, phi, vz = np.atleast_1d(p, theta, phi, vz)
        mask = self.accept(p, theta, phi, vz)
        p_rec, theta_rec, phi_rec, vz_rec = self.smear(p[mask], theta[mask], phi[mask], vz[mask])
        return {
            "accepted": mask,
            "p_mc": p[mask], "theta_mc": theta[mask], "phi_mc": phi[mask], "vz_mc": vz[mask],
            "p_rec": p_rec, "theta_rec": theta_rec, "phi_rec": phi_rec, "vz_rec": vz_rec,
        }


def load_model_auto(model_dir, particle_name):
    """Return a FastMC for `{name}.pt`, or a FastMCPair for `{name}_FD.pt` + `{name}_CD.pt`.

    If `<model_dir>/calibration.json` exists and contains a calibration entry for
    this particle, the (T, b) is attached to the returned model so that
    inference applies the temperature+bias correction automatically.

    Returns None if no model is found.
    """
    single = os.path.join(model_dir, f"{particle_name}.pt")
    fd     = os.path.join(model_dir, f"{particle_name}_FD.pt")
    cd     = os.path.join(model_dir, f"{particle_name}_CD.pt")
    if os.path.exists(fd) and os.path.exists(cd):
        m = FastMCPair(model_fd=fd, model_cd=cd)
    elif os.path.exists(single):
        m = FastMC(model_file=single)
    else:
        return None

    # Optionally attach calibration
    calib_path = os.path.join(model_dir, "calibration.json")
    if os.path.exists(calib_path):
        try:
            import json
            with open(calib_path) as f:
                cal = json.load(f)
            entry = cal.get("particles", {}).get(particle_name)
            if entry is not None:
                m.calibration_T = float(entry["T"])
                m.calibration_b = float(entry["b"])
        except Exception as exc:
            print(f"  (calibration load failed for {particle_name}: {exc})")
    return m


if __name__ == "__main__":
    fmc = FastMC()
    rng = np.random.default_rng(42)
    p   = rng.uniform(1, 11, 10000)
    th  = rng.uniform(5, 40, 10000)
    phi = rng.uniform(-40, 40, 10000)
    vz  = rng.uniform(-5.5, -0.5, 10000)

    result = fmc.simulate(p, th, phi, vz)
    n_acc = result["accepted"].sum()
    print(f"Generated: {len(p)},  Accepted: {n_acc} ({100*n_acc/len(p):.1f}%)")
    print(f"p_rec[:5]:     {result['p_rec'][:5]}")
    print(f"theta_rec[:5]: {result['theta_rec'][:5]}")
    print(f"phi_rec[:5]:   {result['phi_rec'][:5]}")
