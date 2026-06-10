#!/usr/bin/env python3
"""Train a single particle's fast MC model from .dat training data.

Usage:
  python train_single_particle.py phi_train.dat -o models/phi_v2 --particle_index 0
"""

import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from scipy.optimize import brentq

PDG_TO_SHORT = {
    11: 'e-', -11: 'e+', 13: 'mu-', -13: 'mu+',
    2212: 'p', 2112: 'n',
    211: 'pi+', -211: 'pi-',
    321: 'K+', -321: 'K-',
    22: 'gamma',
}


def read_single_particle(filename, particle_index, max_events=0, subsystem='all',
                         min_status=1):
    """Read .dat file and extract data for a single particle index.

    subsystem:
        'all'  — keep all events (default; status=0 mapped to negative target).
        'FD'   — keep only events where det==1 (rejected events also kept as
                 'not accepted in FD' so the acceptance classifier still has
                 negative examples).
        'CD'   — same, but kept as 'not accepted in CD'.

    min_status:
        1 — any match (charge-sign only, PID-blind)
        2 — require PID match

    For FD/CD modes, the binary label `accepted` becomes:
        1 if status>=min_status AND det == target_det
        0 otherwise
    """
    if subsystem not in ('all', 'FD', 'CD'):
        raise ValueError(f"subsystem must be 'all'|'FD'|'CD', got {subsystem!r}")
    target_det = {'all': None, 'FD': 1, 'CD': 2}[subsystem]

    header = {}
    mc_p, mc_theta, mc_phi, mc_vz = [], [], [], []
    det_list = []
    accepted = []
    delta_p, delta_theta, delta_phi, delta_vz = [], [], [], []
    pid = None

    current_event_started = False
    particle_count = 0
    n_events = 0
    n_old_format = 0      # counter for missing det column (4-feature .dat compat)

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

            # Particle line
            if line[0] == ' ':
                if not current_event_started:
                    continue

                if particle_count == particle_index:
                    parts = line.split()
                    status = int(parts[0])
                    p_pid = int(parts[1])

                    # Detect format: new compact has det as 3rd col (int),
                    # old format has p_gen as 3rd col (float with '.').
                    if '.' not in parts[2]:
                        # ── New compact format ──
                        # status pid det p_gen theta_gen phi_gen vz_gen [p_rec ...]
                        det = int(parts[2])
                        p_gen   = float(parts[3])
                        t_gen   = float(parts[4])
                        phi_gen = float(parts[5])
                        vz_gen  = float(parts[6])
                        if status > 0 and len(parts) >= 11:
                            p_rec   = float(parts[7])
                            t_rec   = float(parts[8])
                            phi_rec = float(parts[9])
                            vz_rec  = float(parts[10])
                        else:
                            p_rec = t_rec = phi_rec = vz_rec = -999.0
                    else:
                        # ── Old format ──
                        # status pid p_gen theta_gen phi_gen vz_gen p_rec ... [det]
                        p_gen   = float(parts[2])
                        t_gen   = float(parts[3])
                        phi_gen = float(parts[4])
                        vz_gen  = float(parts[5])
                        p_rec   = float(parts[6])
                        t_rec   = float(parts[7])
                        phi_rec = float(parts[8])
                        vz_rec  = float(parts[9])
                        if len(parts) >= 11:
                            det = int(parts[10])
                        else:
                            det = 1 if status >= 1 else 0
                            n_old_format += 1

                    if pid is None:
                        pid = p_pid

                    # Decide acceptance label for this subsystem mode.
                    if target_det is None:
                        is_accepted = (status >= min_status)
                    else:
                        is_accepted = (status >= min_status) and (det == target_det)

                    mc_p.append(p_gen)
                    mc_theta.append(t_gen)
                    mc_phi.append(phi_gen)
                    mc_vz.append(vz_gen)
                    det_list.append(det)
                    accepted.append(1 if is_accepted else 0)

                    if is_accepted:
                        delta_p.append(p_rec - p_gen)
                        delta_theta.append(t_rec - t_gen)
                        dp = phi_rec - phi_gen
                        if dp > 180: dp -= 360
                        if dp < -180: dp += 360
                        delta_phi.append(dp)
                        delta_vz.append(vz_rec - vz_gen)
                    else:
                        delta_p.append(0.0)
                        delta_theta.append(0.0)
                        delta_phi.append(0.0)
                        delta_vz.append(0.0)

                particle_count += 1
                continue

            # Event line
            current_event_started = True
            particle_count = 0
            n_events += 1

            if max_events > 0 and n_events > max_events:
                break

    if n_old_format:
        print(f"  NOTE: {n_old_format} particle lines without det column "
              f"(treated as FD when accepted).")

    name = PDG_TO_SHORT.get(pid, str(pid))
    return header, {
        'pid': pid,
        'name': name,
        'mc_p': np.array(mc_p, dtype=np.float32),
        'mc_theta': np.array(mc_theta, dtype=np.float32),
        'mc_phi': np.array(mc_phi, dtype=np.float32),
        'mc_vz': np.array(mc_vz, dtype=np.float32),
        'det': np.array(det_list, dtype=np.int8),
        'accepted': np.array(accepted, dtype=np.int8),
        'delta_p': np.array(delta_p, dtype=np.float32),
        'delta_theta': np.array(delta_theta, dtype=np.float32),
        'delta_phi': np.array(delta_phi, dtype=np.float32),
        'delta_vz': np.array(delta_vz, dtype=np.float32),
    }, n_events


N_FEATURES = 6   # p, theta, phi, vz, sin(6*phi), cos(6*phi)


def phi_features(phi_deg):
    """sin(6φ), cos(6φ) — sector-periodic Fourier features."""
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
    def __init__(self, n_gauss=5, n_out=4, n_in=N_FEATURES):
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
        sigma = torch.exp(params[:, self.n_gauss*(1+self.n_out):].clamp(-5, 2)).view(-1, self.n_gauss, self.n_out)
        return pi, mu, sigma

    def sample(self, x):
        pi, mu, sigma = self.forward(x)
        comp = torch.multinomial(pi, 1).squeeze(-1)
        batch_idx = torch.arange(len(x), device=x.device)
        return mu[batch_idx, comp] + sigma[batch_idx, comp] * torch.randn_like(mu[batch_idx, comp])


def mdn_loss(pi, mu, sigma, y):
    y = y.unsqueeze(1).expand_as(mu)
    log_prob = -0.5 * ((y - mu) / sigma)**2 - torch.log(sigma) - 0.5 * np.log(2 * np.pi)
    log_prob = log_prob.sum(dim=2)
    log_pi = torch.log(pi + 1e-8)
    return -torch.logsumexp(log_pi + log_prob, dim=1).mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", help="Training data file (.dat)")
    p.add_argument("-o", "--output", required=True, help="Output directory")
    p.add_argument("--particle_index", type=int, required=True, help="Particle index (0-3)")
    p.add_argument("--model_name", default=None,
                   help="Override the saved model name (default: species name, "
                        "e.g. 'e-'). Needed when the same species appears twice "
                        "in the final state — J/psi has two e-: index 0 (FT, "
                        "scattered) and index 3 (FD, decay) — to avoid both "
                        "writing 'e-.pt'.")
    p.add_argument("--subsystem", choices=["all", "FD", "CD"], default="all",
                   help="Detector subsystem to train on (all = no split).  "
                        "Saved model is named e.g. 'K+_FD.pt' for non-'all' modes.")
    p.add_argument("--max_events", type=int, default=0)
    p.add_argument("--max_train", type=int, default=10_000_000)
    p.add_argument("--n_gauss", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--acc_epochs", type=int, default=50)
    p.add_argument("--mdn_epochs", type=int, default=300)
    p.add_argument("--mdn_patience", type=int, default=30)
    p.add_argument("--min_status", type=int, default=1,
                   help="Minimum status for 'detected' (1=any match, 2=PID match)")
    p.add_argument("--matching_cuts", default="matching_cuts_phi.json",
                   help="JSON of momentum-dependent resolution windows. "
                        "Used to apply Nσ sanity cut on Δp/Δθ/Δφ before MDN training "
                        "(rejects TruthMatch tracks with reconstruction pathologies). "
                        "Set to '' to disable.")
    p.add_argument("--sanity_nsigma", type=float, default=5.0,
                   help="Nσ for the sanity cut on top of TruthMatch (default 5).")
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Read data for this particle only
    print(f"Reading particle index {args.particle_index} from {args.input} "
          f"[subsystem={args.subsystem}, min_status={args.min_status}] ...")
    sys.stdout.flush()
    t0 = time.time()
    header, data, n_events = read_single_particle(
        args.input, args.particle_index,
        max_events=args.max_events, subsystem=args.subsystem,
        min_status=args.min_status)
    t_read = time.time() - t0

    # name = saved-model name (may be overridden); data['name'] stays the
    # species name and is still used for the matching-cuts JSON lookup.
    name = args.model_name or data['name']
    pid = data['pid']
    n_acc = data['accepted'].sum()
    print(f"  {name} (pid={pid}): {len(data['mc_p']):,} events, "
          f"{n_acc:,} accepted ({100*n_acc/len(data['mc_p']):.1f}%) "
          f"[read in {t_read:.1f}s]")
    sys.stdout.flush()

    mc_p = data['mc_p']
    mc_theta = data['mc_theta']
    mc_phi = data['mc_phi']
    mc_vz = data['mc_vz']
    det = data['det']
    accepted = data['accepted']
    delta_p = data['delta_p']
    delta_theta = data['delta_theta']
    delta_phi = data['delta_phi']
    delta_vz = data['delta_vz']

    # Subsample
    if args.max_train > 0 and len(mc_p) > args.max_train:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(mc_p), args.max_train, replace=False)
        mc_p = mc_p[idx]; mc_theta = mc_theta[idx]
        mc_phi = mc_phi[idx]; mc_vz = mc_vz[idx]
        det = det[idx]
        accepted = accepted[idx]
        delta_p = delta_p[idx]; delta_theta = delta_theta[idx]
        delta_phi = delta_phi[idx]; delta_vz = delta_vz[idx]
        print(f"  Subsampled to {args.max_train:,} events, "
              f"{accepted.sum():,} accepted ({100*accepted.mean():.1f}%)")
        sys.stdout.flush()

    # 6-feature input: kinematics + sector-periodic Fourier basis.
    sin6phi, cos6phi = phi_features(mc_phi)
    X = np.column_stack([mc_p, mc_theta, mc_phi, mc_vz, sin6phi, cos6phi]).astype(np.float32)
    y_acc = accepted.astype(np.float32)
    y_smear = np.column_stack([delta_p, delta_theta, delta_phi, delta_vz])

    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    # Guard against zero-variance features (shouldn't happen but cheap).
    X_std = np.where(X_std > 1e-6, X_std, 1.0)
    X_norm = (X - X_mean) / X_std

    # ── Acceptance classifier ──────────────────────────────────────
    print(f"\n  Training acceptance classifier ({args.acc_epochs} epochs)...")
    sys.stdout.flush()

    X_train, X_test, y_train, y_test = train_test_split(
        X_norm, y_acc, test_size=0.2, random_state=42)

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size)

    acc_model = AcceptanceNet().to(device)
    optimizer = torch.optim.Adam(acc_model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    t0 = time.time()
    for epoch in range(args.acc_epochs):
        acc_model.train()
        train_loss = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(acc_model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_ds)

        if (epoch + 1) % 10 == 0:
            acc_model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for xb, yb in test_dl:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = (torch.sigmoid(acc_model(xb)) > 0.5).float()
                    correct += (pred == yb).sum().item()
                    total += len(yb)
            print(f"    epoch {epoch+1:3d}  loss={train_loss:.4f}  "
                  f"test acc={100*correct/total:.2f}%")
            sys.stdout.flush()

    t_acc = time.time() - t0
    print(f"  Acceptance time: {t_acc:.1f} s")
    sys.stdout.flush()

    # Calibration
    acc_model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            all_logits.append(acc_model(xb.to(device)).cpu())
            all_labels.append(yb)
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    logits_np = all_logits.numpy()
    true_acc = all_labels.mean().item()

    def acc_at_offset(offset):
        return (1 / (1 + np.exp(np.clip(-(logits_np + offset), -500, 500)))).mean() - true_acc

    bias_offset = brentq(acc_at_offset, -5, 5)
    print(f"  Calibration: bias_offset={bias_offset:.5f}")
    sys.stdout.flush()

    # ── Smearing MDN ───────────────────────────────────────────────
    mask = accepted == 1
    X_smear = X_norm[mask]
    y_smear_acc = y_smear[mask]
    n_acc = mask.sum()
    print(f"\n  Training smearing MDN ({n_acc:,} accepted events, {args.mdn_epochs} max epochs)...")
    sys.stdout.flush()

    if n_acc < 100:
        print(f"  WARNING: too few accepted events ({n_acc}), skipping")
        return

    # ── p-dependent Nσ sanity cut on top of TruthMatch ────────────
    # TruthMatch (MC::RecMatch quality>0.98) matches by hit sharing,
    # so it occasionally lets through tracks whose reconstructed
    # momentum is pathologically wrong (rec_p drifting to 100s of GeV
    # from track-fit failures). Apply the same momentum/angle-dependent
    # resolution windows used for PID-blind matching, at Nσ
    # (default 5) — anything beyond Nσ from the calibrated detector
    # resolution is reconstruction pathology, not real smearing.
    if args.matching_cuts and os.path.exists(args.matching_cuts):
        try:
            from matching_cuts import MatchingCuts
            mc_cuts = MatchingCuts(args.matching_cuts)
            mc_cuts.n_sigma = args.sanity_nsigma                # override
        except Exception as e:
            print(f"  WARNING: could not load matching cuts ({e}), skipping sanity cut")
            mc_cuts = None
    else:
        mc_cuts = None

    if mc_cuts is not None and data['name'] in mc_cuts.particles:
        # Get per-event MC kinematics and det for the accepted subset.
        # NOTE: use local subsampled arrays (mc_p, mc_theta, det) — NOT data['...']
        # which is still the un-subsampled full array.
        p_acc   = mc_p[mask]
        th_acc  = mc_theta[mask]
        det_acc = det[mask]

        # Split into FD (det==1) and CD (det==2) for separate per-event windows
        keep = np.ones(len(y_smear_acc), dtype=bool)
        for det_int, det_name in [(1, 'FD'), (2, 'CD'), (3, 'FT')]:
            sel = (det_acc == det_int)
            if not sel.any() or det_name not in mc_cuts.data['particles'][data['name']]:
                continue
            (dp_lo, dp_hi), (dt_lo, dt_hi), (dphi_lo, dphi_hi) = mc_cuts.window_arrays(
                data['name'], det_name, p_acc[sel], th_acc[sel])
            sub = y_smear_acc[sel]
            ok = ((sub[:, 0] > dp_lo)   & (sub[:, 0] < dp_hi)   &
                  (sub[:, 1] > dt_lo)   & (sub[:, 1] < dt_hi)   &
                  (sub[:, 2] > dphi_lo) & (sub[:, 2] < dphi_hi))
            # Map back to global indices
            global_idx = np.where(sel)[0]
            keep[global_idx[~ok]] = False
            print(f"  {data['name']} {det_name}: kept {ok.sum():,} / {sel.sum():,} "
                  f"({100*ok.sum()/max(sel.sum(),1):.4f}%) after {args.sanity_nsigma}σ cut")

        n_drop = (~keep).sum()
        print(f"  Total rejected by {args.sanity_nsigma}σ p-dependent cut: "
              f"{n_drop:,} / {n_acc:,} ({100*n_drop/n_acc:.4f}%)")
        y_smear_acc = y_smear_acc[keep]
        X_smear     = X_smear[keep]
    else:
        print(f"  (no JSON cuts loaded for {data['name']}, skipping sanity cut)")

    y_mean = y_smear_acc.mean(axis=0)
    y_std  = y_smear_acc.std(axis=0)
    y_norm = (y_smear_acc - y_mean) / y_std

    X_tr, X_te, y_tr, y_te = train_test_split(X_smear, y_norm, test_size=0.2, random_state=42)
    train_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
    test_ds = TensorDataset(torch.tensor(X_te), torch.tensor(y_te))
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size)

    mdn_model = SmearingMDN(n_gauss=args.n_gauss).to(device)
    optimizer = torch.optim.Adam(mdn_model.parameters(), lr=1e-3)

    best_test_loss = float("inf")
    best_state = None
    wait = 0

    t0 = time.time()
    for epoch in range(args.mdn_epochs):
        mdn_model.train()
        train_loss = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pi, mu, sigma = mdn_model(xb)
            loss = mdn_loss(pi, mu, sigma, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mdn_model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_ds)

        mdn_model.eval()
        test_loss = 0
        with torch.no_grad():
            for xb, yb in test_dl:
                xb, yb = xb.to(device), yb.to(device)
                pi, mu, sigma = mdn_model(xb)
                test_loss += mdn_loss(pi, mu, sigma, yb).item() * len(xb)
        test_loss /= len(test_ds)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_state = {k: v.cpu().clone() for k, v in mdn_model.state_dict().items()}
            wait = 0
            marker = " *"
        else:
            wait += 1
            marker = ""

        if (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1:3d}  train={train_loss:.4f}  test={test_loss:.4f}"
                  f"  best={best_test_loss:.4f}{marker}")
            sys.stdout.flush()

        if wait >= args.mdn_patience:
            print(f"    Early stopping at epoch {epoch+1} (best test={best_test_loss:.4f})")
            sys.stdout.flush()
            break

    mdn_model.load_state_dict(best_state)
    mdn_model.to(device)
    t_mdn = time.time() - t0
    print(f"  MDN time: {t_mdn:.1f} s")
    sys.stdout.flush()

    # Save  (name includes subsystem suffix when not 'all')
    if args.subsystem == "all":
        model_file = os.path.join(args.output, f"{name}.pt")
    else:
        model_file = os.path.join(args.output, f"{name}_{args.subsystem}.pt")
    torch.save({
        "acc_model": acc_model.state_dict(),
        "mdn_model": mdn_model.state_dict(),
        "X_mean": torch.tensor(X_mean),
        "X_std": torch.tensor(X_std),
        "y_smear_mean": torch.tensor(y_mean),
        "y_smear_std": torch.tensor(y_std),
        "bias_offset": torch.tensor(bias_offset),
        "n_gauss": args.n_gauss,
        "n_features": N_FEATURES,
        "pid": pid,
        "particle_name": name,
        "subsystem": args.subsystem,
        "reaction": header.get('reaction', 'unknown'),
    }, model_file)
    print(f"\n  Saved: {model_file}")
    print(f"  Total time: {t_acc + t_mdn + t_read:.1f} s")


if __name__ == "__main__":
    main()
