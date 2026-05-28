"""Finite-difference verification of the hand-coded gradients."""
import numpy as np
from pursuit_evasion_numpy import (
    TrainConfig, fourier_features, init_params,
    forward_dirs, backward_dirs, rollout, backward_rollout,
    forward_loss, backward_loss, sample_evader_ICs,
)


def make_setup(seed=42, n_h=4, n_b=8, n_freqs=6, n_t=20):
    cfg = TrainConfig(n_h=n_h, n_b=n_b, n_freqs=n_freqs, n_t=n_t,
                      t_max=2.0, L=2.0, eps=0.1, tau=0.01)
    rng = np.random.default_rng(seed)
    D = 2 * cfg.n_freqs + 1
    paramsE = init_params(cfg.n_h, cfg.n_b, D, rng)
    paramsP = init_params(cfg.n_h, cfg.n_b, D, rng)
    t_grid = np.linspace(0.0, cfg.t_max, cfg.n_t + 1)
    phi = fourier_features(t_grid, cfg.t_max, cfg.n_freqs)
    x_E0 = sample_evader_ICs(cfg.n_h, cfg.L, min_dist=0.3, seed=seed)
    return cfg, paramsE, paramsP, phi, x_E0


def loss_for(paramsE, paramsP, phi, x_E0, cfg, who):
    uE, _ = forward_dirs(paramsE, phi)
    uP, _ = forward_dirs(paramsP, phi)
    xE, xP = rollout(uE, uP, x_E0, cfg.L, cfg.v_E, cfg.v_P, cfg.t_max, cfg.n_t)
    L_E, L_P, _ = forward_loss(xE, xP, cfg.L, cfg.eps, cfg.tau, cfg.t_max,
                               cfg.n_t, cfg.no_capture_penalty)
    return L_P if who == "P" else L_E


def analytic_grad(paramsE, paramsP, phi, x_E0, cfg, who):
    uE, cE = forward_dirs(paramsE, phi)
    uP, cP = forward_dirs(paramsP, phi)
    xE, xP = rollout(uE, uP, x_E0, cfg.L, cfg.v_E, cfg.v_P, cfg.t_max, cfg.n_t)
    L_E, L_P, cL = forward_loss(xE, xP, cfg.L, cfg.eps, cfg.tau, cfg.t_max,
                                cfg.n_t, cfg.no_capture_penalty)
    is_p = (who == "P")
    dxE, dxP = backward_loss(1.0, cL, cfg.n_h, cfg.no_capture_penalty,
                             cfg.t_max, is_pursuer=is_p)
    duE, duP = backward_rollout(dxE, dxP, cfg.L, cfg.v_E, cfg.v_P, cfg.t_max,
                                cfg.n_t)
    if is_p:
        return backward_dirs(paramsP, cP, duP), "P"
    else:
        return backward_dirs(paramsE, cE, duE), "E"


def fd_grad(paramsE, paramsP, phi, x_E0, cfg, who, target_player, key,
            n_samples=15, eps_fd=1e-5, seed=0):
    """Check finite-difference gradient at randomly chosen indices."""
    params = paramsP if target_player == "P" else paramsE
    arr = params[key]
    rng = np.random.default_rng(seed)
    rows = []
    idxs = []
    for _ in range(n_samples):
        idx = tuple(rng.integers(0, s) for s in arr.shape)
        orig = arr[idx]
        arr[idx] = orig + eps_fd
        Lp = loss_for(paramsE, paramsP, phi, x_E0, cfg, who)
        arr[idx] = orig - eps_fd
        Lm = loss_for(paramsE, paramsP, phi, x_E0, cfg, who)
        arr[idx] = orig
        rows.append((idx, (Lp - Lm) / (2 * eps_fd)))
        idxs.append(idx)
    return rows


def check(who, target_player, key):
    cfg, paramsE, paramsP, phi, x_E0 = make_setup()
    g_an, _ = analytic_grad(paramsE, paramsP, phi, x_E0, cfg, who)
    if (target_player == "P" and who != "P") or (target_player == "E" and who != "E"):
        # only the player being updated has analytic gradients in our setup
        return None
    rows = fd_grad(paramsE, paramsP, phi, x_E0, cfg, who, target_player, key)
    abs_errs, rel_errs = [], []
    for idx, fd_val in rows:
        an_val = g_an[key][idx]
        abs_err = abs(fd_val - an_val)
        rel_err = abs_err / (abs(fd_val) + abs(an_val) + 1e-9)
        abs_errs.append(abs_err); rel_errs.append(rel_err)
    return (max(abs_errs), max(rel_errs))


if __name__ == "__main__":
    print(f"{'side':4s} {'param':>7s} {'max|abs|':>12s} {'max|rel|':>12s}")
    for who, target, label in [
        ("P", "P", "Pursuer"),
        ("E", "E", "Evader"),
    ]:
        for key in ["W", "B", "b_body"]:
            r = check(who, target, key)
            if r is None:
                continue
            ae, re = r
            ok = "✓" if re < 5e-3 else "✗"
            print(f"{label:7s} {key:>7s} {ae:12.3e} {re:12.3e}  {ok}")
