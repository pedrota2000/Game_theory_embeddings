"""Stronger gradient check: compare on entries with the largest analytic gradient."""
import numpy as np
from pursuit_evasion_numpy import (
    TrainConfig, fourier_features, init_params,
    forward_dirs, backward_dirs, rollout, backward_rollout,
    forward_loss, backward_loss, sample_evader_ICs,
)


def make_setup(seed=42):
    # Use a setup where the evader starts close enough that the soft-capture
    # signal is appreciable -> gradients are not in the noise floor.
    cfg = TrainConfig(n_h=4, n_b=8, n_freqs=4, n_t=16,
                      t_max=1.5, L=2.0, eps=0.2, tau=0.05)
    rng = np.random.default_rng(seed)
    D = 2 * cfg.n_freqs + 1
    paramsE = init_params(cfg.n_h, cfg.n_b, D, rng)
    paramsP = init_params(cfg.n_h, cfg.n_b, D, rng)
    t_grid = np.linspace(0.0, cfg.t_max, cfg.n_t + 1)
    phi = fourier_features(t_grid, cfg.t_max, cfg.n_freqs)
    x_E0 = sample_evader_ICs(cfg.n_h, cfg.L, min_dist=0.15, seed=seed)
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
        return backward_dirs(paramsP, cP, duP)
    else:
        return backward_dirs(paramsE, cE, duE)


def check_player(who, target_player, n_pick=10, eps_fd=1e-4):
    cfg, paramsE, paramsP, phi, x_E0 = make_setup()
    g_an = analytic_grad(paramsE, paramsP, phi, x_E0, cfg, who)
    params = paramsP if target_player == "P" else paramsE
    print(f"\n[{who}] base loss = {loss_for(paramsE, paramsP, phi, x_E0, cfg, who):.6f}")
    for key in ["W", "B", "b_body"]:
        arr = params[key]
        gn = g_an[key]
        # pick the n_pick indices with largest |gn|
        flat = np.abs(gn).ravel()
        top = np.argsort(flat)[::-1][:n_pick]
        print(f"\n  param '{key}':  top |grad| entries")
        print(f"    {'idx':30s}  {'analytic':>13s}  {'finite_diff':>13s}  {'rel_err':>9s}")
        for ii in top:
            idx = np.unravel_index(ii, gn.shape)
            orig = arr[idx]
            arr[idx] = orig + eps_fd
            Lp = loss_for(paramsE, paramsP, phi, x_E0, cfg, who)
            arr[idx] = orig - eps_fd
            Lm = loss_for(paramsE, paramsP, phi, x_E0, cfg, who)
            arr[idx] = orig
            fd = (Lp - Lm) / (2 * eps_fd)
            an = gn[idx]
            rel = abs(fd - an) / (abs(fd) + abs(an) + 1e-12)
            print(f"    {str(idx):30s}  {an:13.5e}  {fd:13.5e}  {rel:9.3e}")


if __name__ == "__main__":
    check_player("P", "P")
    check_player("E", "E")
