"""
Standalone numpy implementation of the multi-head adversarial pursuit-evasion
network on the 2-torus.  Hand-coded forward and backward passes (no autograd
framework available in this sandbox).  This is the script that produces the
figures used in the accompanying LaTeX write-up.

Architecture
------------
For each player α ∈ {E, P}:

    phi(t) = [sin(ω_k t), cos(ω_k t), t / t_max]_{k=1..K}     (Fourier features)
    z_body(t) = B_α phi(t) + b_α                              (shared body)
    H_α(t)    = tanh(z_body(t))                               (latent embedding)
    z^j_α(t)  = W^j_α H_α(t)                                  (j-th linear head)
    û^j_α(t)  = z^j_α(t) / || z^j_α(t) ||                     (unit direction)
    v^j_α(t)  = ||v_α|| · û^j_α(t)                            (rescaled velocity)

The N_h heads correspond to N_h different initial conditions of the evader.

Dynamics on T^2 = [0, L)^2:
    x^j_E(t+dt) = ( x^j_E(t) + dt·v_E·û^j_E(t) ) mod L
    x^j_P(t+dt) = ( x^j_P(t) + dt·v_P·û^j_P(t) ) mod L
    x^j_P(0) = (0, 0),   x^j_E(0) sampled away from origin

Loss (differentiable expected first-capture time)
-------------------------------------------------
    d^j_k    = || x^j_E(t_k) - x^j_P(t_k) ||_T                (torus distance)
    p^j_k    = σ((ε² - (d^j_k)²)/τ)                           (smooth capture)
    S^j_k    = ∏_{l ≤ k}(1 - p^j_l)                           (survival)
    E[T]^j   ≈ Δt · trapezoidal_∫_0^{t_max} S^j(t) dt
    L_P      = mean_j ( E[T]^j + λ · S^j_{N_t} · t_max )
    L_E      = -mean_j E[T]^j

Two Adam optimizers update the pursuer and evader networks alternately.
"""
from __future__ import annotations

import math
import os
import json
import time
from dataclasses import dataclass, asdict, field

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Fourier features and parameter initialisation
# =============================================================================

def fourier_features(t: np.ndarray, t_max: float, n_freqs: int) -> np.ndarray:
    """phi(t) of shape (T, 2*n_freqs + 1)."""
    omegas = (2.0 * np.pi / t_max) * np.arange(1, n_freqs + 1)
    ang = t[:, None] * omegas[None, :]
    return np.concatenate(
        [np.sin(ang), np.cos(ang), (t[:, None] / t_max)], axis=1
    )


def init_params(n_h: int, n_b: int, D: int, rng: np.random.Generator) -> dict:
    return {
        "B": rng.standard_normal((n_b, D)) * np.sqrt(1.5 / D),
        "b_body": np.zeros(n_b),
        "W": rng.standard_normal((n_h, 2, n_b)) * np.sqrt(1.0 / n_b),
    }


# =============================================================================
# Forward / backward through the direction network
# =============================================================================

def forward_dirs(params: dict, phi: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Returns
    -------
    u : (n_h, T, 2) unit directions
    cache : intermediates needed for the backward pass
    """
    B, b, W = params["B"], params["b_body"], params["W"]
    z_body = phi @ B.T + b                              # (T, n_b)
    H = np.tanh(z_body)                                 # (T, n_b)
    z_head = np.einsum("jab,tb->jta", W, H)             # (n_h, T, 2)
    norm_z = np.linalg.norm(z_head, axis=-1, keepdims=True).clip(min=1e-8)
    u = z_head / norm_z                                 # (n_h, T, 2)
    cache = {"phi": phi, "H": H, "z_body": z_body,
             "z_head": z_head, "u": u, "norm_z": norm_z, "W": W}
    return u, cache


def backward_dirs(params: dict, cache: dict, du: np.ndarray) -> dict:
    """
    Given dL/du of shape (n_h, T, 2), returns gradients of L w.r.t. params.
    """
    phi = cache["phi"]
    H = cache["H"]
    u = cache["u"]
    norm_z = cache["norm_z"]
    W = cache["W"]

    # u = z / ||z|| ; dz = (du - (u·du) u) / ||z||
    dot = (u * du).sum(axis=-1, keepdims=True)
    dz_head = (du - dot * u) / norm_z

    # z_head[j, t, a] = sum_b W[j, a, b] H[t, b]
    dW = np.einsum("jta,tb->jab", dz_head, H)
    dH = np.einsum("jab,jta->tb", W, dz_head)

    # H = tanh(z_body)
    dz_body = (1.0 - H ** 2) * dH

    # z_body = phi @ B^T + b
    dB = dz_body.T @ phi
    db_body = dz_body.sum(axis=0)
    return {"B": dB, "b_body": db_body, "W": dW}


# =============================================================================
# Trajectory rollout on the torus (forward + analytic backward)
# =============================================================================

def rollout(uE: np.ndarray,
            uP: np.ndarray,
            x_E0: np.ndarray,
            L: float,
            v_E: float,
            v_P: float,
            t_max: float,
            n_t: int) -> tuple[np.ndarray, np.ndarray]:
    """Explicit trapezoidal (midpoint-average) step with periodic wrap."""
    dt = t_max / n_t
    n_h = uE.shape[0]
    xE = np.empty((n_h, n_t + 1, 2))
    xP = np.empty_like(xE)
    xE[:, 0] = x_E0
    xP[:, 0] = 0.0
    for k in range(n_t):
        u_mid_E = 0.5 * (uE[:, k] + uE[:, k + 1])
        u_mid_P = 0.5 * (uP[:, k] + uP[:, k + 1])
        xE[:, k + 1] = (xE[:, k] + dt * v_E * u_mid_E) % L
        xP[:, k + 1] = (xP[:, k] + dt * v_P * u_mid_P) % L
    return xE, xP


def backward_rollout(dxE: np.ndarray,
                     dxP: np.ndarray,
                     L: float,
                     v_E: float,
                     v_P: float,
                     t_max: float,
                     n_t: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Given dL/dx_E and dL/dx_P (shape (n_h, n_t+1, 2)) coming from the loss,
    propagate gradients through the trajectory to get dL/du_E, dL/du_P.

    Returns
    -------
    duE, duP : (n_h, n_t+1, 2)
    """
    dt = t_max / n_t
    duE = np.zeros_like(dxE)
    duP = np.zeros_like(dxP)
    # dx_α[k+1] flows into both u_α[k] and u_α[k+1] (midpoint average)
    # and is also passed through to dx_α[k] (since the position carries over).
    grad_xE = dxE[:, -1].copy()
    grad_xP = dxP[:, -1].copy()
    for k in range(n_t - 1, -1, -1):
        # dL/d u_mid_α[k] = dt v_α · grad_x_α[k+1]
        du_mid_E = dt * v_E * grad_xE
        du_mid_P = dt * v_P * grad_xP
        # u_mid_α[k] = 0.5 (u_α[k] + u_α[k+1])
        duE[:, k]     += 0.5 * du_mid_E
        duE[:, k + 1] += 0.5 * du_mid_E
        duP[:, k]     += 0.5 * du_mid_P
        duP[:, k + 1] += 0.5 * du_mid_P
        # carry: x_α[k+1] also depends linearly on x_α[k] → grad passes through
        # plus accumulate the boundary gradient at this step (from loss term)
        grad_xE = grad_xE + dxE[:, k]
        grad_xP = grad_xP + dxP[:, k]
    return duE, duP


# =============================================================================
# Loss (forward + backward)
# =============================================================================

def forward_loss(xE: np.ndarray,
                 xP: np.ndarray,
                 L: float,
                 eps: float,
                 tau: float,
                 t_max: float,
                 n_t: int,
                 no_cap_pen: float,
                 lam_d: float = 1.0) -> tuple[float, float, dict]:
    """
    Composite loss with three terms.

    1. Soft expected first-capture-time  E[T]    (only sharp near capture)
    2. No-capture penalty                S_final
    3. Distance-shaping ⟨d⟩  --  always-on adversarial signal so the
       gradient never vanishes far from capture: lam_d * mean_{t,h} d(t,h).

    L_P =  E[T] + λ_nc · S(t_max) · t_max + λ_d · ⟨d⟩
    L_E = -E[T]                              -  λ_d · ⟨d⟩
    (the distance term is zero-sum)
    """
    dt = t_max / n_t
    disp = (xE - xP + L / 2.0) % L - L / 2.0          # (n_h, T, 2)
    d2 = (disp ** 2).sum(axis=-1)                      # (n_h, T)
    d = np.sqrt(d2 + 1e-12)                            # (n_h, T)
    arg = (eps ** 2 - d2) / tau
    log_one_minus_p = -np.logaddexp(0.0, arg)
    p = 1.0 - np.exp(log_one_minus_p)
    log_S = np.cumsum(log_one_minus_p, axis=1)
    S = np.exp(log_S)
    ET = dt * (0.5 * S[:, 0] + S[:, 1:-1].sum(axis=1) + 0.5 * S[:, -1])
    no_cap_term = no_cap_pen * S[:, -1] * t_max
    d_mean = d.mean(axis=1)                            # (n_h,)
    L_P = float((ET + no_cap_term + lam_d * d_mean).mean())
    L_E = float((-ET - lam_d * d_mean).mean())
    cache = {"disp": disp, "d2": d2, "d": d, "arg": arg, "p": p, "S": S,
             "dt": dt, "tau": tau, "ET": ET, "lam_d": lam_d}
    return L_E, L_P, cache


def backward_loss(d_loss: float,
                  cache: dict,
                  n_h: int,
                  no_cap_pen: float,
                  t_max: float,
                  is_pursuer: bool) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (dxE, dxP), each (n_h, T, 2), as the gradient of the player's loss
    w.r.t. the trajectory positions.
    """
    S = cache["S"]; p = cache["p"]; disp = cache["disp"]; d = cache["d"]
    dt = cache["dt"]; tau = cache["tau"]; lam_d = cache["lam_d"]
    T = S.shape[1]

    # ----- gradient through E[T] (and -E[T] for evader) -----
    sign = +1.0 if is_pursuer else -1.0
    dET = sign * d_loss * np.ones(n_h) / n_h
    dS = np.zeros_like(S)
    dS[:, 0] = 0.5 * dt * dET
    if T > 2:
        dS[:, 1:-1] = (dt * dET)[:, None]
    dS[:, -1] = 0.5 * dt * dET
    if is_pursuer:
        dS[:, -1] += no_cap_pen * t_max * d_loss / n_h
    d_logS = dS * S
    d_log_one_minus_p = np.cumsum(d_logS[:, ::-1], axis=1)[:, ::-1]
    d_arg = -p * d_log_one_minus_p
    d_d2 = -d_arg / tau                          # contribution from E[T] term

    # ----- gradient through λ_d ⟨d⟩ distance shaping (zero-sum) -----
    # L contains  sign · lam_d · mean_t d(t,h)  per head, with mean over heads.
    # dL/d(d(t,h)) = sign · lam_d / (n_h · T)
    # d/d(d2)  = (1/(2 d)) · dL/d(d)
    d_d_mean = sign * lam_d * d_loss / (n_h * T) * np.ones_like(d)   # (n_h, T)
    d_d2 = d_d2 + d_d_mean * 0.5 / d

    # ----- chain through disp -----
    d_disp = 2.0 * disp * d_d2[..., None]
    dxE = d_disp.copy()
    dxP = -d_disp.copy()
    return dxE, dxP


# =============================================================================
# Adam optimiser
# =============================================================================

class Adam:
    def __init__(self, params: dict, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params: dict, grads: dict, sign: float = -1.0) -> None:
        self.t += 1
        for k in params:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            m_hat = self.m[k] / (1 - self.b1 ** self.t)
            v_hat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] += sign * self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrainConfig:
    n_iters: int = 12000
    n_h: int = 12
    n_b: int = 16
    n_freqs: int = 12
    t_max: float = 4.0
    n_t: int = 64
    v_E: float = 1.0
    v_P: float = 1.2
    eps: float = 0.08
    tau: float = 5e-3
    L: float = 2.0
    lr_E: float = 3e-3
    lr_P: float = 1e-2
    no_capture_penalty: float = 3.0
    lam_d: float = 5.0   # distance-shaping zero-sum coefficient
    pretrain_P_iters: int = 800  # pretrain pursuer vs fixed evader first
    n_P_steps: int = 3
    n_E_steps: int = 1
    log_every: int = 200
    seed: int = 0


# =============================================================================
# IC sampling
# =============================================================================

def sample_evader_ICs(n_h: int, L: float, min_dist: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n_h:
        p = rng.uniform(0.0, L, size=2)
        dx = min(p[0], L - p[0])
        dy = min(p[1], L - p[1])
        if math.sqrt(dx * dx + dy * dy) >= min_dist:
            pts.append(p)
    return np.array(pts)


# =============================================================================
# Training loop
# =============================================================================

def train(cfg: TrainConfig, out_dir: str):
    rng = np.random.default_rng(cfg.seed)
    D = 2 * cfg.n_freqs + 1
    paramsE = init_params(cfg.n_h, cfg.n_b, D, rng)
    paramsP = init_params(cfg.n_h, cfg.n_b, D, rng)
    optE = Adam(paramsE, lr=cfg.lr_E)
    optP = Adam(paramsP, lr=cfg.lr_P)

    t_grid = np.linspace(0.0, cfg.t_max, cfg.n_t + 1)
    phi = fourier_features(t_grid, cfg.t_max, cfg.n_freqs)
    x_E0 = sample_evader_ICs(cfg.n_h, cfg.L, min_dist=0.4, seed=cfg.seed)

    history = {"iter": [], "L_E": [], "L_P": [], "ET": [],
               "final_d": [], "no_cap_frac": []}

    def step(update_who: str):
        # Forward
        uE, cacheE = forward_dirs(paramsE, phi)
        uP, cacheP = forward_dirs(paramsP, phi)
        xE, xP = rollout(uE, uP, x_E0, cfg.L, cfg.v_E, cfg.v_P, cfg.t_max, cfg.n_t)
        L_E_val, L_P_val, cacheL = forward_loss(
            xE, xP, cfg.L, cfg.eps, cfg.tau, cfg.t_max, cfg.n_t,
            cfg.no_capture_penalty, lam_d=cfg.lam_d)
        # Diagnostics
        diag = {
            "ET_mean": float(cacheL["ET"].mean()),
            "final_dist_mean": float(np.sqrt(cacheL["d2"][:, -1]).mean()),
            "no_capture_frac": float((cacheL["S"][:, -1] > 0.5).mean()),
        }
        # Backward — only for the player being updated
        is_pursuer = (update_who == "P")
        dxE, dxP = backward_loss(1.0, cacheL, cfg.n_h, cfg.no_capture_penalty,
                                 cfg.t_max, is_pursuer=is_pursuer)
        duE, duP = backward_rollout(dxE, dxP, cfg.L, cfg.v_E, cfg.v_P,
                                    cfg.t_max, cfg.n_t)
        if is_pursuer:
            gradsP = backward_dirs(paramsP, cacheP, duP)
            optP.step(paramsP, gradsP, sign=-1.0)
        else:
            gradsE = backward_dirs(paramsE, cacheE, duE)
            optE.step(paramsE, gradsE, sign=-1.0)
        return L_E_val, L_P_val, diag

    t0 = time.time()
    # ---- optional pursuer pre-training against the random initial evader ----
    for it in range(cfg.pretrain_P_iters):
        L_E_val, L_P_val, diag = step("P")
        if it % max(cfg.pretrain_P_iters // 4, 1) == 0:
            print(f"[pretrain-P] it {it:4d}  L_P={L_P_val:.4f}  "
                  f"E[T]={diag['ET_mean']:.3f}  "
                  f"final_d={diag['final_dist_mean']:.3f}")

    for it in range(cfg.n_iters):
        for _ in range(cfg.n_P_steps):
            L_E_val, L_P_val, diag = step("P")
        for _ in range(cfg.n_E_steps):
            L_E_val, L_P_val, diag = step("E")
        if it % cfg.log_every == 0 or it == cfg.n_iters - 1:
            history["iter"].append(it)
            history["L_E"].append(L_E_val)
            history["L_P"].append(L_P_val)
            history["ET"].append(diag["ET_mean"])
            history["final_d"].append(diag["final_dist_mean"])
            history["no_cap_frac"].append(diag["no_capture_frac"])
            elapsed = time.time() - t0
            print(f"it {it:5d}  L_P={L_P_val:.4f}  L_E={L_E_val:.4f}  "
                  f"E[T]={diag['ET_mean']:.3f}  "
                  f"final_d={diag['final_dist_mean']:.3f}  "
                  f"no_cap_frac={diag['no_capture_frac']:.3f}  "
                  f"[{elapsed:.0f}s]")

    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "checkpoint.npz"),
             B_E=paramsE["B"], b_E=paramsE["b_body"], W_E=paramsE["W"],
             B_P=paramsP["B"], b_P=paramsP["b_body"], W_P=paramsP["W"],
             x_E0=x_E0)
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    return paramsE, paramsP, x_E0, history


# =============================================================================
# Visualisation
# =============================================================================

def plot_loss_curves(history: dict, path: str) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history["iter"], history["L_P"], label=r"$\mathcal{L}_P$",
               color="tab:red")
    ax[0].plot(history["iter"], history["L_E"], label=r"$\mathcal{L}_E$",
               color="tab:blue")
    ax[0].set_xlabel("iteration"); ax[0].set_ylabel("loss"); ax[0].legend()
    ax[0].set_title("Adversarial losses")
    ax[1].plot(history["iter"], history["ET"],
               label=r"$\mathbb{E}[T]_{\rm soft}$", color="tab:purple")
    ax[1].plot(history["iter"], history["final_d"],
               label=r"$\|x_E-x_P\|_T(t_{\max})$", color="tab:orange")
    ax[1].plot(history["iter"], history["no_cap_frac"],
               label="no-capture fraction", color="tab:green")
    ax[1].set_xlabel("iteration"); ax[1].legend(fontsize=8)
    ax[1].set_title("Capture diagnostics")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_torus_path(ax, traj, L, color, label=None, lw=1.0):
    xs, ys = traj[:, 0], traj[:, 1]
    seg_x, seg_y = [], []
    first = True
    for k in range(len(xs)):
        if k > 0 and (abs(xs[k] - xs[k - 1]) > L / 2 or
                       abs(ys[k] - ys[k - 1]) > L / 2):
            if seg_x:
                ax.plot(seg_x, seg_y, color=color, lw=lw,
                        label=label if first else None)
                first = False
                seg_x, seg_y = [], []
        seg_x.append(xs[k]); seg_y.append(ys[k])
    if seg_x:
        ax.plot(seg_x, seg_y, color=color, lw=lw,
                label=label if first else None)


def plot_trajectories(paramsE, paramsP, x_E0, cfg: TrainConfig, path: str,
                      n_show: int = 9) -> None:
    t_grid = np.linspace(0.0, cfg.t_max, cfg.n_t + 1)
    phi = fourier_features(t_grid, cfg.t_max, cfg.n_freqs)
    uE, _ = forward_dirs(paramsE, phi)
    uP, _ = forward_dirs(paramsP, phi)
    xE, xP = rollout(uE, uP, x_E0, cfg.L, cfg.v_E, cfg.v_P, cfg.t_max, cfg.n_t)
    n_h = xE.shape[0]
    n_show = min(n_h, n_show)
    cols = 3
    rows = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 3.3 * rows))
    axes = np.array(axes).reshape(-1)
    for j in range(n_show):
        ax = axes[j]
        _plot_torus_path(ax, xE[j], cfg.L, "tab:blue", label="Evader", lw=1.2)
        _plot_torus_path(ax, xP[j], cfg.L, "tab:red", label="Pursuer", lw=1.2)
        ax.scatter([xE[j, 0, 0]], [xE[j, 0, 1]], s=45, c="tab:blue",
                   marker="o", edgecolor="k", zorder=5)
        ax.scatter([xP[j, 0, 0]], [xP[j, 0, 1]], s=45, c="tab:red",
                   marker="s", edgecolor="k", zorder=5)
        ax.scatter([xE[j, -1, 0]], [xE[j, -1, 1]], s=30, c="tab:blue",
                   marker="x", zorder=5)
        ax.scatter([xP[j, -1, 0]], [xP[j, -1, 1]], s=30, c="tab:red",
                   marker="x", zorder=5)
        diff = (xE[j] - xP[j] + cfg.L / 2) % cfg.L - cfg.L / 2
        dists = np.sqrt((diff ** 2).sum(axis=-1))
        cap_idx = np.where(dists <= cfg.eps)[0]
        Tc = (cap_idx[0] / cfg.n_t) * cfg.t_max if len(cap_idx) else float("nan")
        ax.set_xlim(0, cfg.L); ax.set_ylim(0, cfg.L); ax.set_aspect("equal")
        ax.set_title(f"IC {j}: $T_c$≈{Tc:.2f}"
                     if not math.isnan(Tc) else f"IC {j}: no capture")
        if j == 0:
            ax.legend(loc="upper right", fontsize=7, framealpha=0.85)
    for j in range(n_show, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Learned trajectories on the torus $\\mathbb{T}^2$", y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_heading_angles(paramsE, paramsP, cfg: TrainConfig, path: str,
                        head_idx: int = 0) -> None:
    t_grid = np.linspace(0.0, cfg.t_max, 400)
    phi = fourier_features(t_grid, cfg.t_max, cfg.n_freqs)
    uE, _ = forward_dirs(paramsE, phi)
    uP, _ = forward_dirs(paramsP, phi)
    angE = np.arctan2(uE[head_idx, :, 1], uE[head_idx, :, 0])
    angP = np.arctan2(uP[head_idx, :, 1], uP[head_idx, :, 0])
    fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))
    ax.plot(t_grid, np.unwrap(angE), color="tab:blue", label=r"$\theta_E(t)$")
    ax.plot(t_grid, np.unwrap(angP), color="tab:red", label=r"$\theta_P(t)$")
    ax.set_xlabel("t"); ax.set_ylabel("heading angle (rad, unwrapped)")
    ax.set_title(f"Learned heading angles (IC {head_idx})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_latent_pca(params, cfg: TrainConfig, path: str, title_player: str,
                    n_grid: int = 400) -> None:
    t_grid = np.linspace(0.0, cfg.t_max, n_grid)
    phi = fourier_features(t_grid, cfg.t_max, cfg.n_freqs)
    _, cache = forward_dirs(params, phi)
    H = cache["H"]
    Hc = H - H.mean(axis=0, keepdims=True)
    C = Hc.T @ Hc / Hc.shape[0]
    eig = np.linalg.eigvalsh(C)[::-1]
    eig = np.clip(eig, 0, None)
    total = eig.sum() + 1e-12
    norm = eig / total
    cum = np.cumsum(norm)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.bar(np.arange(1, len(eig) + 1), norm, alpha=0.7,
           color="tab:gray", label="explained variance")
    ax.set_xlabel("PC index"); ax.set_ylabel("normalised eigenvalue")
    ax2 = ax.twinx()
    ax2.plot(np.arange(1, len(eig) + 1), cum, "o-", color="tab:red",
             label="cumulative")
    ax2.set_ylabel("cumulative explained variance")
    ax2.set_ylim(0, 1.05)
    ax.set_title(f"PCA of the {title_player} latent embedding")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pursuer_field(paramsP, cfg: TrainConfig, path: str,
                        head_idx: int = 0, t_snap: float = 0.0) -> None:
    """Snapshot of the pursuer field for a single head at a fixed time."""
    t_arr = np.array([t_snap])
    phi = fourier_features(t_arr, cfg.t_max, cfg.n_freqs)
    uP, _ = forward_dirs(paramsP, phi)
    u = uP[head_idx, 0]
    # Visualise the head profile as a function of time, since the model
    # parametrises u(t) (not u(x)); we show direction vs time as a quiver.
    t_dense = np.linspace(0.0, cfg.t_max, 30)
    phi_d = fourier_features(t_dense, cfg.t_max, cfg.n_freqs)
    uPd, _ = forward_dirs(paramsP, phi_d)
    uEd, _ = forward_dirs(paramsP, phi_d)  # reuse for layout
    fig, ax = plt.subplots(1, 1, figsize=(7, 3))
    for j in range(min(cfg.n_h, 6)):
        ax.quiver(t_dense, j * np.ones_like(t_dense),
                  uPd[j, :, 0], uPd[j, :, 1],
                  color="tab:red", alpha=0.7, scale=18,
                  width=0.005)
    ax.set_yticks(range(min(cfg.n_h, 6)))
    ax.set_yticklabels([f"IC {j}" for j in range(min(cfg.n_h, 6))])
    ax.set_xlabel("t"); ax.set_title("Pursuer unit-direction field $\\hat u_P^j(t)$")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Quick smoke-test entry
# =============================================================================

if __name__ == "__main__":
    cfg = TrainConfig()
    out_dir = os.path.dirname(os.path.abspath(__file__))
    paramsE, paramsP, x_E0, history = train(cfg, out_dir=out_dir)
    plot_loss_curves(history, os.path.join(out_dir, "fig_loss_curves.png"))
    plot_trajectories(paramsE, paramsP, x_E0, cfg,
                      os.path.join(out_dir, "fig_trajectories.png"))
    plot_heading_angles(paramsE, paramsP, cfg,
                        os.path.join(out_dir, "fig_heading_angles.png"))
    plot_latent_pca(paramsE, cfg,
                    os.path.join(out_dir, "fig_latent_pca_E.png"),
                    "evader $H_E(t)$")
    plot_latent_pca(paramsP, cfg,
                    os.path.join(out_dir, "fig_latent_pca_P.png"),
                    "pursuer $H_P(t)$")
    plot_pursuer_field(paramsP, cfg,
                       os.path.join(out_dir, "fig_pursuer_field.png"))
    print("Wrote artefacts to", out_dir)
