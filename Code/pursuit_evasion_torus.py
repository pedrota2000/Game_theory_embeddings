"""
Multi-head adversarial neural network for pursuit-evasion on the 2-torus.

The setup mirrors the embeddings construction of Jimenez et al.:
- A *body* MLP maps time t -> latent embedding H(t) ∈ R^{n_b}.
- A bank of N_h *linear heads* (one per evader initial condition) maps H(t)
  to a raw 2-vector that is normalised to a unit direction û(t) ∈ S^1.
- Two such multi-head networks (one for the evader E, one for the pursuer P)
  are trained adversarially in a GAN-like alternating loop.

The trajectory dynamics
    ẋ_E = v_E · û_E(t)        ẋ_P = v_P · û_P(t)
are integrated by an explicit trapezoidal step on the torus T^2 = [0,L)^2 with
periodic boundary conditions, and the loss is a differentiable expected
first-capture-time built from a smooth survival function.

Author: generated for the differential-game embeddings project.
"""
from __future__ import annotations

import math
import os
import json
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Torus utilities (periodic boundary conditions on [0, L)^2)
# ---------------------------------------------------------------------------

def wrap_torus(x: torch.Tensor, L: float) -> torch.Tensor:
    """Wrap coordinates onto the torus [0, L)^d."""
    return torch.remainder(x, L)


def torus_displacement(x_from: torch.Tensor, x_to: torch.Tensor, L: float) -> torch.Tensor:
    """Signed minimum-image displacement from `x_from` to `x_to`."""
    return torch.remainder(x_to - x_from + L / 2.0, L) - L / 2.0


def torus_distance(x: torch.Tensor, y: torch.Tensor, L: float) -> torch.Tensor:
    """Euclidean torus distance ||x - y||_T."""
    return torch.norm(torus_displacement(x, y, L), dim=-1)


# ---------------------------------------------------------------------------
# Multi-head time-dependent velocity network
# ---------------------------------------------------------------------------

class TimeEncoding(nn.Module):
    """Sinusoidal time encoding plus the raw time scaled by 1/t_max."""

    def __init__(self, t_max: float, n_freqs: int = 8):
        super().__init__()
        self.t_max = t_max
        self.n_freqs = n_freqs
        freqs = (2.0 * math.pi / t_max) * torch.arange(1, n_freqs + 1, dtype=torch.float32)
        self.register_buffer("freqs", freqs)

    def out_dim(self) -> int:
        return 2 * self.n_freqs + 1

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # (N_t, 1)
        ang = t * self.freqs  # (N_t, n_freqs)
        return torch.cat([torch.sin(ang), torch.cos(ang), t / self.t_max], dim=-1)


class MultiHeadVelocityNet(nn.Module):
    """
    Body: MLP encoding t -> H(t) ∈ R^{n_b}.
    Heads: N_h linear maps W^j ∈ R^{2 × n_b} that produce a raw 2-vector
           per evader IC; the output direction is normalised to unit length.

    The shared body encodes information common to all initial conditions;
    each head specialises to one IC.  This is the multi-head construction
    used in the PDE embeddings paper, adapted for two-dimensional velocity
    outputs.
    """

    def __init__(self,
                 t_max: float,
                 n_b: int = 16,
                 n_h: int = 16,
                 hidden: int = 128,
                 n_layers: int = 4,
                 n_freqs: int = 8):
        super().__init__()
        self.t_max = t_max
        self.n_b = n_b
        self.n_h = n_h
        self.enc = TimeEncoding(t_max, n_freqs=n_freqs)
        in_dim = self.enc.out_dim()

        layers = []
        d = in_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.Tanh())
            d = hidden
        layers.append(nn.Linear(hidden, n_b))
        self.body = nn.Sequential(*layers)

        # Linear heads: tensor of shape (N_h, 2, n_b).
        self.W = nn.Parameter(0.2 * torch.randn(n_h, 2, n_b))

    # ----- latent ---------------------------------------------------------
    def latent(self, t: torch.Tensor) -> torch.Tensor:
        return self.body(self.enc(t))  # (N_t, n_b)

    # ----- unit directions per head --------------------------------------
    def unit_directions(self, t: torch.Tensor) -> torch.Tensor:
        """
        Returns the per-head unit directions for every time in `t`.

        Output shape: (n_h, N_t, 2)
        """
        H = self.latent(t)                                  # (N_t, n_b)
        raw = torch.einsum("jab,tb->jta", self.W, H)        # (n_h, N_t, 2)
        norm = torch.norm(raw, dim=-1, keepdim=True).clamp_min(1e-8)
        return raw / norm

    # ----- head orthogonality penalty (paper-style) ----------------------
    def head_orth_penalty(self) -> torch.Tensor:
        """||W^T W - I||_F^2 + ||W W^T - I||_F^2 on the flattened head matrix."""
        Wflat = self.W.reshape(self.n_h * 2, self.n_b)  # (2 N_h, n_b)
        I_nb = torch.eye(self.n_b, device=Wflat.device)
        I_h = torch.eye(self.n_h * 2, device=Wflat.device)
        return ((Wflat.t() @ Wflat - I_nb) ** 2).sum() + ((Wflat @ Wflat.t() - I_h) ** 2).sum()


# ---------------------------------------------------------------------------
# Trajectory rollout with periodic BCs
# ---------------------------------------------------------------------------

def rollout(net_E: MultiHeadVelocityNet,
            net_P: MultiHeadVelocityNet,
            x_E0: torch.Tensor,
            L: float,
            v_E: float,
            v_P: float,
            t_max: float,
            n_t: int):
    """
    Integrate trajectories for all N_h ICs simultaneously, with explicit
    trapezoidal (midpoint-average) steps and torus wrap-around.

    Pursuer is initialised at the origin for every IC, as specified.

    Returns
    -------
    xE_traj : (n_h, n_t+1, 2) tensor
    xP_traj : (n_h, n_t+1, 2) tensor
    t_grid  : (n_t+1,) tensor
    """
    device = x_E0.device
    dt = t_max / n_t
    t_grid = torch.linspace(0.0, t_max, n_t + 1, device=device)

    uE = net_E.unit_directions(t_grid)  # (n_h, n_t+1, 2)
    uP = net_P.unit_directions(t_grid)  # (n_h, n_t+1, 2)

    xE = x_E0.clone()
    xP = torch.zeros_like(x_E0)
    xE_list = [xE]
    xP_list = [xP]
    for k in range(n_t):
        uE_mid = 0.5 * (uE[:, k] + uE[:, k + 1])
        uP_mid = 0.5 * (uP[:, k] + uP[:, k + 1])
        xE = wrap_torus(xE_list[-1] + dt * v_E * uE_mid, L)
        xP = wrap_torus(xP_list[-1] + dt * v_P * uP_mid, L)
        xE_list.append(xE)
        xP_list.append(xP)

    xE_traj = torch.stack(xE_list, dim=1)
    xP_traj = torch.stack(xP_list, dim=1)
    return xE_traj, xP_traj, t_grid


# ---------------------------------------------------------------------------
# Differentiable expected first-capture-time loss
# ---------------------------------------------------------------------------

def capture_losses(xE_traj: torch.Tensor,
                   xP_traj: torch.Tensor,
                   t_grid: torch.Tensor,
                   L: float,
                   eps: float,
                   tau: float,
                   t_max: float,
                   no_capture_penalty: float = 2.0):
    """
    Differentiable first-capture-time via a smooth survival function.

    Per timestep capture probability:
        p_k = sigmoid((eps^2 - d_k^2) / tau)
    Survival function:
        S_k = prod_{j<=k} (1 - p_j)
    Expected first-capture-time (T = t_max if no capture in [0, t_max]):
        E[T] = ∫_0^{t_max} S(t) dt
    Pursuer minimises  E[T] + λ · S(t_max) · t_max  (extra penalty if it fails
    to capture by the horizon).
    Evader minimises  -E[T].
    """
    diff = torus_displacement(xP_traj, xE_traj, L)        # (n_h, n_t+1, 2)
    d2 = (diff ** 2).sum(dim=-1)                          # (n_h, n_t+1)
    p = torch.sigmoid((eps ** 2 - d2) / tau)              # (n_h, n_t+1)
    log_one_minus_p = torch.log((1 - p).clamp_min(1e-8))  # (n_h, n_t+1)
    log_S = torch.cumsum(log_one_minus_p, dim=1)
    S = torch.exp(log_S)                                  # (n_h, n_t+1)

    dt = (t_grid[1] - t_grid[0])
    # trapezoidal integral of S over [0, t_max]
    ET = dt * (0.5 * S[:, 0] + S[:, 1:-1].sum(dim=1) + 0.5 * S[:, -1])
    no_cap_pen = no_capture_penalty * S[:, -1] * t_max

    L_P = (ET + no_cap_pen).mean()
    L_E = -ET.mean()

    with torch.no_grad():
        diagnostics = {
            "ET_mean": ET.mean().item(),
            "ET_min": ET.min().item(),
            "ET_max": ET.max().item(),
            "final_dist_mean": torch.sqrt(d2[:, -1]).mean().item(),
            "no_capture_frac": (S[:, -1] > 0.5).float().mean().item(),
        }
    return L_E, L_P, diagnostics


# ---------------------------------------------------------------------------
# Initial-condition sampling for the evader
# ---------------------------------------------------------------------------

def sample_evader_ICs(n_h: int, L: float, min_dist: float = 0.4, seed: int = 0) -> torch.Tensor:
    """Sample evader positions on the torus, away from the origin (pursuer)."""
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n_h:
        p = rng.uniform(0.0, L, size=2)
        # torus distance from origin
        dx = min(p[0], L - p[0])
        dy = min(p[1], L - p[1])
        if math.sqrt(dx * dx + dy * dy) >= min_dist:
            pts.append(p)
    return torch.tensor(np.array(pts), dtype=torch.float32)


# ---------------------------------------------------------------------------
# Training configuration & loop
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    n_iters: int = 12000
    n_h: int = 12
    n_b: int = 16
    t_max: float = 4.0
    n_t: int = 64
    v_E: float = 1.0
    v_P: float = 1.2
    eps: float = 0.08
    tau: float = 0.005
    L: float = 2.0
    lr_E: float = 1e-3
    lr_P: float = 1e-3
    no_capture_penalty: float = 3.0
    head_orth_weight: float = 1e-4
    n_P_steps: int = 1
    n_E_steps: int = 1
    log_every: int = 200
    seed: int = 0


def train(cfg: TrainConfig, out_dir: str, device: str = "cpu"):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    net_E = MultiHeadVelocityNet(t_max=cfg.t_max, n_b=cfg.n_b, n_h=cfg.n_h).to(device)
    net_P = MultiHeadVelocityNet(t_max=cfg.t_max, n_b=cfg.n_b, n_h=cfg.n_h).to(device)
    opt_E = optim.Adam(net_E.parameters(), lr=cfg.lr_E)
    opt_P = optim.Adam(net_P.parameters(), lr=cfg.lr_P)
    x_E0 = sample_evader_ICs(cfg.n_h, L=cfg.L, min_dist=0.4, seed=cfg.seed).to(device)

    history = {"iter": [], "L_E": [], "L_P": [], "ET": [],
               "final_d": [], "no_cap_frac": []}

    for it in range(cfg.n_iters):
        # ------ pursuer step ------
        for _ in range(cfg.n_P_steps):
            opt_P.zero_grad()
            xE_t, xP_t, t_grid = rollout(net_E, net_P, x_E0,
                                         cfg.L, cfg.v_E, cfg.v_P,
                                         cfg.t_max, cfg.n_t)
            L_E_val, L_P_val, diag = capture_losses(
                xE_t, xP_t, t_grid, cfg.L, cfg.eps, cfg.tau, cfg.t_max,
                no_capture_penalty=cfg.no_capture_penalty)
            loss_P = L_P_val + cfg.head_orth_weight * net_P.head_orth_penalty()
            loss_P.backward()
            opt_P.step()

        # ------ evader step ------
        for _ in range(cfg.n_E_steps):
            opt_E.zero_grad()
            xE_t, xP_t, t_grid = rollout(net_E, net_P, x_E0,
                                         cfg.L, cfg.v_E, cfg.v_P,
                                         cfg.t_max, cfg.n_t)
            L_E_val, L_P_val, diag = capture_losses(
                xE_t, xP_t, t_grid, cfg.L, cfg.eps, cfg.tau, cfg.t_max,
                no_capture_penalty=cfg.no_capture_penalty)
            loss_E = L_E_val + cfg.head_orth_weight * net_E.head_orth_penalty()
            loss_E.backward()
            opt_E.step()

        if it % cfg.log_every == 0 or it == cfg.n_iters - 1:
            history["iter"].append(it)
            history["L_E"].append(L_E_val.item())
            history["L_P"].append(L_P_val.item())
            history["ET"].append(diag["ET_mean"])
            history["final_d"].append(diag["final_dist_mean"])
            history["no_cap_frac"].append(diag["no_capture_frac"])
            print(f"it {it:5d}  L_P={L_P_val.item():.4f}  L_E={L_E_val.item():.4f}  "
                  f"E[T]={diag['ET_mean']:.3f}  final_d={diag['final_dist_mean']:.3f}  "
                  f"no_cap_frac={diag['no_capture_frac']:.3f}")

    # Save artefacts
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"net_E": net_E.state_dict(), "net_P": net_P.state_dict(),
                "x_E0": x_E0.cpu(), "cfg": asdict(cfg)},
               os.path.join(out_dir, "checkpoint.pt"))
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    return net_E, net_P, x_E0, history


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def plot_loss_curves(history: dict, path: str) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history["iter"], history["L_P"], label=r"$\mathcal{L}_P$", color="tab:red")
    ax[0].plot(history["iter"], history["L_E"], label=r"$\mathcal{L}_E$", color="tab:blue")
    ax[0].set_xlabel("iteration")
    ax[0].set_ylabel("loss")
    ax[0].legend()
    ax[0].set_title("Adversarial losses")

    ax[1].plot(history["iter"], history["ET"], label=r"$\mathbb{E}[T]_{\rm soft}$",
               color="tab:purple")
    ax[1].plot(history["iter"], history["final_d"], label=r"$\|x_E(t_{\max})-x_P(t_{\max})\|_T$",
               color="tab:orange")
    ax[1].plot(history["iter"], history["no_cap_frac"], label="no-capture fraction",
               color="tab:green")
    ax[1].set_xlabel("iteration")
    ax[1].legend(fontsize=8)
    ax[1].set_title("Capture diagnostics")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_torus_path(ax, traj, L, color, label=None, lw=1.0):
    """Plot a periodic-torus path, breaking line segments at wrap discontinuities."""
    xs, ys = traj[:, 0], traj[:, 1]
    seg_x, seg_y = [], []
    first = True
    for k in range(len(xs)):
        if k > 0 and (abs(xs[k] - xs[k - 1]) > L / 2 or abs(ys[k] - ys[k - 1]) > L / 2):
            if seg_x:
                ax.plot(seg_x, seg_y, color=color, lw=lw, label=label if first else None)
                first = False
                seg_x, seg_y = [], []
        seg_x.append(xs[k])
        seg_y.append(ys[k])
    if seg_x:
        ax.plot(seg_x, seg_y, color=color, lw=lw, label=label if first else None)


def plot_trajectories(net_E, net_P, x_E0, cfg: TrainConfig, path: str,
                      n_show: int = 9) -> None:
    with torch.no_grad():
        xE, xP, _ = rollout(net_E, net_P, x_E0,
                            cfg.L, cfg.v_E, cfg.v_P, cfg.t_max, cfg.n_t)
    xE = xE.cpu().numpy()
    xP = xP.cpu().numpy()
    n_h = xE.shape[0]
    n_show = min(n_h, n_show)
    cols = 3
    rows = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows))
    axes = np.array(axes).reshape(-1)
    for j in range(n_show):
        ax = axes[j]
        _plot_torus_path(ax, xE[j], cfg.L, "tab:blue", label="Evader", lw=1.1)
        _plot_torus_path(ax, xP[j], cfg.L, "tab:red", label="Pursuer", lw=1.1)
        ax.scatter([xE[j, 0, 0]], [xE[j, 0, 1]], s=40, c="tab:blue",
                   marker="o", edgecolor="k", zorder=5)
        ax.scatter([xP[j, 0, 0]], [xP[j, 0, 1]], s=40, c="tab:red",
                   marker="s", edgecolor="k", zorder=5)
        # final positions
        ax.scatter([xE[j, -1, 0]], [xE[j, -1, 1]], s=25, c="tab:blue",
                   marker="x", zorder=5)
        ax.scatter([xP[j, -1, 0]], [xP[j, -1, 1]], s=25, c="tab:red",
                   marker="x", zorder=5)
        ax.set_xlim(0, cfg.L); ax.set_ylim(0, cfg.L)
        ax.set_aspect("equal")
        # Approximate capture time for plot title
        from numpy.linalg import norm
        diff = (xE[j] - xP[j] + cfg.L / 2) % cfg.L - cfg.L / 2
        dists = np.sqrt((diff ** 2).sum(axis=-1))
        cap_idx = np.where(dists <= cfg.eps)[0]
        Tc = (cap_idx[0] / cfg.n_t) * cfg.t_max if len(cap_idx) else float("nan")
        ax.set_title(f"IC {j}: T_c≈{Tc:.2f}" if not math.isnan(Tc) else f"IC {j}: no capture")
        if j == 0:
            ax.legend(loc="upper right", fontsize=7, framealpha=0.85)
    for j in range(n_show, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Learned trajectories on the torus T$^2$", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_velocity_field_evader(net_E, cfg: TrainConfig, path: str, head_idx: int = 0):
    """Visualise the time-resolved evader unit direction for one head."""
    with torch.no_grad():
        t = torch.linspace(0.0, cfg.t_max, 200)
        u = net_E.unit_directions(t)[head_idx].cpu().numpy()
    angles = np.arctan2(u[:, 1], u[:, 0])
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(t.numpy(), u[:, 0], label=r"$\hat u_{E,x}$", color="tab:blue")
    ax[0].plot(t.numpy(), u[:, 1], label=r"$\hat u_{E,y}$", color="tab:red")
    ax[0].set_xlabel("t"); ax[0].legend()
    ax[0].set_title(f"Evader heading components (IC {head_idx})")
    ax[1].plot(t.numpy(), angles, color="tab:purple")
    ax[1].set_xlabel("t"); ax[1].set_ylabel(r"$\theta_E(t)$ [rad]")
    ax[1].set_title("Heading angle")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_latent_pca(net_E, cfg: TrainConfig, path: str, n_grid: int = 400):
    """Centred-latent covariance PCA, paper-style."""
    with torch.no_grad():
        t = torch.linspace(0.0, cfg.t_max, n_grid)
        H = net_E.latent(t).cpu().numpy()  # (n_grid, n_b)
    Hc = H - H.mean(axis=0, keepdims=True)
    C = Hc.T @ Hc / Hc.shape[0]
    eig = np.linalg.eigvalsh(C)[::-1]
    eig = np.clip(eig, 0, None)
    total = eig.sum() + 1e-12
    norm = eig / total
    cum = np.cumsum(norm)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    bar = ax.bar(np.arange(1, len(eig) + 1), norm, alpha=0.7, label="explained variance")
    ax.set_xlabel("PC index"); ax.set_ylabel("normalised eigenvalue")
    ax2 = ax.twinx()
    ax2.plot(np.arange(1, len(eig) + 1), cum, "o-", color="tab:red", label="cumulative")
    ax2.set_ylabel("cumulative explained variance")
    ax2.set_ylim(0, 1.05)
    ax.set_title("PCA of the evader latent embedding $H_E(t)$")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = TrainConfig()
    out_dir = os.path.dirname(os.path.abspath(__file__))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device={device}")
    net_E, net_P, x_E0, history = train(cfg, out_dir=out_dir, device=device)

    plot_loss_curves(history,
                     os.path.join(out_dir, "fig_loss_curves.png"))
    plot_trajectories(net_E, net_P, x_E0, cfg,
                      os.path.join(out_dir, "fig_trajectories.png"))
    plot_velocity_field_evader(net_E, cfg,
                               os.path.join(out_dir, "fig_evader_heading.png"))
    plot_latent_pca(net_E, cfg,
                    os.path.join(out_dir, "fig_latent_pca_E.png"))
    plot_latent_pca(net_P, cfg,
                    os.path.join(out_dir, "fig_latent_pca_P.png"))
    print("Done — figures and checkpoint written to", out_dir)
