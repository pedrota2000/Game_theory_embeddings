"""Driver script: trains the model and produces all figures."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pursuit_evasion_numpy import (
    TrainConfig, train,
    plot_loss_curves, plot_trajectories, plot_heading_angles,
    plot_latent_pca, plot_pursuer_field,
)

cfg = TrainConfig(
    n_iters=1200, log_every=100, n_h=12,
    n_P_steps=4, n_E_steps=1,
    lr_P=1e-2, lr_E=1e-3, lam_d=5.0,
    pretrain_P_iters=1500, no_capture_penalty=2.0,
)
out = os.path.dirname(os.path.abspath(__file__)) + "/"
pE, pP, x0, hist = train(cfg, out_dir=out)
plot_loss_curves(hist, out + "fig_loss_curves.png")
plot_trajectories(pE, pP, x0, cfg, out + "fig_trajectories.png")
plot_heading_angles(pE, pP, cfg, out + "fig_heading_angles.png")
plot_latent_pca(pE, cfg, out + "fig_latent_pca_E.png", "evader $H_E(t)$")
plot_latent_pca(pP, cfg, out + "fig_latent_pca_P.png", "pursuer $H_P(t)$")
plot_pursuer_field(pP, cfg, out + "fig_pursuer_field.png")
print("DONE")
