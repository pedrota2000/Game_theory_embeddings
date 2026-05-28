# Pursuit-Evasion Embeddings: Codebase Overview

## Project Summary
This repository implements and analyzes adversarial neural network models for the pursuit-evasion differential game on the 2-torus. The codebase provides both a pure NumPy implementation (for full transparency and gradient checking) and a PyTorch implementation (for scalable, autograd-enabled training). The models are designed to learn optimal strategies for a pursuer and an evader, with a focus on embedding the game dynamics and learning interpretable latent representations.

---

## Repository Structure

```
Code/
  pursuit_evasion_numpy.py   # Main NumPy implementation (hand-coded gradients)
  pursuit_evasion_torus.py   # Main PyTorch implementation (autograd, scalable)
  run_train.py               # Driver script for NumPy version (runs training, plots figures)
  grad_check.py              # Finite-difference gradient check (NumPy)
  grad_check2.py             # Stronger gradient check (NumPy)
  checkpoint.npz             # Model checkpoint (NumPy)
  history.json               # Training history (NumPy)
Documents/
  *.tex, *.pdf, ...          # LaTeX write-ups and generated figures
```

---

## Core Concepts

### 1. Pursuit-Evasion Game on the Torus
- **Players:** Pursuer (P) and Evader (E) move on a 2D torus $[0, L)^2$.
- **Controls:** Each player's velocity is parameterized by a neural network.
- **Objective:**
  - Pursuer: Minimize expected time to capture the evader.
  - Evader: Maximize expected time to capture (i.e., evade as long as possible).

### 2. Neural Network Architecture
- **Body:** Shared MLP maps time $t$ (with Fourier features) to a latent embedding $H(t)$.
- **Heads:** Each initial condition (IC) has a dedicated linear head mapping $H(t)$ to a 2D velocity direction.
- **Multi-head:** Enables simultaneous training on multiple ICs.

### 3. Dynamics & Loss
- **Trajectory Integration:** Explicit trapezoidal step with periodic boundary conditions.
- **Loss:**
  - Differentiable expected first-capture time (via smooth survival function).
  - No-capture penalty for the pursuer.
  - Optional distance-shaping term for stable adversarial training.

---

## Main Files

### `pursuit_evasion_numpy.py`
- **Standalone NumPy implementation** with hand-coded forward and backward passes (no autograd).
- Implements all core logic: network, rollout, loss, Adam optimizer, training loop, and plotting.
- Used for transparent gradient checking and figure generation for the paper.

### `pursuit_evasion_torus.py`
- **PyTorch implementation** with autograd and modular neural network classes.
- Supports scalable training and GPU acceleration.
- Mirrors the NumPy version but leverages PyTorch's features.

### `run_train.py`
- **Driver script** for the NumPy version.
- Runs training, saves checkpoints, and generates all figures (loss curves, trajectories, latent PCA, etc.).

### `grad_check.py` & `grad_check2.py`
- **Gradient checking scripts** for the NumPy implementation.
- `grad_check.py`: Finite-difference check for all parameters.
- `grad_check2.py`: Focuses on largest-gradient entries for more robust verification.

### `checkpoint.npz` & `history.json`
- **Model checkpoint** and **training history** for the NumPy version.
- Used for reproducibility and further analysis.

---

## How to Use

### 1. NumPy Version
- Run `run_train.py` to train the model and generate figures.
- Use `grad_check.py` and `grad_check2.py` to verify gradient correctness.

### 2. PyTorch Version
- Run `pursuit_evasion_torus.py` as a script to train the PyTorch model and generate figures.
- All training artefacts are saved in the same directory.

---

## Figures & Outputs
- **Loss curves:** Training progress for both players.
- **Trajectories:** Learned paths for pursuer and evader.
- **Latent PCA:** Principal component analysis of the learned embeddings.
- **Velocity fields:** Visualization of learned control strategies.

---

## References
- This codebase implements the methods described in the associated LaTeX write-ups in the `Documents/` folder.
- For theoretical background, see the referenced papers in the LaTeX files.

---


---

## Requirements
- Python 3.x
- NumPy, Matplotlib (for NumPy version)
- PyTorch, Matplotlib (for PyTorch version)

---

## Contact
For questions or collaboration, please refer to the contact information in the LaTeX write-ups or reach out to the repository maintainer.
