#!/usr/bin/env python3
"""
Validate Jacobian of the temperature emulator using Taylor expansion.

Picks a sample, computes T(S) and dT/dS, then verifies:
    T(S + dS) ≈ T(S) + (dT/dS) * dS

Compares the linearized prediction against the full nonlinear emulator
for a range of dS magnitudes.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from ufsemulator.model import UfsEmulatorFFNN

# ── Configuration (hard-coded for temperature emulator) ──────────────
CONFIG_FILE  = "/scratch3/NCEPDEV/da/Guillaume.Vernieres/runs/aibalance/runs/run_oceantemp/config_ocntemp.yaml"
MODEL_DIR    = "/scratch3/NCEPDEV/da/Guillaume.Vernieres/runs/aibalance/runs/run_oceantemp/models_ocntemp"
DATA_FILE    = "/scratch3/NCEPDEV/da/Guillaume.Vernieres/runs/aibalance/runs/run_oceantemp/ocean_training_data.npz"
N_LEVELS     = 50
OUTPUT_DIR   = Path("jacobian_validation")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Load model ───────────────────────────────────────────────────────
print("Loading model ...")
checkpoint = torch.load(f"{MODEL_DIR}/best_model.pt", map_location="cpu", weights_only=False)
state_dict = checkpoint["model_state_dict"]
cfg = checkpoint["config"]["model"]

has_conv1d = "conv1d.weight" in state_dict
conv_kw = {}
if has_conv1d:
    conv_kw = dict(
        use_conv1d=True,
        conv_channels=state_dict["conv1d.weight"].shape[0],
        conv_kernel_size=state_dict["conv1d.weight"].shape[2],
    )

model = UfsEmulatorFFNN(
    input_size=cfg["input_size"], hidden_size=cfg["hidden_size"],
    output_size=cfg["output_size"], hidden_layers=cfg["hidden_layers"],
    activation=cfg.get("activation", "gelu"), **conv_kw,
)
model.load_state_dict(state_dict)
model.eval()

# Load normalization
norm = torch.load(f"{MODEL_DIR}/normalization.pt", weights_only=False)
model.init_norm(norm["input_mean"], norm["input_std"],
                norm["output_mean"], norm["output_std"])

input_mean = norm["input_mean"].numpy()
input_std  = norm["input_std"].numpy()
output_mean = norm["output_mean"].numpy()
output_std  = norm["output_std"].numpy()

# ── Load one sample ─────────────────────────────────────────────────
print("Loading data ...")
data = np.load(DATA_FILE)
X_all = data["inputs"].astype(np.float32)
idx = np.random.randint(len(X_all))
x0 = X_all[idx]                    # (input_size,)
salt0 = x0[:N_LEVELS].copy()       # salinity profile
depth = x0[N_LEVELS:2*N_LEVELS]    # depth profile
print(f"  Sample index: {idx}")

# ── Helper: predict in physical space ────────────────────────────────
def predict(x_phys):
    """Run emulator: physical input → physical output."""
    with torch.no_grad():
        t = torch.from_numpy(x_phys).float().unsqueeze(0)
        return model.predict(t).squeeze(0).numpy()

# ── Step 1 & 2: baseline prediction T(S) ────────────────────────────
T0 = predict(x0)
print(f"  T(S) range: [{T0.min():.2f}, {T0.max():.2f}] °C")

# ── Step 3 & 4 & 5: Jacobian computation ────────────────────────────
# Compute full Jacobian dT/d(all inputs) in physical space
x_tensor = torch.from_numpy(x0).float().unsqueeze(0)  # (1, input_size)

# Use torch autograd for physical-space Jacobian
x_tensor.requires_grad_(False)
jac_full = torch.autograd.functional.jacobian(
    lambda x: model.predict(x), x_tensor
)  # shape (1, output_size, 1, input_size)
J = jac_full.squeeze().numpy()  # (output_size, input_size)
J_salt = J[:, :N_LEVELS]  # dT/dS block: (N_LEVELS, N_LEVELS)

print(f"  Jacobian dT/dS shape: {J_salt.shape}")

# ── Sweep over dS magnitudes ────────────────────────────────────────
dS_values = np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4])
rmse_nonlinear = []
rmse_linear = []

print("\n  dS (psu)   | RMSE(linear) °C | RMSE(nonlinear) °C | Ratio")
print("  " + "-" * 64)

for dS in dS_values:
    # Perturbed input: add constant dS to salinity
    x_pert = x0.copy()
    x_pert[:N_LEVELS] += dS

    # Nonlinear prediction: T(S + dS)
    T_nonlinear = predict(x_pert)

    # Linearized prediction: T(S) + J_salt @ dS_vector
    dS_vec = np.full(N_LEVELS, dS)
    T_linear = T0 + J_salt @ dS_vec

    # Errors relative to nonlinear
    err_lin = np.sqrt(np.mean((T_linear - T_nonlinear) ** 2))
    err_nl  = np.sqrt(np.mean((T_nonlinear - T0) ** 2))

    rmse_linear.append(err_lin)
    rmse_nonlinear.append(err_nl)

    ratio = err_lin / err_nl if err_nl > 1e-12 else float("nan")
    print(f"  {dS:10.3f}  | {err_lin:15.6f} | {err_nl:18.6f} | {ratio:.4f}")

# ── Plot 1: Profile comparison for a moderate dS ────────────────────
dS_plot = 0.1  # psu
x_pert = x0.copy()
x_pert[:N_LEVELS] += dS_plot
T_nl = predict(x_pert)
T_lin = T0 + J_salt @ np.full(N_LEVELS, dS_plot)

fig, axes = plt.subplots(1, 3, figsize=(18, 7))

# Panel 1: Temperature profiles
ax = axes[0]
ax.plot(T0, depth, "k-o", ms=3, lw=2, label="T(S)")
ax.plot(T_nl, depth, "b-s", ms=3, lw=2, label=f"T(S+{dS_plot} psu)  [nonlinear]")
ax.plot(T_lin, depth, "r--^", ms=3, lw=2, label=f"T(S) + J·dS  [linear]")
ax.set_xlabel("Temperature (°C)", fontsize=12)
ax.set_ylabel("Depth (m)", fontsize=12)
ax.set_title("Temperature Profiles", fontsize=14, fontweight="bold")
ax.invert_yaxis()
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 2: Difference from baseline
ax = axes[1]
ax.plot(T_nl - T0, depth, "b-s", ms=3, lw=2, label="Nonlinear ΔT")
ax.plot(T_lin - T0, depth, "r--^", ms=3, lw=2, label="Linear ΔT (J·dS)")
ax.set_xlabel("ΔT (°C)", fontsize=12)
ax.set_ylabel("Depth (m)", fontsize=12)
ax.set_title(f"Temperature Increment (dS = {dS_plot} psu)", fontsize=14, fontweight="bold")
ax.invert_yaxis()
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.axvline(0, color="gray", ls="--", alpha=0.3)

# Panel 3: Linearization error
ax = axes[2]
err_profile = T_lin - T_nl
ax.plot(err_profile, depth, "g-o", ms=3, lw=2)
ax.set_xlabel("Linearization Error (°C)", fontsize=12)
ax.set_ylabel("Depth (m)", fontsize=12)
ax.set_title("Error: Linear − Nonlinear", fontsize=14, fontweight="bold")
ax.invert_yaxis()
ax.grid(True, alpha=0.3)
ax.axvline(0, color="gray", ls="--", alpha=0.3)
rmse_val = np.sqrt(np.mean(err_profile**2))
ax.text(0.05, 0.95, f"RMSE: {rmse_val:.4f} °C", transform=ax.transAxes,
        fontsize=11, va="top", bbox=dict(boxstyle="round", fc="wheat", alpha=0.8))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "profile_comparison.png", dpi=150, bbox_inches="tight")
print(f"\nSaved: {OUTPUT_DIR / 'profile_comparison.png'}")
plt.close()

# ── Plot 2: RMSE vs dS magnitude (Taylor convergence) ───────────────
fig, ax = plt.subplots(figsize=(8, 6))
ax.loglog(dS_values, rmse_linear, "ro-", lw=2, ms=8, label="Linearization error ||T_lin − T_nl||")
ax.loglog(dS_values, rmse_nonlinear, "bs--", lw=2, ms=8, label="Signal ||T(S+dS) − T(S)||")

# Reference lines for O(dS) and O(dS²)
ds_ref = np.array([dS_values[0], dS_values[-1]])
scale1 = rmse_nonlinear[0] / dS_values[0]
scale2 = rmse_linear[0] / dS_values[0] ** 2
ax.loglog(ds_ref, scale1 * ds_ref, "b:", alpha=0.5, lw=1.5, label="O(dS)")
ax.loglog(ds_ref, scale2 * ds_ref ** 2, "r:", alpha=0.5, lw=1.5, label="O(dS²)")

ax.set_xlabel("dS perturbation (psu)", fontsize=12)
ax.set_ylabel("RMSE (°C)", fontsize=12)
ax.set_title("Jacobian Validation: Taylor Convergence", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which="both")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "taylor_convergence.png", dpi=150, bbox_inches="tight")
print(f"Saved: {OUTPUT_DIR / 'taylor_convergence.png'}")
plt.close()

# ── Contraction check & Newton iteration ─────────────────────────────
# Full Newton iteration to recover the nonlinear solution:
#
#   Given a target T* = T(S0 + dS_true), find dS such that T(S0+dS) = T*.
#
#   At each iteration k:
#       1. Evaluate  T_k  = T(S0 + dS_k)         (nonlinear forward model)
#       2. Compute   J_k  = dT/dS |_{S0 + dS_k}  (Jacobian at current state)
#       3. Update    dS_{k+1} = dS_k + J_k^{-1} · (T* − T_k)
#
#   This is Newton's method.  If the Jacobian is well-conditioned and the
#   map is locally contracting, we expect quadratic convergence.

print("\n" + "=" * 60)
print("Newton Iteration (Jacobian recomputed at each step)")
print("=" * 60)

def compute_jacobian_salt(x_phys):
    """Compute dT/dS Jacobian at a given physical input state."""
    xt = torch.from_numpy(x_phys).float().unsqueeze(0)
    jac_full = torch.autograd.functional.jacobian(
        lambda x: model.predict(x), xt
    )
    J = jac_full.squeeze().numpy()  # (output_size, input_size)
    return J[:, :N_LEVELS]          # dT/dS block: (N_LEVELS, N_LEVELS)

# ── Regularized solve via truncated SVD ──────────────────────────────
SVD_RTOL = 1e-6  # singular values < rtol * σ_max are discarded

def truncated_svd_solve(J, rhs, rtol=SVD_RTOL):
    """Solve J @ x = rhs using truncated SVD (pseudo-inverse with cutoff).

    Singular values smaller than rtol * σ_max are zeroed out so that
    near-null-space directions don't blow up the solution.
    Returns (x, effective_rank, condition_number).
    """
    U, s, Vt = np.linalg.svd(J, full_matrices=False)
    thresh = rtol * s[0]
    rank = int(np.sum(s > thresh))
    # Invert only the significant singular values
    s_inv = np.where(s > thresh, 1.0 / s, 0.0)
    x = Vt.T @ (s_inv * (U.T @ rhs))
    cond = s[0] / s[rank - 1] if rank > 0 else float("inf")
    return x, rank, cond

# Check Jacobian properties at baseline
U0, s0, Vt0 = np.linalg.svd(J_salt, full_matrices=False)
eff_rank0 = int(np.sum(s0 > SVD_RTOL * s0[0]))
cond_full = s0[0] / s0[-1] if s0[-1] > 1e-15 else float("inf")
cond_eff  = s0[0] / s0[eff_rank0 - 1] if eff_rank0 > 0 else float("inf")

print(f"\nJacobian dT/dS at baseline:")
print(f"  Spectral norm (σ_max):   {s0[0]:.6f}")
print(f"  Smallest σ:              {s0[-1]:.2e}")
print(f"  Full condition number:   {cond_full:.2e}")
print(f"  Effective rank (rtol={SVD_RTOL}): {eff_rank0} / {N_LEVELS}")
print(f"  Truncated cond number:   {cond_eff:.2f}")

# Target: T* = T(S0 + dS_true)
# Define a linear decay for dS_true with depth
dS_true = np.linspace(0.5, 0.0, N_LEVELS)  # psu, decays linearly from 0.1 at the surface to 0.0 at the bottom
x_target = x0.copy()
x_target[:N_LEVELS] += dS_true
T_target = predict(x_target)

print(f"\nNewton iteration: recovering T(S0 + dS_true)")
print(f"  dS_true: linear decay from {dS_true[0]:.3f} to {dS_true[-1]:.3f} psu")
print(f"  Target T range: [{T_target.min():.2f}, {T_target.max():.2f}] °C")

max_iter = 100
tol = 1e-8
dS_k = np.zeros(N_LEVELS)  # Start with no perturbation

residual_history = []
dS_norm_history = []

# Line-search parameters
LS_ALPHA_MIN = 1e-4   # smallest step fraction
LS_SHRINK    = 0.5     # backtrack factor

print(f"\n  {'Iter':>4}  |  {'||residual||':>14}  |  {'||dS_k||':>12}  |  {'||dS-dS*||':>14}  |  {'rank':>4}  |  {'cond':>10}  |  {'α':>6}")
print("  " + "-" * 88)

for k in range(max_iter):
    # Current state
    x_k = x0.copy()
    x_k[:N_LEVELS] += dS_k

    # Nonlinear evaluation at current guess
    T_k = predict(x_k)

    # Residual: how far are we from the target?
    residual = T_target - T_k
    res_norm = np.linalg.norm(residual)
    residual_history.append(res_norm)

    # Track dS error
    dS_err = np.linalg.norm(dS_k - dS_true)
    dS_norm_history.append(dS_err)

    # Recompute Jacobian at current state & regularized solve
    J_k = compute_jacobian_salt(x_k)
    step, rank_k, cond_k = truncated_svd_solve(J_k, residual)

    # Backtracking line search: ensure the residual actually decreases
    alpha = 1.0
    while alpha > LS_ALPHA_MIN:
        dS_trial = dS_k + alpha * step
        x_trial = x0.copy()
        x_trial[:N_LEVELS] += dS_trial
        T_trial = predict(x_trial)
        res_trial = np.linalg.norm(T_target - T_trial)
        if res_trial < res_norm:
            break
        alpha *= LS_SHRINK

    print(f"  {k:4d}  |  {res_norm:14.8f}  |  {np.linalg.norm(dS_k):12.6f}  |  {dS_err:14.8f}  |  {rank_k:4d}  |  {cond_k:10.2f}  |  {alpha:6.4f}")

    if res_norm < tol:
        print(f"\n  Converged at iteration {k}!")
        break

    # Apply the (possibly damped) Newton update
    dS_k = dS_k + alpha * step

# Final state
x_final = x0.copy()
x_final[:N_LEVELS] += dS_k
T_final = predict(x_final)
final_rmse = np.sqrt(np.mean((T_final - T_target) ** 2))
print(f"\n  Final RMSE: {final_rmse:.8f} °C")
print(f"  Final dS RMSE: {np.sqrt(np.mean((dS_k - dS_true)**2)):.8f} psu")

# ── Plot 3: Newton iteration convergence ─────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Panel 1: Residual norm convergence
ax = axes[0]
ax.semilogy(range(len(residual_history)), residual_history, "ro-", lw=2, ms=6)
ax.set_xlabel("Iteration", fontsize=12)
ax.set_ylabel("||T_target − T(S0 + dS_k)|| (°C)", fontsize=12)
ax.set_title("Newton Iteration: Residual Convergence", fontsize=14, fontweight="bold")
ax.grid(True, alpha=0.3)
if len(residual_history) > 2:
    # Estimate convergence order from last few iterations
    rates = [residual_history[i+1] / residual_history[i]
             for i in range(len(residual_history)-1) if residual_history[i] > 1e-14]
    if rates:
        avg_rate = np.mean(rates[-min(5, len(rates)):])
        ax.text(0.05, 0.05, f"Avg rate: {avg_rate:.4f}\nSVD rtol: {SVD_RTOL}\nEff. rank: {eff_rank0}/{N_LEVELS}",
                transform=ax.transAxes, fontsize=10, va="bottom",
                bbox=dict(boxstyle="round", fc="wheat", alpha=0.8))

# Panel 2: Temperature profile convergence
ax = axes[1]
ax.plot(T0, depth, "k-", lw=2, label="T(S0) — baseline")
ax.plot(T_target, depth, "b-", lw=3, label="T(S0+dS_true) — target")
ax.plot(T_final, depth, "r--", lw=2, label=f"Newton result (iter {len(residual_history)-1})")
# Also show first linear guess
T_lin_first = T0 + J_salt @ dS_true
ax.plot(T_lin_first, depth, "g:", lw=2, label="Single linear step")
ax.set_xlabel("Temperature (°C)", fontsize=12)
ax.set_ylabel("Depth (m)", fontsize=12)
ax.set_title("Profile: Newton vs Nonlinear Target", fontsize=14, fontweight="bold")
ax.invert_yaxis()
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 3: dS convergence
ax = axes[2]
ax.semilogy(range(len(dS_norm_history)), dS_norm_history, "bs-", lw=2, ms=6)
ax.set_xlabel("Iteration", fontsize=12)
ax.set_ylabel("||dS_k − dS_true|| (psu)", fontsize=12)
ax.set_title("Salinity Correction Convergence", fontsize=14, fontweight="bold")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "newton_convergence.png", dpi=150, bbox_inches="tight")
print(f"\nSaved: {OUTPUT_DIR / 'newton_convergence.png'}")
plt.close()

print("\nDone.")
