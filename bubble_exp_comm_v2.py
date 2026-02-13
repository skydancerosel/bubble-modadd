#!/usr/bin/env python3
"""
bubble_exp_comm_v2.py — Corrected commutator analysis for the modadd paper.

Fixes from v1 (bubble_exp_comm.py):
  1. Basis B is now PCA of the weight trajectory (not SVD of current weights).
     The paper claims projection onto "the top k principal components from PCA"
     but the original code used extract_parameter_subspace() which computes
     top-k SVD of current weights — a fundamentally different basis.

  2. Random subspace control.  Without comparing proj_exec vs proj_rand,
     we cannot distinguish "commutator avoids the execution subspace" from
     "any K-dim subspace captures only √(K/P) fraction."  The corrected code
     computes both and reports the ratio exec/random.

  3. Projection computed at every measurement step (not only after step 5000,
     and not only every 5000 steps).

  4. Normalization made explicit.  Paper equations omit the scale-normalization
     ||eta gA|| * ||eta gB|| that the code uses; we now report BOTH raw and
     normalized quantities and the paper text is updated to match.

  5. Correct interpretation of rho.  rho = proj/full → small means the
     commutator lives OUTSIDE the execution manifold (not inside).  We also
     report resid/full ≈ 1.0 to make the claim unambiguous.
"""

import math
import random
import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Import model and data from original code ────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))
from bubble_exp_comm import (
    BubbleTransformer,
    sample_m_bubble_batch,
    extract_attention_blocks,
    bubble_metrics,
)

OUT_DIR = Path(__file__).parent / "modadd_plots"

# ═══════════════════════════════════════════════════════════════════════════
# Core functions
# ═══════════════════════════════════════════════════════════════════════════

def flatten_model_params(model):
    return torch.cat([
        p.detach().flatten()
        for p in model.parameters()
        if p.requires_grad
    ])


def _param_offsets(model):
    offsets = {}
    cursor = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        offsets[id(p)] = cursor
        cursor += p.numel()
    return offsets, cursor


def write_params(model, theta):
    with torch.no_grad():
        offset = 0
        for p in model.parameters():
            if not p.requires_grad:
                continue
            n = p.numel()
            p.copy_(theta[offset:offset+n].view_as(p))
            offset += n


# ═══════════════════════════════════════════════════════════════════════════
# Commutator defect (same as original, but cleaner interface)
# ═══════════════════════════════════════════════════════════════════════════

def commutator_defect(model, batch_fn, device, eta=1e-3, eps=1e-12):
    """
    Compute the commutator delta = theta_AB - theta_BA.

    Returns:
        defect: ||delta|| / (||eta*gA|| * ||eta*gB||)  [scale-normalized]
        delta:  raw commutator vector [P]
        grad_cos: cosine similarity of gA and gB
        normA, normB: ||eta*gA||, ||eta*gB||
    """
    was_training = model.training
    model.train()

    def batch_grad(x, y):
        model.zero_grad(set_to_none=True)
        logits, _ = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        return torch.cat([
            (p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
            for p in model.parameters() if p.requires_grad
        ])

    xA, yA, _ = batch_fn()
    xB, yB, _ = batch_fn()
    xA, yA = xA.to(device), yA.to(device)
    xB, yB = xB.to(device), yB.to(device)

    theta0 = flatten_model_params(model)

    # Gradients at theta0
    gA = batch_grad(xA, yA)
    gB = batch_grad(xB, yB)
    grad_cos = (gA @ gB / (gA.norm() * gB.norm() + eps)).item()

    # Path AB: step A first, then B
    write_params(model, theta0 - eta * gA)
    gB_after_A = batch_grad(xB, yB)
    thetaAB = theta0 - eta * gA - eta * gB_after_A

    # Path BA: step B first, then A
    write_params(model, theta0 - eta * gB)
    gA_after_B = batch_grad(xA, yA)
    thetaBA = theta0 - eta * gB - eta * gA_after_B

    # Restore
    write_params(model, theta0)
    if not was_training:
        model.eval()

    delta = thetaAB - thetaBA
    normA = (eta * gA).norm()
    normB = (eta * gB).norm()
    defect = (delta.norm() / (normA * normB + eps)).item()

    return defect, delta.detach(), grad_cos, normA.detach(), normB.detach()


# ═══════════════════════════════════════════════════════════════════════════
# BASIS CONSTRUCTION: PCA of weight trajectory (FIX #1)
# ═══════════════════════════════════════════════════════════════════════════

def build_trajectory_pca_basis(weight_log, model, n_components=3):
    """
    Build basis B from PCA of the weight trajectory — the correct approach.

    For each attention block (WQ, WK, WV, WO per layer), collect the
    flattened weight vectors across all checkpoints, run PCA, take top-k
    components, embed them in the full parameter space, and QR-orthogonalize.

    This is what the paper *describes* but the original code did not do.
    """
    if len(weight_log) < 5:
        return None

    offsets, total_params = _param_offsets(model)

    # Identify blocks and their offsets in flat param vector
    blocks = {}
    for layer in model.layers:
        layer_idx = list(model.layers).index(layer)
        qkv_w = layer.qkv.weight
        out_w = layer.out.weight
        d = qkv_w.shape[0] // 3

        qkv_off = offsets.get(id(qkv_w), None)
        out_off = offsets.get(id(out_w), None)

        if qkv_off is not None:
            blocks[f"L{layer_idx}_WQ"] = {
                "offset": qkv_off,
                "numel": d * qkv_w.shape[1],
                "shape": (d, qkv_w.shape[1]),
                "key": f"layer{layer_idx}.WQ",
            }
            blocks[f"L{layer_idx}_WK"] = {
                "offset": qkv_off + d * qkv_w.shape[1],
                "numel": d * qkv_w.shape[1],
                "shape": (d, qkv_w.shape[1]),
                "key": f"layer{layer_idx}.WK",
            }
            blocks[f"L{layer_idx}_WV"] = {
                "offset": qkv_off + 2 * d * qkv_w.shape[1],
                "numel": d * qkv_w.shape[1],
                "shape": (d, qkv_w.shape[1]),
                "key": f"layer{layer_idx}.WV",
            }
        if out_off is not None:
            blocks[f"L{layer_idx}_WO"] = {
                "offset": out_off,
                "numel": out_w.numel(),
                "shape": out_w.shape,
                "key": f"layer{layer_idx}.WO",
            }

    # For each block, collect trajectory and compute PCA
    basis_vecs = []

    for bname, binfo in blocks.items():
        key = binfo["key"]
        numel = binfo["numel"]

        # Collect flattened weight vectors across checkpoints
        traj = []
        for step, wdict in weight_log:
            if key in wdict:
                traj.append(wdict[key].flatten().float())

        if len(traj) < 3:
            continue

        traj_mat = torch.stack(traj, dim=0)  # [T, numel]
        traj_centered = traj_mat - traj_mat.mean(dim=0, keepdim=True)

        # PCA via SVD of centered trajectory
        U, S, Vh = torch.linalg.svd(traj_centered, full_matrices=False)

        k = min(n_components, S.numel())
        for i in range(k):
            if S[i] < 1e-10:
                break
            # Vh[i] is the i-th principal direction in block space [numel]
            pc_vec = Vh[i]

            # Embed in full parameter space
            gv = torch.zeros(total_params)
            gv[binfo["offset"]:binfo["offset"] + numel] = pc_vec
            basis_vecs.append(gv)

    if not basis_vecs:
        return None

    B = torch.stack(basis_vecs, dim=1)  # [P, K]
    B_ortho, _ = torch.linalg.qr(B, mode="reduced")
    return B_ortho


# ═══════════════════════════════════════════════════════════════════════════
# PROJECTION + RANDOM CONTROL (FIX #2)
# ═══════════════════════════════════════════════════════════════════════════

def projected_commutator(delta, B, normA, normB, eps=1e-12):
    """
    Project commutator delta onto basis B.
    Returns proj_norm, resid_norm, full_norm (all scale-normalized).
    """
    delta = delta.reshape(-1).cpu().float()

    if B is None or delta.numel() != B.shape[0]:
        full_val = (delta.norm() / (normA * normB + eps)).item()
        return {"proj": float("nan"), "resid": float("nan"), "full": full_val,
                "proj_raw": float("nan"), "resid_raw": float("nan"), "full_raw": delta.norm().item()}

    B = B.cpu().float()
    coeffs = B.T @ delta
    proj = B @ coeffs
    resid = delta - proj

    if hasattr(normA, 'cpu'):
        normA = normA.cpu()
    if hasattr(normB, 'cpu'):
        normB = normB.cpu()
    scale = float(normA * normB) + eps
    return {
        "proj": (proj.norm().item() / scale),
        "resid": (resid.norm().item() / scale),
        "full": (delta.norm().item() / scale),
        # Also raw (unnormalized) for clarity
        "proj_raw": proj.norm().item(),
        "resid_raw": resid.norm().item(),
        "full_raw": delta.norm().item(),
    }


def random_projection(delta, K, n_trials=10):
    """
    Project delta onto K random orthonormal directions.
    Returns mean proj_norm across trials.
    """
    delta = delta.reshape(-1).cpu().float()
    P = delta.numel()
    if K == 0 or K > P:
        return 0.0

    projs = []
    for _ in range(n_trials):
        G = torch.randn(P, K)
        Q, _ = torch.linalg.qr(G, mode="reduced")
        p = Q @ (Q.T @ delta)
        projs.append(p.norm().item())
    return float(np.mean(projs))


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING LOOP with corrected measurement
# ═══════════════════════════════════════════════════════════════════════════

def train_with_corrected_commutator(
    model, total_steps, batch_fn, device,
    comm_every=500, comm_K=9, comm_eta=1e-3,
    pca_components=3, n_random_trials=10,
    log_every=500,
):
    """
    Train model and measure corrected commutator projections.

    Key fixes:
    - Basis B built from PCA of weight trajectory (updated periodically)
    - Random baseline computed at every measurement
    - Projection at every measurement step (not just after step 5000)
    """
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()
    model.train()

    weight_log = []
    comm_records = []
    train_records = []
    B_pca = None
    t0 = time.time()

    for step in range(total_steps):
        # Training step
        x, y, _ = batch_fn()
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        loss = loss_fn(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % log_every == 0:
            # Save weight snapshot
            w = extract_attention_blocks(model)
            weight_log.append((step, w))

            # Evaluate accuracy
            model.eval()
            with torch.no_grad():
                x_eval, y_eval, _ = batch_fn()
                x_eval, y_eval = x_eval.to(device), y_eval.to(device)
                logits_eval, attn = model(x_eval, return_attn=True)
                acc = (logits_eval.argmax(-1) == y_eval).float().mean().item()
                ent, _ = bubble_metrics(attn[len(attn)-1], k=attn[len(attn)-1].shape[-1])
            model.train()

            train_records.append({
                "step": step,
                "loss": loss.item(),
                "acc": acc,
                "entropy": ent,
            })

            # Update PCA basis periodically (every 2000 steps, once we have enough data)
            if len(weight_log) >= 5 and step % 2000 == 0:
                B_pca = build_trajectory_pca_basis(
                    weight_log, model, n_components=pca_components
                )

            # Commutator measurements
            defects = []
            deltas = []
            for _ in range(comm_K):
                D, delta, gcos, nA, nB = commutator_defect(
                    model, batch_fn, device, eta=comm_eta
                )
                defects.append(D)
                deltas.append((delta, nA, nB))

            # Use median sample
            med_idx = torch.tensor(defects).argsort()[len(defects)//2]
            delta_med, nA_med, nB_med = deltas[med_idx]

            # Project onto PCA basis
            if B_pca is not None:
                K_dim = B_pca.shape[1]
                pc = projected_commutator(delta_med, B_pca, nA_med, nB_med)

                # Random baseline (same K)
                rand_proj_norm = random_projection(delta_med, K_dim, n_trials=n_random_trials)
                scale = float(nA_med.cpu() * nB_med.cpu()) + 1e-12
                rand_proj_normalized = rand_proj_norm / scale

                # Ratio: exec/random
                exec_frac = pc["proj_raw"] / (pc["full_raw"] + 1e-15)
                rand_frac = rand_proj_norm / (pc["full_raw"] + 1e-15)
                ratio = exec_frac / (rand_frac + 1e-15)
            else:
                pc = {"proj": float("nan"), "resid": float("nan"), "full": float("nan"),
                      "proj_raw": float("nan"), "resid_raw": float("nan"), "full_raw": float("nan")}
                rand_proj_normalized = float("nan")
                K_dim = 0
                exec_frac = float("nan")
                rand_frac = float("nan")
                ratio = float("nan")

            defect_med = float(np.median(defects))

            comm_records.append({
                "step": step,
                "defect_median": defect_med,
                "defect_p90": float(np.percentile(defects, 90)),
                # Projection results (normalized by ||eta*gA|| * ||eta*gB||)
                "proj_norm": pc["proj"],
                "resid_norm": pc["resid"],
                "full_norm": pc["full"],
                # Raw (unnormalized)
                "proj_raw": pc["proj_raw"],
                "resid_raw": pc["resid_raw"],
                "full_raw": pc["full_raw"],
                # Fraction of commutator in/out of basis
                "proj_frac": exec_frac,            # ||proj|| / ||delta||
                "resid_frac": 1.0 - exec_frac if np.isfinite(exec_frac) else float("nan"),
                # Random baseline
                "rand_proj_frac": rand_frac,       # ||rand_proj|| / ||delta||
                "exec_over_random": ratio,          # key metric
                "K": K_dim,
            })

            elapsed = (time.time() - t0) / 60
            if step % 2000 == 0:
                print(f"  step {step:6d} | loss={loss.item():.4f} | acc={acc:.3f} | "
                      f"def={defect_med:.1f} | pf={exec_frac:.3f} | "
                      f"rf={rand_frac:.3f} | ratio={ratio:.2f}x | {elapsed:.1f}m")

    return {
        "weight_log": weight_log,
        "comm_records": comm_records,
        "train_records": train_records,
    }


# ═══════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════

def fig1_commutator_decomposition(results):
    """
    Corrected Figure 5: explicit decomposition with random baseline.
    Shows proj_frac and resid_frac vs training step, plus random baseline.
    """
    cr = results["comm_records"]
    steps = [r["step"] for r in cr]
    proj_frac = [r["proj_frac"] for r in cr]
    resid_frac = [r["resid_frac"] for r in cr]
    rand_frac = [r["rand_proj_frac"] for r in cr]
    ratio = [r["exec_over_random"] for r in cr]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: proj fraction vs random fraction
    ax = axes[0, 0]
    ax.plot(steps, proj_frac, linewidth=2, color="#2ecc71", label="Exec basis (PCA)")
    ax.plot(steps, rand_frac, linewidth=2, color="#e74c3c", linestyle="--", label="Random basis")
    ax.set_xlabel("Training step")
    ax.set_ylabel("||proj|| / ||delta||")
    ax.set_title("A. Fraction of commutator in subspace")
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel B: resid fraction (should be ~1 if commutator avoids execution subspace)
    ax = axes[0, 1]
    ax.plot(steps, resid_frac, linewidth=2, color="#3498db",
            label=r"$\|[\nabla_A,\nabla_B]_\perp\| / \|[\nabla_A,\nabla_B]\|$")
    ax.axhline(y=1.0, color="red", linestyle=":", linewidth=1.5, alpha=0.5)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Residual fraction")
    ax.set_title("B. Perpendicular component fraction")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel C: exec/random ratio
    ax = axes[1, 0]
    valid = [(s, r) for s, r in zip(steps, ratio) if np.isfinite(r)]
    if valid:
        vs, vr = zip(*valid)
        ax.plot(vs, vr, linewidth=2, color="#9b59b6")
    ax.axhline(y=1.0, color="red", linestyle=":", linewidth=2, label="Random = 1.0")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Exec / Random ratio")
    ax.set_title("C. Exec-basis vs random-basis projection")
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel D: raw defect over training
    ax = axes[1, 1]
    defects = [r["defect_median"] for r in cr]
    ax.plot(steps, defects, linewidth=2, color="#e67e22")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Defect (normalized)")
    ax.set_title("D. Commutator defect over training")
    ax.grid(alpha=0.3)

    fig.suptitle("SGD Commutator Decomposition\n"
                 "Basis = PCA of weight trajectory | Random control included",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_corrected_commutator_decomp.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_corrected_commutator_decomp.png")


def fig2_rho_ratio(results):
    """
    Corrected Figure 4: rho = proj/full over training.
    Now clearly labeled and with random baseline √(K/P).
    """
    cr = results["comm_records"]
    steps = [r["step"] for r in cr]
    proj_frac = [r["proj_frac"] for r in cr]
    rand_frac = [r["rand_proj_frac"] for r in cr]

    # Theoretical random expectation: sqrt(K/P) for Gaussian random projection
    K_vals = [r["K"] for r in cr]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, proj_frac, linewidth=2, color="#2ecc71",
            label=r"$\rho_{\mathrm{exec}} = \|\delta_\parallel\| / \|\delta\|$")
    ax.plot(steps, rand_frac, linewidth=2, color="#e74c3c", linestyle="--",
            label=r"$\rho_{\mathrm{rand}}$ (random K-dim basis)")
    ax.set_xlabel("Training step")
    ax.set_ylabel(r"$\rho$")
    ax.set_title(r"Projection fraction $\rho$ — exec basis vs random")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_corrected_rho.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_corrected_rho.png")


def fig3_training_overview(results):
    """Training loss, accuracy, entropy, and defect."""
    tr = results["train_records"]
    cr = results["comm_records"]

    steps_tr = [r["step"] for r in tr]
    loss = [r["loss"] for r in tr]
    acc = [r["acc"] for r in tr]
    ent = [r["entropy"] for r in tr]

    steps_cr = [r["step"] for r in cr]
    defect = [r["defect_median"] for r in cr]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0,0].plot(steps_tr, loss, linewidth=1.5)
    axes[0,0].set_title("Training loss")
    axes[0,0].set_xlabel("Step")
    axes[0,0].grid(alpha=0.3)

    axes[0,1].plot(steps_tr, acc, linewidth=1.5, color="green")
    axes[0,1].set_title("Accuracy")
    axes[0,1].set_xlabel("Step")
    axes[0,1].grid(alpha=0.3)

    axes[1,0].plot(steps_tr, ent, linewidth=1.5, color="purple")
    axes[1,0].set_title("Attention entropy (EOS)")
    axes[1,0].set_xlabel("Step")
    axes[1,0].grid(alpha=0.3)

    axes[1,1].plot(steps_cr, defect, linewidth=1.5, color="orange")
    axes[1,1].set_title("Commutator defect")
    axes[1,1].set_xlabel("Step")
    axes[1,1].grid(alpha=0.3)

    fig.suptitle("Training Overview", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_training_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_training_overview.png")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    OUT_DIR.mkdir(exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # Config matching the original paper
    T = 16
    vocab = 256
    n_classes = 8
    batch_size = 128
    d_model = 128
    n_layers = 5
    n_heads = 4
    m_schedule = [1, 2, 3, 4]
    total_steps = 20_000
    seed = 42

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    model = BubbleTransformer(
        vocab, d_model, n_classes, n_layers, n_heads, max_seq_len=T
    ).to(device)

    def batch_fn():
        m = random.choice(m_schedule)
        return sample_m_bubble_batch(batch_size, T, m, vocab, n_classes)

    print(f"\nTraining for {total_steps} steps with corrected commutator analysis...")
    print(f"  Mixed-uniform m={m_schedule}, seed={seed}")
    print(f"  Model: {n_layers}L, d={d_model}, {n_heads}H")
    print()

    results = train_with_corrected_commutator(
        model, total_steps, batch_fn, device,
        comm_every=500, comm_K=9, comm_eta=1e-3,
        pca_components=3, n_random_trials=10,
        log_every=500,
    )

    # Print summary
    cr = results["comm_records"]
    valid = [r for r in cr if np.isfinite(r["exec_over_random"])]
    if valid:
        # Early (first 20%)
        n = len(valid)
        early = valid[:n//5]
        late = valid[-n//5:]

        if early:
            e_ratio = np.median([r["exec_over_random"] for r in early])
            e_pf = np.median([r["proj_frac"] for r in early])
            print(f"\n  Early  (first 20%): exec/rand = {e_ratio:.2f}x, proj_frac = {e_pf:.3f}")
        if late:
            l_ratio = np.median([r["exec_over_random"] for r in late])
            l_pf = np.median([r["proj_frac"] for r in late])
            print(f"  Late   (last  20%): exec/rand = {l_ratio:.2f}x, proj_frac = {l_pf:.3f}")

    # Generate figures
    print("\nGenerating corrected figures...")
    fig1_commutator_decomposition(results)
    fig2_rho_ratio(results)
    fig3_training_overview(results)

    # Save results
    save_path = OUT_DIR / "corrected_commutator_results.pt"
    torch.save(results, save_path)
    print(f"\nResults saved to {save_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
