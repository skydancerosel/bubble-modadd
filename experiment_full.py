# ================================
# experiment_full.py
# ================================

from math import *
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from helpers import *

# ----------------------------
# MAIN EXPERIMENT
# ----------------------------

def run(check_points=None):
    check_points = check_points or {}
    class Cfg:
        T = 16
        vocab = 256
        n_classes = 8
        batch_size = 128

        # Training schedule
        schedule = "mixed_uniform"   # or "curriculum"
        #schedule = "curriculum"      # or "mixed_uniform"
        m_schedule = [1, 2, 3, 4]    # used by curriculum + probes
        steps_per_stage = 5_000     # curriculum only
        total_steps = 20_000         # mixed only (match 4*20k)

        ks = [16, 8, 4, 2, 1]
        d_model = 128
        n_layers =5
        n_heads = 4
        device = "mps" if torch.mps.is_available() else "cpu"

    cfg = Cfg()

        # Fixed probe batches for entropy tracking (same x each time)
    probe_batches = {}
    for m in cfg.m_schedule:
        x_probe, y_probe, _ = sample_m_bubble_batch(
            cfg.batch_size, cfg.T, m, cfg.vocab, cfg.n_classes
        )
        probe_batches[m] = (x_probe.to(cfg.device), y_probe.to(cfg.device))

    model = BubbleTransformer(
            cfg.vocab,
            cfg.d_model,
            cfg.n_classes,
            cfg.n_layers,
            cfg.n_heads,
            max_seq_len=cfg.T
        ).to(cfg.device)

    if cfg.schedule == "curriculum":
        all_entropy = {}  # {m: [(global_step, ent), ...]}
        global_step_offset = 0
        all_weight = []
        all_comm = []
        all_probe = []

        for m in cfg.m_schedule:
            print(f"\n=== Training stage: m = {m} ===")
            def batch_fn():
                return sample_m_bubble_batch(
                    cfg.batch_size, cfg.T, m, cfg.vocab, cfg.n_classes
                )

            model, entropy_logs, weight_log, comm_logs, comm_logs_sorted, comm_logs_proj, probe_logs = train(
                model, cfg.steps_per_stage, batch_fn, cfg.device,
                probe_batches=probe_batches, log_every=2000, check_points=check_points
            )
            all_weight.extend([(step + global_step_offset, w) for (step, w) in weight_log])
            all_comm.extend([
                (step + global_step_offset, D, delta, mA, mB, gcos)
                for (step, D, delta, mA, mB, gcos) in comm_logs
            ])
            all_probe.extend([(step + global_step_offset, *p[1:]) for p in probe_logs])
            # shift steps to global axis
            for mm, series in entropy_logs.items():
                all_entropy.setdefault(mm, [])
                all_entropy[mm].extend([(global_step_offset + s, e) for (s, e) in series])

            global_step_offset += cfg.steps_per_stage

              # ---- ENTROPY VS TRAINING STEP ----
                  

    elif cfg.schedule == "mixed_uniform":
        def batch_fn():
            m = random.choice(cfg.m_schedule)  # uniform over {1,2,3,4}
            return sample_m_bubble_batch(
                cfg.batch_size, cfg.T, m, cfg.vocab, cfg.n_classes
            )

        print(f"\n=== Training mixed-uniform over m={cfg.m_schedule} for {cfg.total_steps} steps ===")
        model, entropy_logs, weight_log, comm_logs, comm_logs_sorted, comm_logs_proj, probe_logs = train(
            model, cfg.total_steps, batch_fn, cfg.device,
            probe_batches=probe_batches, log_every=500, check_points=check_points)
        
        all_entropy = entropy_logs
        all_weight = weight_log
        all_comm = comm_logs
        all_probe = probe_logs

    else:
        raise ValueError(f"Unknown schedule: {cfg.schedule}")
    
        # ---- ENTROPY VS TRAINING STEP (per m) ----
    '''
    plt.figure(figsize=(7,4))
    for m in cfg.m_schedule:
        if m in all_entropy and len(all_entropy[m]) > 0:
            steps_, ents_ = zip(*all_entropy[m])
            plt.plot(steps_, ents_, label=f"m={m}")
    plt.xlabel("Training step")
    plt.ylabel("Attention entropy (EOS, last layer)")
    plt.title(f"Entropy vs step ({cfg.schedule})")
    plt.legend()
    plt.tight_layout()
    plt.show()
    '''
    accs, ents = [], []
    for k in cfg.ks:
        acc, ent, _ = eval_with_k(model, batch_fn, k, cfg.device)
        accs.append(acc)
        ents.append(ent)
        print(f"k={k:2d} | acc={acc:.4f} | entropy={ent:.4f}")
    '''
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(cfg.ks, accs, marker="o")
    plt.gca().invert_xaxis()
    plt.title("Accuracy vs k")

    plt.subplot(1,2,2)
    plt.plot(cfg.ks, ents, marker="o")
    plt.gca().invert_xaxis()
    plt.title("Entropy vs k")
    '''
    return model, all_entropy, all_weight, all_comm, comm_logs_sorted, comm_logs_proj, all_probe, {
        "ks": cfg.ks,
        "accs": accs,
        "ents": ents,
        "comm_logs": all_comm,
        "probe_logs": all_probe
    }  

if __name__ == "__main__":

    all_runs = []
    seeds = range(12, 13)

    SAE_CHECKPOINTS = {
        "early": 2000,
        "mid": 8000,
        "late": 18000,
    }

    for seed in seeds:
        torch.manual_seed(seed)
        model, entropy, weight_log, comm_logs, comm_logs_sorted, comm_logs_proj, probe_logs, metrics = run(SAE_CHECKPOINTS)
        # ---- unpack logs ----
        # comm_logs: [(step, defect, delta, mA, mB, gcos), ...]
        # probe_logs: [(step, cosine), ...]   # label ↔ m
        # entropy_logs: {m: [(step, entropy), ...]}

        steps_comm_raw = [t[0] for t in comm_logs]
        defects_raw = [t[1] for t in comm_logs]
        steps_probe, probe_cos = zip(*probe_logs)

        # Choose one representative m for entropy (e.g. largest m)
        m_ref = max(entropy.keys())
        steps_ent, ent_vals = zip(*entropy[m_ref])


        # Use the same device as model
        device = next(model.parameters()).device

        # mixed-m batch fn (same as your training)
        def batch_fn():
            m = random.choice([1,2,3,4])
            return sample_m_bubble_batch(128, 16, m, 256, 8)

        H_layers, Y, M = collect_layer_eos(model, batch_fn, device, n_batches=40)

        L = len(H_layers)
        acc_m = []
        acc_y = []

        for i in range(L):
            acc_m.append(probe_accuracy(H_layers[i], M, n_classes=5))  # m in {1..4}; +1 safe
            acc_y.append(probe_accuracy(H_layers[i], Y, n_classes=8))  # y in {0..7}

        plt.figure(figsize=(6,4))
        plt.plot(range(L), acc_m, marker="o", label="Probe acc for m")
        plt.plot(range(L), acc_y, marker="o", label="Probe acc for y")
        plt.xlabel("Layer index (after AttentionBlock)")
        plt.ylabel("Probe accuracy")
        plt.title("Where task structure (m) and answer (y) live across layers")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # ---- plot ----
        steps = [t[0] for t in comm_logs_proj]
        full  = [t[1] for t in comm_logs_proj]
        proj  = [t[2] for t in comm_logs_proj]
        resid = [t[3] for t in comm_logs_proj]

        # ---- unpack logs ----
        steps_comm_med = [t[0] for t in comm_logs_sorted]
        comm_same = [t[1] for t in comm_logs_sorted]
        comm_cross = [t[2] for t in comm_logs_sorted]
        gcos= [t[3] for t in comm_logs_sorted   ]

        steps_probe, probe_cos = zip(*probe_logs)

        # pick largest m for entropy
        m_ref = max(entropy.keys())
        steps_ent, ent_vals = zip(*entropy[m_ref])

        # Figure 2: attention entropy collapse across m
        plt.figure(figsize=(7, 5))
        for m_key, series in sorted(entropy.items()):
            s, e = zip(*series)
            plt.plot(s, e, label=f"m={m_key}", linewidth=2)

        plt.xlabel("Training step")
        plt.ylabel("EOS attention entropy")
        plt.title("Figure 2: Attention entropy collapse and emergence of attention bubbling")
        plt.figtext(0.5, -0.08, "Attention entropy at the final token decreases sharply during training across task difficulties, signalling highly concentrated attention ('bubbling') that coincides with collapsed parameter dynamics.", ha="center", wrap=True, fontsize=9)
        plt.legend()
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig("figure2_entropy_bubbling.png", dpi=200, bbox_inches="tight")
        plt.show()

        # Figure 3: raw commutator norm vs step
        plt.figure(figsize=(6, 4))
        plt.plot(steps_comm_raw, defects_raw, label="Raw commutator norm", linewidth=2, color="tab:purple")
        plt.xlabel("Training step")
        plt.ylabel("Commutator norm (normalized)")
        plt.title("Figure 3: Persistent non-commutativity of SGD updates")
        plt.figtext(0.5, -0.12, "Despite loss convergence and stable attention, SGD updates remain strongly non-commutative in ambient parameter space, with frequent spikes indicating path-dependent dynamics.", ha="center", wrap=True, fontsize=9)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig("figure3_commutator_raw.png", dpi=200, bbox_inches="tight")
        plt.show()

        # Figure 4: ratio projected/full commutator
        ratio = []
        for f, p in zip(full, proj):
            if not (isnan(f) or f == 0):
                ratio.append(p / f)
            else:
                ratio.append(float("nan"))
        plt.figure(figsize=(6, 4))
        plt.plot(steps, ratio, label="‖Proj(commutator)‖ / ‖commutator‖", linewidth=2, color="tab:cyan")
        plt.xlabel("Training step")
        plt.ylabel("Ratio")
        plt.title("Figure 4: Projected commutator magnitude relative to full commutator")
        plt.figtext(0.5, -0.12, "Projecting commutators onto the learned execution subspace drives their magnitude near zero, showing non-integrability is largely orthogonal to execution.", ha="center", wrap=True, fontsize=9)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig("figure4_comm_ratio.png", dpi=200, bbox_inches="tight")
        plt.show()

        # Figure 5: decomposition of commutator components
        plt.figure(figsize=(6,4))
        plt.plot(steps, full,  label="Full commutator", linewidth=2)
        plt.plot(steps, proj,  label="Projected (execution)", linewidth=2)
        plt.plot(steps, resid, label="Residual (orthogonal)", linewidth=2)

        plt.xlabel("Training step")
        plt.ylabel("Normalized commutator")
        plt.title("Figure 5: Decomposition of SGD commutators into execution and orthogonal components")
        plt.figtext(0.5, -0.12, "Nearly all noncommutativity lives in directions orthogonal to execution; the projected component stays near zero, indicating near-commuting execution dynamics.", ha="center", wrap=True, fontsize=9)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("figure5_comm_decomposition.png", dpi=200, bbox_inches="tight")
        plt.show()

        # Collect per-(mA,mB) lists
        ms = sorted(list(set(entropy.keys())))
        m_to_idx = {m:i for i, m in enumerate(ms)}
        K = len(ms)
        cell_def = [[[] for _ in range(K)] for _ in range(K)]
        cell_cos = [[[] for _ in range(K)] for _ in range(K)]

        for (_, D_i, _, mA_i, mB_i, gcos_i) in comm_logs:
            if mA_i in m_to_idx and mB_i in m_to_idx:
                ia, ib = m_to_idx[mA_i], m_to_idx[mB_i]
                cell_def[ia][ib].append(D_i)
                cell_cos[ia][ib].append(gcos_i)

        def median_or_nan(xs):
            if len(xs) == 0:
                return float("nan")
            xs = sorted(xs)
            mid = len(xs)//2
            if len(xs) % 2 == 1:
                return float(xs[mid])
            return float(0.5*(xs[mid-1] + xs[mid]))

        mat_def = [[median_or_nan(cell_def[i][j]) for j in range(K)] for i in range(K)]
        mat_cos = [[median_or_nan(cell_cos[i][j]) for j in range(K)] for i in range(K)]
        
        all_runs.append({
            "seed": seed,
            "entropy": entropy,
            "weights": weight_log,
            "comm_logs": comm_logs,
            "comm_logs_sorted": comm_logs_sorted,
            "comm_logs_proj": comm_logs_proj,   
            "probe_logs": probe_logs,
            "metrics": metrics
        })

        print("\n=== Running SAE demo ===")

        device = next(model.parameters()).device

        def make_fixed_batch_fn():
            def fn():
                #m = random.choice([1, 2, 3, 4])
                m=4
                return sample_m_bubble_batch(128, 16, m, 256, 8)
            return fn

        fixed_batch_fn = make_fixed_batch_fn()


        X, M = collect_eos_activations(
            model, fixed_batch_fn, device, n_batches=40
        )

        sae = TopKSAE(d_in=X.shape[1], d_sae=256, k=16)
        sae, sae_logs = train_sae(sae, X, device)

        print("\n=== SAE checkpoint ablation experiment ===")

        ablation_results = {}

        for tag, step in SAE_CHECKPOINTS.items():
            print(f"\n--- Checkpoint: {tag} (step {step}) ---")

            # load checkpoint
            model.load_state_dict(torch.load(f"ckpt_step_{step}.pt"))
            model.eval()

            # collect token activations
            X_tok = collect_token_activations(
                model,
                fixed_batch_fn,
                device,
                n_batches=40
            )

            # train SAE
            sae = TopKSAE(
                d_in=X_tok.shape[1],
                d_sae=256,
                k=16
            ).to(device)

            sae, _ = train_sae(
                sae,
                X_tok,
                device,
                steps=2500
            )

            # find execution latents
            exec_latents = top_variance_latents(sae, X_tok, topn=12)
            print("Execution latents:", exec_latents[:6])

            # ablate
            acc_before, acc_after = eval_with_token_sae_ablation(
                model,
                sae,
                fixed_batch_fn,
                device,
                exec_latents[:6]
            )

                # find execution latents
            exec_latents = top_variance_latents(sae, X_tok, topn=12)
            print("Execution latents:", exec_latents[:6])

            # ablate
            acc_before, acc_after = eval_with_token_sae_ablation(
                model,
                sae,
                fixed_batch_fn,
                device,
                exec_latents[:6]
            )

            print(f"Accuracy before: {acc_before:.3f}")
            print(f"Accuracy after : {acc_after:.3f}")

            ablation_results[tag] = (acc_before, acc_after)

            print("\n=== Ablation summary ===")
            for tag, (a0, a1) in ablation_results.items():
                print(f"{tag:>5s} | before={a0:.3f} | after={a1:.3f} | drop={a0 - a1:.3f}")

        # Figure 6: SAE ablation accuracy before/after
        if ablation_results:
            tags = list(ablation_results.keys())
            a0s = [v[0] for v in ablation_results.values()]
            a1s = [v[1] for v in ablation_results.values()]
            x = np.arange(len(tags))
            width = 0.35

            plt.figure(figsize=(6,4))
            plt.bar(x - width/2, a0s, width, label="Before ablation", color="tab:gray")
            plt.bar(x + width/2, a1s, width, label="After ablation", color="tab:orange")
            plt.xticks(x, tags)
            plt.ylim(0, 1.05)
            plt.xlabel("Checkpoint")
            plt.ylabel("Accuracy")
            plt.title("Figure 6: SAE probes capture auxiliary structure but not execution")
            plt.figtext(0.5, -0.16, "SAE latents correlate with task structure, but ablating them causes only small, stage-dependent drops—execution remains distributed across a low-dimensional manifold.", ha="center", wrap=True, fontsize=9)
            plt.legend()
            plt.tight_layout()
            plt.savefig("figure6_sae_ablation.png", dpi=200, bbox_inches="tight")
            plt.show()
        '''
        for m, series in entropy.items():
            steps, ents = zip(*series)
            plt.plot(steps, ents, label=f"m={m}")

        plt.xlabel("Training step")
        plt.ylabel("EOS entropy")
        plt.title("Entropy vs training step (by m)")
        plt.legend()
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()
        plt.savefig("bubble_circuit_transition.png", dpi=200)
        for m, series in entropy.items():
            steps, ents = zip(*series)
            plt.plot(steps, ents, label=f"m={m}")

        plt.xlabel("Training step")
        plt.ylabel("EOS entropy")
        plt.title("Entropy vs training step (by m)")
        plt.legend()
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()
        plt.savefig("bubble_circuit_transition.png", dpi=200)


        # interpolate entropy to commutator steps
        from scipy.interpolate import interp1d

        ent_interp = interp1d(
            steps_ent,
            ent_vals,
            bounds_error=False,
            fill_value="extrapolate"
        )

        ent_at_comm = ent_interp(steps_comm_raw)

        plt.figure(figsize=(5, 4))
        plt.scatter(ent_at_comm, defects_raw, s=20, alpha=0.7)

        plt.xlabel("EOS entropy")
        plt.ylabel("Commutator defect")
        plt.title("Geometry vs attention disorder")
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

        # ---- heatmaps: commutator defect and gradient cosine by (mA, mB) ----
        ms = sorted(list(set(m for m in entropy.keys())))  # typically [1,2,3,4]
        m_to_idx = {m:i for i, m in enumerate(ms)}
        K = len(ms)

        # Heatmap 1: median commutator defect
        plt.figure(figsize=(5, 4))
        plt.imshow(mat_def, aspect="auto")
        plt.colorbar(label="Median commutator defect")
        plt.xticks(range(K), [f"mB={m}" for m in ms], rotation=45, ha="right")
        plt.yticks(range(K), [f"mA={m}" for m in ms])
        plt.title("Commutator defect by (mA, mB)")
        plt.tight_layout()
        plt.show()

        # Heatmap 2: median gradient cosine similarity
        plt.figure(figsize=(5, 4))
        plt.imshow(mat_cos, aspect="auto", vmin=-1.0, vmax=1.0)
        plt.colorbar(label="Median cos(gA, gB)")
        plt.xticks(range(K), [f"mB={m}" for m in ms], rotation=45, ha="right")
        plt.yticks(range(K), [f"mA={m}" for m in ms])
        plt.title("Gradient cosine by (mA, mB)")
        plt.tight_layout()
        plt.show()
        '''
    torch.save(all_runs, "single_run_full.pt") 
