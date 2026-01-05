import torch
import numpy as np
import random
from collections import defaultdict
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# -------------------------
# CONFIG
# -------------------------
RUN_FILE = "runs.pt"
LAYERS = [0, 1, 2, 3, 4]
BLOCKS = ["WQ", "WK", "WV", "WO"]

# -------------------------
# HELPERS
# -------------------------
def flatten(x):
    return x.reshape(-1).float()

def load_runs(path):
    return torch.load(path, map_location="cpu")

def alignment_to_final(run, block_name):
    weights = run["weights"]
    w0 = weights[0][1][block_name]
    wT = weights[-1][1][block_name]

    vT = flatten(wT - w0)
    vT = vT / (vT.norm() + 1e-12)

    A = []
    for rec in weights:
        wt = rec[1][block_name]
        vt = flatten(wt - w0)
        vt = vt / (vt.norm() + 1e-12)
        A.append(float((vt @ vT).clamp(-1, 1)))
    return np.array(A)

def effective_rank(run, block_name):
    w0 = run["weights"][0][1][block_name]
    X = []
    for rec in run["weights"]:
        wt = rec[1][block_name]
        X.append(flatten(wt - w0).numpy())
    X = np.stack(X)

    X = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False)
    p = s / (s.sum() + 1e-12)
    return float(np.exp(-(p * np.log(p + 1e-12)).sum()))

def pca_spectrum(run, block_name, k=8):
    w0 = run["weights"][0][1][block_name]
    X = []
    for rec in run["weights"]:
        wt = rec[1][block_name]
        X.append(flatten(wt - w0).numpy())
    X = np.stack(X)
    X -= X.mean(axis=0, keepdims=True)

    pca = PCA(n_components=k)
    pca.fit(X)
    return pca.explained_variance_ratio_


LAYERS = [0,1,2,3,4]
BLOCKS = ["WQ", "WK", "WV", "WO"]

def plot_effective_rank(results):
    plt.figure(figsize=(7,5))

    for block in BLOCKS:
        means = []
        stds = []
        for layer in LAYERS:
            r = np.array(results[(layer, block)]["eff_rank"])
            means.append(r.mean())
            stds.append(r.std())

        plt.errorbar(
            LAYERS, means, yerr=stds,
            marker="o", capsize=3, label=block
        )

    plt.xlabel("Layer")
    plt.ylabel("Effective Rank")
    plt.title("Effective Dimensionality vs Layer")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def rank_at_90(pca_spectrum):
    cumsum = np.cumsum(pca_spectrum)
    return int(np.searchsorted(cumsum, 0.9) + 1)

def plot_rank90(results):
    plt.figure(figsize=(7,5))

    for block in BLOCKS:
        means = []
        stds = []
        for layer in LAYERS:
            ranks = [
                rank_at_90(pca)
                for pca in results[(layer, block)]["pca"]
            ]
            means.append(np.mean(ranks))
            stds.append(np.std(ranks))

        plt.errorbar(
            LAYERS, means, yerr=stds,
            marker="o", capsize=3, label=block
        )

    plt.xlabel("Layer")
    plt.ylabel("Rank for 90% Variance")
    plt.title("Intrinsic Dimensionality Across Layers")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_final_alignment(results):
    plt.figure(figsize=(7,5))

    for block in BLOCKS:
        means = []
        stds = []
        for layer in LAYERS:
            vals = np.array(results[(layer, block)]["final_alignment"])
            means.append(vals.mean())
            stds.append(vals.std())

        plt.errorbar(
            LAYERS, means, yerr=stds,
            marker="o", capsize=3, label=block
        )

    plt.xlabel("Layer")
    plt.ylabel("Final alignment with w_T")
    plt.title("Final alignment vs layer")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def shuffled_effective_rank(run, block):
    weights = run["weights"]

    # collect deltas
    w0 = weights[0][1][block]
    deltas = [
        (rec[1][block] - w0).reshape(-1).float().numpy()
        for rec in weights
    ]

    # shuffle time order
    random.shuffle(deltas)

    X = np.stack(deltas)
    X = X - X.mean(axis=0, keepdims=True)

    s = np.linalg.svd(X, compute_uv=False)
    p = s / (s.sum() + 1e-12)
    return float(np.exp(-(p * np.log(p + 1e-12)).sum()))

# -------------------------
# MAIN ANALYSIS
# -------------------------
runs = load_runs(RUN_FILE)

results = defaultdict(lambda: defaultdict(list))
alignments = {} 
k=0

for run in runs:
    for layer in LAYERS:
        for block in BLOCKS:
            name = f"layer{layer}.{block}"

            # alignment curve
            A = alignment_to_final(run, name)
            results[(layer, block)]["final_alignment"].append(A[-1])
            key = (k, name)
            alignments[key] = A
            plt.plot(A, label=name)
            k += 1

            # effective rank
            r_eff = effective_rank(run, name)
            results[(layer, block)]["eff_rank"].append(r_eff)

            # PCA spectrum
            pca = pca_spectrum(run, name)
            results[(layer, block)]["pca"].append(pca)

# -------------------------
# SUMMARY
# -------------------------
print("\n===== SUMMARY (mean ± std across seeds) =====\n")

for layer in LAYERS:
    for block in BLOCKS:
        key = (layer, block)
        pcas = np.array(results[key]["pca"])
        r_eff = np.array(results[key]["eff_rank"])
        align = np.array(results[key]["final_alignment"])

        print(f"{key}:")
        print(f"  eff_rank: {r_eff.mean():.2f} ± {r_eff.std():.2f}")
        print(f"  final alignment: {align.mean():.3f} ± {align.std():.3f}")
        print(f"  PCA:")
        for i in range(5):
            print(f"    PC{i+1}: {pcas[:,i].mean():.4f} ± {pcas[:,i].std():.4f}")
        print()

plot_effective_rank(results)
plot_rank90(results)
plot_final_alignment(results)
