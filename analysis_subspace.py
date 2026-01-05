import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

RUN_FILE = "runs.pt"
BLOCK_NAME = "layer0.WO"
K = 16

def flatten(t):
    return t.reshape(-1).float().numpy()

def load_runs(path):
    return torch.load(path, map_location="cpu")

runs = load_runs(RUN_FILE)

def analyze_single_run(run, seed_id=0, show=True):
    weights = run["weights"]

    # collect trajectory
    W = []
    w0 = weights[0][1][BLOCK_NAME]

    for rec in weights:
        wt = rec[1][BLOCK_NAME]
        W.append(flatten(wt - w0))

    W = np.stack(W)
    W -= W.mean(axis=0, keepdims=True)

    # PCA on this run only
    k = min(K, W.shape[0], W.shape[1])
    pca = PCA(n_components=k)
    Z = pca.fit_transform(W)

    if show:
        plt.figure(figsize=(6,4))
        plt.plot(Z[:,0], label="PC1")
        plt.plot(Z[:,1], label="PC2")
        plt.title(f"Seed {seed_id} — PCA trajectory")
        plt.xlabel("Training step")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {
        "Z": Z,
        "explained_var": pca.explained_variance_ratio_,
    }


def main(show=True):
    if len(runs) == 0:
        print(f"No runs found in {RUN_FILE}")
        return

    explained = []
    k_used = None
    for seed_id, run in enumerate(runs):
        res = analyze_single_run(run, seed_id=seed_id, show=show)
        explained.append(res["explained_var"])
        k_used = len(res["explained_var"]) if k_used is None else min(k_used, len(res["explained_var"]))
        print(f"Run {seed_id}: explained variance ratio (top {len(res['explained_var'])} PCs): {res['explained_var']}")

    exp = np.stack([e[:k_used] for e in explained])
    print(f"PCA explained variance ratio (mean ± std) across {exp.shape[0]} runs (top {exp.shape[1]} PCs):")
    for i in range(exp.shape[1]):
        mu, sigma = exp[:, i].mean(), exp[:, i].std()
        print(f"  PC{i+1}: {mu:.4f} ± {sigma:.4f}")


if __name__ == "__main__":
    main()
