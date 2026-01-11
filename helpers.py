from math import *
import random
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ----------------------------
# DATA
# ----------------------------

def _sample_positions(T, m):
    if m > 1:
        while True:
            pos = sorted(random.sample(range(1, T - 1), m))
            if min(pos[i+1] - pos[i] for i in range(m - 1)) > 1:
                return pos
    return sorted(random.sample(range(1, T - 1), m))


def sample_m_bubble_batch(B, T, m, vocab, n_classes, marker_offset=100):
    # Background tokens should NOT include:
    # - value tokens: [0, n_classes-1]
    # - marker tokens: [marker_offset, marker_offset + m - 1]
    # Otherwise the model sees many spurious "values/markers" and m=4 becomes dominated by noise.
    forbid = set(range(0, n_classes)) | set(range(marker_offset, marker_offset + m))
    allowed = [i for i in range(vocab) if i not in forbid]
    x = torch.tensor(random.choices(allowed, k=B * T), dtype=torch.long).view(B, T)
    y = torch.zeros(B, dtype=torch.long)

    positions = _sample_positions(T, m)

    for b in range(B):
        vals = []
        for i, p in enumerate(positions):
            marker = marker_offset + i
            val = torch.randint(0, n_classes, (1,)).item()
            x[b, p] = marker
            x[b, p + 1] = val
            vals.append(val)

        y[b] = sum(vals) % n_classes

    return x, y, positions


# ----------------------------
# MODEL
# ----------------------------

class AttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x, attn_override=None):
        B, T, D = x.shape
        h = self.ln(x)

        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / sqrt(self.d_head)
        mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
        scores = scores.masked_fill(~mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)

        if attn_override is not None:
            attn = attn_override

        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.out(out)
        #x = x + self.mlp(x)
        return x, attn


class BubbleTransformer(nn.Module):
    def __init__(self, vocab, d_model, n_classes, n_layers=2, n_heads=4, max_seq_len=128):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.max_seq_len = max_seq_len
        self.layers = nn.ModuleList([
            AttentionBlock(d_model, n_heads)
            for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x, attn_override=None, return_attn=False, return_h=False):
        B, T = x.shape
        if T > self.max_seq_len:
            raise ValueError(f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}")
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.embed(x) + self.pos_emb(pos)
        all_attn = {}

        for i, layer in enumerate(self.layers):
            override = None if attn_override is None else attn_override.get(i)
            h, attn = layer(h, override)
            if return_attn:
                all_attn[i] = attn

        h = self.ln(h)
        logits = self.head(h[:, -1])
        if return_h:
            return logits, all_attn, h
        return logits, all_attn

# ----------------------------
# SPARSE AUTOENCODER (Top-K SAE)
# ----------------------------

class TopKSAE(nn.Module):
    """
    OpenAI-style Top-K Sparse Autoencoder.
    Encodes residual-stream activations into k-sparse latent features.
    """
    def __init__(self, d_in, d_sae, k=32):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.k = k

        self.b_pre = nn.Parameter(torch.zeros(d_in))
        self.W_enc = nn.Parameter(torch.randn(d_sae, d_in) * 0.02)
        self.W_dec = nn.Parameter(torch.randn(d_in, d_sae) * 0.02)
        self.b_dec = nn.Parameter(torch.zeros(d_in))

    def topk(self, z):
        if self.k >= z.shape[-1]:
            return z
        vals, idx = torch.topk(z, self.k, dim=-1)
        out = torch.zeros_like(z)
        out.scatter_(-1, idx, vals)
        return out

    def forward(self, x):
        # x: [N, d_in]
        x0 = x - self.b_pre
        z_pre = x0 @ self.W_enc.t()
        z = self.topk(z_pre)
        x_hat = z @ self.W_dec.t() + self.b_dec + self.b_pre
        return x_hat, z, z_pre

# ----------------------------
# METRICS
# ----------------------------

def apply_topk(attn, k):
    if k >= attn.shape[-1]:
        return attn
    idx = attn.topk(k, dim=-1).indices
    mask = torch.zeros_like(attn, dtype=torch.bool)
    mask.scatter_(-1, idx, True)
    out = torch.where(mask, attn, torch.zeros_like(attn))
    return out / out.sum(dim=-1, keepdim=True).clamp_min(1e-9)


@torch.no_grad()
def bubble_metrics(attn, k):
    A = attn[:, :, -1]  # EOS attention
    entropy = -(A * (A + 1e-9).log()).sum(-1).mean().item()
    topk = A.topk(min(k, A.shape[-1]), -1).values.sum(-1).mean().item()
    return entropy, topk


def extract_attention_blocks(model):
    """
    Extract Q, K, V, O weight blocks from each attention layer.
    Returns a dict of tensors.
    """
    blocks = {}

    for i, layer in enumerate(model.layers):
        W = layer.qkv.weight.detach().cpu()

        d = W.shape[0] // 3
        blocks[f"layer{i}.WQ"] = W[:d]
        blocks[f"layer{i}.WK"] = W[d:2*d]
        blocks[f"layer{i}.WV"] = W[2*d:3*d]

        blocks[f"layer{i}.WO"] = layer.out.weight.detach().cpu()

    return blocks


def flatten_model_params(model):
    return torch.cat([
        p.detach().flatten()
        for p in model.parameters()
        if p.requires_grad
    ])

# ----------------------------
# COMMUTATORS
# ----------------------------

def commutator_defect(model, batch_fn, device, eta=1e-3, eps=1e-12):
    """
    Measures a scale-normalized commutator:
        ||theta_AB - theta_BA|| / (||eta gA|| * ||eta gB||)
    """
    was_training = model.training
    model.train()

    def flat_params():
        return torch.cat([p.flatten() for p in model.parameters() if p.requires_grad])

    def write_params(theta):
        with torch.no_grad():
            offset = 0
            for p in model.parameters():
                if not p.requires_grad:
                    continue
                n = p.numel()
                p.copy_(theta[offset:offset+n].view_as(p))
                offset += n

    def batch_grad(x, y):
        model.zero_grad(set_to_none=True)
        logits, _ = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        return torch.cat([
            (p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
            for p in model.parameters() if p.requires_grad
        ])

    # sample two batches, extract positions and mA/mB
    xA, yA, positionsA = batch_fn()
    xB, yB, positionsB = batch_fn()
    mA = len(positionsA)
    mB = len(positionsB)
    xA, yA = xA.to(device), yA.to(device)
    xB, yB = xB.to(device), yB.to(device)

    theta0 = flatten_model_params(model)

    # gradients at theta0
    gA = batch_grad(xA, yA)
    gB = batch_grad(xB, yB)
    # cosine similarity at theta0
    gA_norm0 = gA.norm()
    gB_norm0 = gB.norm()
    grad_cos = (gA @ gB) / (gA_norm0 * gB_norm0 + eps)
    grad_cos = grad_cos.item()

    # AB
    write_params(theta0 - eta * gA)
    gB1 = batch_grad(xB, yB)
    thetaAB = theta0 - eta * gA - eta * gB1

    # BA
    write_params(theta0 - eta * gB)
    gA1 = batch_grad(xA, yA)
    thetaBA = theta0 - eta * gB - eta * gA1

    # restore
    write_params(theta0)
    if not was_training:
        model.eval()

    # scale-normalized commutator
    normA = (eta * gA).norm()
    normB = (eta * gB).norm()

    delta = thetaAB - thetaBA
    raw_norm = delta.norm()

    defect = (raw_norm / (normA * normB + eps)).item()
    return defect, delta.detach(), mA, mB, grad_cos, normA.detach(), normB.detach()

def commutator_defect_median(
    model, batch_fn, device, K=9, eta=1e-3, eps=1e-12
):
    Ds = []
    deltas = []
    records = []

    for _ in range(K):
        D, delta, mA, mB, gcos, _, _ = commutator_defect(
            model, batch_fn, device, eta=eta, eps=eps
        )
        Ds.append(D)
        deltas.append(delta)
        records.append((mA, mB, gcos))

    Ds_t = torch.tensor(Ds)
    D_med = Ds_t.median().item()
    D_p90 = Ds_t.quantile(0.9).item()

    return {
        "median": D_med,
        "p90": D_p90,
        "raw": Ds,
        "delta": deltas,
        "meta": records,
    }

def split_commutators(out):
    same = []
    cross = []
    gcos_all = []

    for D, (mA, mB, gcos) in zip(out["raw"], out["meta"]):
        if mA == mB:
            same.append(D)
        else:
            cross.append(D)
        gcos_all.append(gcos)

    def safe_median(xs):
        if len(xs) == 0:
            return float("nan")
        return float(torch.tensor(xs).median().item())

    return {
        "same_median": safe_median(same),
        "cross_median": safe_median(cross),
        "gcos": safe_median(gcos_all),
        "same": same,
        "cross": cross
    }

def _param_offsets(model):
    """Return start offsets for each trainable parameter in flat_params order."""
    offsets = {}
    cursor = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        offsets[id(p)] = cursor
        cursor += p.numel()
    return offsets, cursor


def _block_basis(block, topk):
    """Top-k singular directions (as flattened vectors) for a weight block."""
    U, S, Vh = torch.linalg.svd(block, full_matrices=False)
    vecs = []
    r = min(topk, S.numel())
    for i in range(r):
        # rank-1 component scaled by singular value, flattened
        comp = S[i] * (U[:, i].unsqueeze(1) @ Vh[i].unsqueeze(0))
        flat = comp.reshape(-1)
        norm = flat.norm()
        if norm > 0:
            vecs.append(flat / norm)
    return vecs

def extract_parameter_subspace(model, k=3, device="cpu"):
    """
    Build a basis from the top-k PCA-like directions of WQ/WK/WV/WO in each layer.
    Returns:
        B: [P, K] orthonormal basis aligned to flattened parameter vector.
    """
    offsets, total_params = _param_offsets(model)
    basis_vecs = []

    for layer in model.layers:
        qkv = layer.qkv.weight.detach()
        out_w = layer.out.weight.detach()
        d = qkv.shape[0] // 3
        block_slices = {
            "WQ": (qkv[:d], 0),
            "WK": (qkv[d:2*d], d * d),
            "WV": (qkv[2*d:3*d], 2 * d * d),
        }

        # qkv weight parameter shared; capture its offset once
        qkv_offset = offsets.get(id(layer.qkv.weight), None)
        out_offset = offsets.get(id(layer.out.weight), None)

        for name, (block, offset_in_param) in block_slices.items():
            for vec in _block_basis(block, k):
                if qkv_offset is None:
                    continue
                gv = torch.zeros(total_params, device=device)
                start = qkv_offset + offset_in_param
                end = start + block.numel()
                gv[start:end] = vec.to(device)
                basis_vecs.append(gv)

        # Output projection
        for vec in _block_basis(out_w, k):
            if out_offset is None:
                continue
            gv = torch.zeros(total_params, device=device)
            gv[out_offset:out_offset + out_w.numel()] = vec.to(device)
            basis_vecs.append(gv)

    if not basis_vecs:
        return None

    B = torch.stack(basis_vecs, dim=1)  # [P, K]
    # Orthonormalize on CPU to avoid missing MPS ops
    if B.device.type != "cpu":
        B_cpu = B.cpu()
    else:
        B_cpu = B
    B_ortho, _ = torch.linalg.qr(B_cpu, mode="reduced")
    return B_ortho.to(device)

def projected_commutator(delta, B, normA, normB, eps=1e-12):
    """
    delta: [P] commutator vector
    B: [P, k] learned subspace basis
    """
    delta = delta.reshape(-1)

    # If basis is missing or shapes mismatch, fall back to full norm only.
    if B is None or delta.numel() != B.shape[0]:
        full_val = (delta.norm() / (normA * normB + eps)).item()
        return {
            "proj": float("nan"),
            "resid": float("nan"),
            "full": full_val,
        }

    coeffs = B.T @ delta          # [k]
    proj = B @ coeffs             # [P]
    resid = delta - proj

    proj_norm = proj.norm()
    resid_norm = resid.norm()

    scale = normA * normB + eps

    return {
        "proj": (proj_norm / scale).item(),
        "resid": (resid_norm / scale).item(),
        "full": (delta.norm() / scale).item(),
    }


# ----------------------------
#    SAE
# ----------------------------

@torch.no_grad()
def collect_factor_activations(model, batch_fn, device, n_batches=5):
    """
    Collect EOS hidden states and per-example task factors.
    Robust under mixed-uniform m.
    Returns:    
        H : [N, D] EOS hidden states
        Y : [N] output labels
        M : [N] task complexity m (number of markers)
    """
    model.eval()
    H, Y, M = [], [], []

    for _ in range(n_batches):
        x, y, positions = batch_fn()
        x, y = x.to(device), y.to(device)

        _, _, h = model(x, return_h=True)
        h_eos = h[:, -1].detach().cpu()   # [B, D]

        H.append(h_eos)
        Y.append(y.detach().cpu())

        m_val = len(positions)
        M.append(torch.full((x.shape[0],), m_val, dtype=torch.long))

    H = torch.cat(H, 0)
    Y = torch.cat(Y, 0)
    M = torch.cat(M, 0)
    return H, Y, M


@torch.no_grad()
def collect_layer_eos(model, batch_fn, device, n_batches=30):
    """
    Returns:
      H_layers: list of [N, D] tensors, one per layer output (post-attn residual)
      Y: [N] answer labels
      M: [N] task complexity labels (m)
    """
    model.eval()
    L = len(model.layers)
    H_layers = [ [] for _ in range(L) ]
    Ys, Ms = [], []

    # hook storage
    cache = [None] * L
    hooks = []

    def make_hook(i):
        def hook(module, inp, out):
            # AttentionBlock returns (x, attn)
            x = out[0] if isinstance(out, (tuple, list)) else out
            cache[i] = x.detach()
        return hook

    for i, layer in enumerate(model.layers):
        hooks.append(layer.register_forward_hook(make_hook(i)))

    for _ in range(n_batches):
        x, y, positions = batch_fn()
        m_val = len(positions)

        x = x.to(device)
        y = y.to(device)

        # forward once, hooks fill cache
        _ = model(x, return_attn=False)

        B = x.shape[0]
        Ys.append(y.detach().cpu())
        Ms.append(torch.full((B,), m_val, dtype=torch.long))

        for i in range(L):
            h_i = cache[i]                  # [B, T, D]
            H_layers[i].append(h_i[:, -1].cpu())  # EOS: [B, D]

    # remove hooks
    for h in hooks:
        h.remove()

    H_layers = [torch.cat(xs, 0) for xs in H_layers]
    Y = torch.cat(Ys, 0)
    M = torch.cat(Ms, 0)
    return H_layers, Y, M


def fit_probe_direction(H, labels, n_classes, steps=1500, lr=1e-2):
    W = torch.zeros(H.shape[1], n_classes, requires_grad=True)
    b = torch.zeros(n_classes, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr)

    for _ in range(steps):
        logits = H @ W + b
        loss = F.cross_entropy(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        v = W[:, 0]
        v = v / (v.norm() + 1e-9)
    return v


def cosine(u, v):
    return float(torch.dot(u, v))


def probe_accuracy(H, labels, n_classes, steps=800, lr=1e-2):
    """
    Simple linear classifier on frozen H.
    Returns accuracy on the same data (fine for diagnostic purposes).
    """
    H = H.float()
    labels = labels.long()
    D = H.shape[1]

    W = torch.zeros(D, n_classes, requires_grad=True)
    b = torch.zeros(n_classes, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr)

    for _ in range(steps):
        logits = H @ W + b
        loss = F.cross_entropy(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        pred = (H @ W + b).argmax(dim=-1)
        acc = (pred == labels).float().mean().item()

    return acc

# ----------------------------
# SAE DATA + TRAINING
# ----------------------------

@torch.no_grad()
def collect_eos_activations(model, batch_fn, device, n_batches=50):
    """
    Collect final-layer EOS residual stream activations.
    """
    model.eval()
    X, M = [], []

    for _ in range(n_batches):
        x, y, positions = batch_fn()
        m_val = len(positions)

        x = x.to(device)
        _, _, h = model(x, return_h=True)

        X.append(h[:, -1].detach().cpu())
        M.append(torch.full((x.shape[0],), m_val, dtype=torch.long))

    return torch.cat(X, 0), torch.cat(M, 0)


def train_sae(sae, X, device, steps=4000, lr=3e-4, batch_size=512):
    sae = sae.to(device)
    sae.train()

    opt = torch.optim.AdamW(sae.parameters(), lr=lr)
    X = X.to(device)
    N = X.shape[0]

    logs = []

    for t in range(steps):
        idx = torch.randint(0, N, (batch_size,), device=device)
        xb = X[idx]

        x_hat, z, z_pre = sae(xb)
        recon = F.mse_loss(x_hat, xb)
        reg = 1e-6 * z_pre.pow(2).mean()
        loss = recon + reg

        opt.zero_grad()
        loss.backward()
        opt.step()

        if t % 200 == 0:
            active = (z != 0).float().sum(-1).mean().item()
            logs.append((t, recon.item(), active))
            print(f"[SAE] step {t:4d} | recon={recon.item():.6f} | active={active:.2f}")

    return sae, logs

@torch.no_grad()
def collect_token_activations(model, batch_fn, device, n_batches=40):
    """
    Collect residual-stream activations for marker-following tokens only.
    These are execution-critical positions.
    """
    model.eval()
    X = []

    for _ in range(n_batches):
        x, y, positions = batch_fn()
        x = x.to(device)

        _, _, h = model(x, return_h=True)  # h: [B, T, D]

        for b in range(x.shape[0]):
            for p in positions:
                # marker-following token (value token)
                X.append(h[b, p + 1].detach().cpu())

    return torch.stack(X, dim=0)  # [N_tokens, D]

# ----------------------------
# TRAINING
# ----------------------------

def train(model, steps, batch_fn, device, check_points=None, probe_batches=None, log_every=2000):
    check_points = check_points or {}
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.train()

    # logs: {m: [(global_step, entropy), ...]}
    entropy_logs = {} if probe_batches is not None else None
    weight_log = []   # list of (step, weight_vector)
    comm_logs = []
    probe_logs = []  # (step, cos(label, marker))
    comm_logs_sorted = []
    comm_logs_proj = []
    B_subspace = None  # learned basis for projected commutator (not used here)

    for step in range(steps):
        x, y, _ = batch_fn()
        x, y = x.to(device), y.to(device)

        logits, _ = model(x)
        loss = F.cross_entropy(logits, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if  (step >= 5000) and weight_log and step%5000==0:
            B_subspace = extract_parameter_subspace(model, k=3, device=device)

        if step % 1000 == 0:
            print(f"step {step:5d} | loss {loss.item():.4f}")

        if (probe_batches is not None) and (step % log_every == 0):
            model.eval()
            with torch.no_grad():
                for m, (x_probe, _) in probe_batches.items():
                    _, attn = model(x_probe, return_attn=True)
                    ent, _ = bubble_metrics(attn[len(attn)-1], k=attn[len(attn)-1].shape[-1])
                    entropy_logs.setdefault(m, []).append((step, ent))
            model.train()

        if step % log_every == 0:
            # ---- weight snapshot (no grad) ----
            with torch.no_grad():
                w = extract_attention_blocks(model)
                weight_log.append((step, w))

            # ---- commutator defect (needs grad) ----
            out = commutator_defect_median(model, batch_fn, device, K=9)
            split = split_commutators(out)

            for D_val, delta, meta in zip(out["raw"], out["delta"], out["meta"]):
                mA, mB, gcos = meta
                comm_logs.append((step, D_val, delta, mA, mB, gcos))

            comm_logs_sorted.append((
                step,
                split["same_median"],
                split["cross_median"],
                split["gcos"]
            ))

            D, delta, mA, mB, gcos, normA, normB = commutator_defect(model, batch_fn, device)

            if step > 5000:
                proj = projected_commutator(delta, B_subspace, normA, normB)
                comm_logs_proj.append((
                    step,
                    proj["full"],
                    proj["proj"],
                    proj["resid"],
                ))
            else:
                comm_logs_proj.append((step, float("nan"), float("nan"), float("nan")))

            # ---- probe geometry (needs grad for probe training) ----
            H, Y, M = collect_factor_activations(
                model, batch_fn, device, n_batches=3
            )

            v_y = fit_probe_direction(
                H, Y, n_classes=model.head.out_features
            )

            n_m_classes = int(M.max().item()) + 1
            n_m_classes = max(n_m_classes, 2)  # safety against degenerate batches

            v_m = fit_probe_direction(
                H, M, n_classes=n_m_classes
            )

            probe_logs.append((
                step,
                cosine(v_y, v_m),
            ))

            if step in check_points.values():
                torch.save(
                    model.state_dict(),
                    f"ckpt_step_{step}.pt"
                )

    return model, entropy_logs, weight_log, comm_logs,comm_logs_sorted, comm_logs_proj, probe_logs


# ----------------------------
# EVAL
# ----------------------------

@torch.no_grad()
def eval_with_k(model, batch_fn, k, device):
    x, y, _ = batch_fn()
    x, y = x.to(device), y.to(device)

    _, attn = model(x, return_attn=True)

    override = {
        l: apply_topk(attn[l], k)
        for l in attn
    }

    logits, _ = model(x, attn_override=override)
    acc = (logits.argmax(-1) == y).float().mean().item()

    ent, topk_mass = bubble_metrics(attn[len(attn)-1], k)
    return acc, ent, topk_mass

def eval_with_token_sae_ablation(model, sae, batch_fn, device, latent_idxs):
    model.eval()
    sae.eval()

    x, y, positions = batch_fn()
    x, y = x.to(device), y.to(device)

    logits, _, h = model(x, return_h=True)
    h_mod = h.clone()

    for b in range(x.shape[0]):
        for p in positions:
            _, z, _ = sae(h[b, p + 1])
            for idx in latent_idxs:
                z[idx] = 0.0
            h_mod[b, p + 1] = z @ sae.W_dec.t() + sae.b_dec + sae.b_pre

    logits_mod = model.head(h_mod[:, -1])

    acc0 = (logits.argmax(-1) == y).float().mean().item()
    acc1 = (logits_mod.argmax(-1) == y).float().mean().item()
    return acc0, acc1

# ----------------------------
# SAE DEMO
# ----------------------------

@torch.no_grad()
def sae_latent_correlations(sae, X, M, topn=10):
    sae.eval()
    _, _, z_pre = sae(X.to(next(sae.parameters()).device))
    z_pre = z_pre.cpu()

    m = M.float()
    m = (m - m.mean()) / (m.std() + 1e-9)

    Zc = (z_pre - z_pre.mean(0)) / (z_pre.std(0) + 1e-9)
    corr = (Zc * m[:, None]).mean(0)

    vals, idx = torch.topk(corr.abs(), topn)
    return list(zip(idx.tolist(), corr[idx].tolist()))


@torch.no_grad()
def eval_with_sae_ablation(model, sae, batch_fn, device, latent_idx):
    model.eval()
    sae.eval()

    x, y, _ = batch_fn()
    x, y = x.to(device), y.to(device)

    logits, _, h = model(x, return_h=True)
    h_eos = h[:, -1]

    _, z, _ = sae(h_eos)
    z[:, latent_idx] = 0.0
    h_mod = z @ sae.W_dec.t() + sae.b_dec + sae.b_pre

    logits_mod = model.head(h_mod)

    acc_orig = (logits.argmax(-1) == y).float().mean().item()
    acc_mod = (logits_mod.argmax(-1) == y).float().mean().item()
    return acc_orig, acc_mod

@torch.no_grad()
def top_variance_latents(sae, X, topn=12):
    """
    Return indices of SAE latents with highest variance
    over the dataset X.
    """
    device = next(sae.parameters()).device
    X = X.to(device)

    # SAE forward: we want latent pre-activations
    _, _, z_pre = sae(X)   # z_pre: [N, d_sae]

    var = z_pre.var(dim=0)   # [d_sae]
    _, idx = torch.topk(var, topn)

    return idx.tolist()



# ----------------------------
# Random State recorder
# ----------------------------

class RandomStateRecorder:
    def __init__(self):
        self.pytorch_state = None
        self.numpy_state = None
        self.python_state = None
        self.mps_state = None

    def record_current_state(self):
        self.pytorch_state = torch.get_rng_state()
        self.numpy_state = np.random.get_state()
        self.python_state = random.getstate()
        
        if torch.backends.mps.is_available():
            self.mps_state = torch.mps.get_rng_state()

    def print_recorded_state(self):
        print("Recorded Random States:")
        print(f"PyTorch RNG State: {self.pytorch_state}")
        print(f"NumPy RNG State: {self.numpy_state[1][:5]}... (truncated)")
        print(f"Python random State: {self.python_state[1][:5]}... (truncated)")
        if self.mps_state is not None:
            print(f"MPS RNG State: {self.mps_state}")
    
    def save_state(self, filename):
        state = {
            'pytorch_state': self.pytorch_state,
            'numpy_state': self.numpy_state,
            'python_state': self.python_state,
            'mps_state': self.mps_state
        }
        with open(filename, 'wb') as f:
            pickle.dump(state, f)
        print(f"Random state saved to {filename}")
    
    def load_state(self, filename):
        with open(filename, 'rb') as f:
            state = pickle.load(f)
        self.pytorch_state = state['pytorch_state']
        self.numpy_state = state['numpy_state']
        self.python_state = state['python_state']
        self.mps_state = state['mps_state']
        print(f"Random state loaded from {filename}")
        
    def restore_state(self):
        torch.set_rng_state(self.pytorch_state)
        np.random.set_state(self.numpy_state)
        random.setstate(self.python_state)
        if self.mps_state is not None and torch.backends.mps.is_available():
            torch.mps.set_rng_state(self.mps_state)
        print("Random state restored")

'''
recorder = RandomStateRecorder()
recorder.record_current_state()
recorder.save_state(f'initial_state_bubble1.pkl')
recorder.print_recorded_state()
'''
