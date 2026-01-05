# ================================
# train_mixed.py
# ================================

import math
import random
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

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
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

    def forward(self, x, attn_override=None, return_attn=False):
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
        return logits, all_attn


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


# ----------------------------
# TRAINING
# ----------------------------

def train(model, steps, batch_fn, device, probe_batches=None, log_every=2000):
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.train()

    # logs: {m: [(global_step, entropy), ...]}
    entropy_logs = {} if probe_batches is not None else None
    weight_log = []   # list of (step, weight_vector)

    for step in range(steps):
        x, y, _ = batch_fn()
        x, y = x.to(device), y.to(device)

        logits, _ = model(x)
        loss = F.cross_entropy(logits, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

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
            with torch.no_grad():
                w = extract_attention_blocks(model)
                weight_log.append((step, w))

    return model, entropy_logs, weight_log


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


# ----------------------------
# MAIN EXPERIMENT
# ----------------------------

def run():
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

        for m in cfg.m_schedule:
            print(f"\n=== Training stage: m = {m} ===")
            def batch_fn():
                return sample_m_bubble_batch(
                    cfg.batch_size, cfg.T, m, cfg.vocab, cfg.n_classes
                )

            model, entropy_logs, weight_log = train(
                model, cfg.steps_per_stage, batch_fn, cfg.device,
                probe_batches=probe_batches, log_every=500
            )
            all_weight.extend([(step + global_step_offset, w) for (step, w) in weight_log])
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
        model, entropy_logs, weight_log = train(
            model, cfg.total_steps, batch_fn, cfg.device,
            probe_batches=probe_batches, log_every=500
        )
        all_entropy = entropy_logs
        all_weight = weight_log

    else:
        raise ValueError(f"Unknown schedule: {cfg.schedule}")
    
    # ---- ENTROPY VS TRAINING STEP (per m) ----
    
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
    
    accs_all, ents_all = {}, {}
    for m in cfg.m_schedule:
        batch_fn = lambda: sample_m_bubble_batch(
            cfg.batch_size, cfg.T, m, cfg.vocab, cfg.n_classes
        )
        print(f"\n--- Eval results for m={m} ---")
        accs, ents = [], []
        for k in cfg.ks:
            acc, ent, _ = eval_with_k(model, batch_fn, k, cfg.device)
            accs.append(acc)
            ents.append(ent)
            print(f"m ={m}, k={k:2d} | acc={acc:.4f} | entropy={ent:.4f}")
        accs_all[m] = accs
        ents_all[m] = ents
        # ---- PLOT ACCURACY & ENTROPY VS k ----#
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
        plt.suptitle(f"Eval results for m={m}")
        plt.tight_layout()
        plt.show()
        '''
        m_last = cfg.m_schedule[-1]
    return model, all_entropy, all_weight, {
        "ks": cfg.ks,
        "accs": accs_all[m_last],
        "ents": ents_all[m_last]
    }   


if __name__ == "__main__":

    all_runs = []
    seeds = range(2)

    for seed in seeds:
        torch.manual_seed(seed)
        model, entropy, weight_log, metrics = run()
        
        all_runs.append({
            "seed": seed,
            "entropy": entropy,
            "weights": weight_log,
            "metrics": metrics
        })

    #torch.save(all_runs, "runs.pt")