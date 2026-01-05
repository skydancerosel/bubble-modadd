# bubble-modadd
This repository contains code for studying training dynamics, attention bubbling, and intrinsic dimensionality in transformer models using controlled modular-addition tasks with explicit markers.
The project focuses on mechanistic understanding rather than benchmark performance.

# 🔍 Motivation

Transformer models often exhibit:
- sharp attention concentration (“bubbles”),
- grokking-like generalization transitions,
- interpretable circuit structure.

These phenomena are frequently studied in isolation.

This project explores a unifying hypothesis:
- Training dynamics collapse onto a low-dimensional manifold,
- and bubbles and circuits emerge as projections of this reduced dynamics.

# 🧪 Task: Marker-Based Modular Addition
- Sequences of length T = 32
- m marked positions (non-adjacent), indicated by marker tokens
- Each marker is followed by a value token
- The label is the sum of the marked values modulo C
- Remaining tokens are i.i.d. distractors

Task difficulty is controlled via:
- number of markers m
- number of classes C
- curriculum vs mixed-data sampling

# 🧠 Models

We study transformer variants with:
	- attention-only blocks (no MLP)
	-	standard transformer blocks (with MLP)

This allows direct comparison of how architectural expressivity affects training geometry.

# 📊 Key Findings

- Attention-only transformers exhibit robust collapse of attention weight matrices (WQ, WK, WV, WO) onto a **2–3 dimensional submanifold**, consistent across:
  - layers
  - random seeds
  - task parameters (m = 3–6 markers, varying number of classes)
- Attention "bubbling" (sharp concentration) emerges naturally as saturation along a routing coordinate in this reduced manifold.
- For higher difficulty tasks (e.g., m ≥ 4), standard training fails to generalize, while curriculum or mixed-data sampling enables convergence—suggesting an **optimization barrier** rather than a capacity limit.
- Adding MLPs disrupts the low-dimensional collapse:
  - Attention parameters remain in higher-dimensional space
  - Generalization performance degrades under the same training budget
  - This indicates tight coupling between attention-based routing and computation pathways
- Strict curriculum learning reliably achieves grokking; an open question is whether it converges to the **same** low-dimensional submanifold as mixed-data training.

# 📐 Analysis Tools

This repository provides a suite of diagnostic tools to analyze training dynamics and representations:

- **Intrinsic dimensionality** via effective rank estimation
- **PCA subspace alignment** across layers, seeds, and tasks
- **Attention bubbling metrics** (concentration and saturation)
- **Attention entropy** and other distribution statistics
<!--
- **Distance to final low-dimensional subspace** during training 
-->

All metrics are **basis-invariant** and computed independently per random seed for robust averaging.

# 🚀 Running Experiments

Dependencies are standard PyTorch / NumPy / matplotlib.

Typical Usage
```bash
python groking_v3.py
```

# 📈 Analysis

Analysis is run offline on saved checkpoints:
```bash
python analysis_subspace.py
python analysis_alignment.py
```

# 🧭 Open Questions

This work opens several promising directions for future investigation:

- How does the intrinsic dimension of the learned manifold scale with architectural expressivity (e.g., deeper/wider models, larger embedding size)?
- Do curriculum-trained and mixed-data-trained models converge to the **same** low-dimensional attractor, or to distinct but equally effective submanifolds?
- Can early-training low-dimensional diagnostics (e.g., subspace distance or effective rank) reliably **predict** eventual grokking success?
- How do these collapse and bubbling phenomena behave as model depth and width increase—does the effective dimension remain bounded, or grow slowly?

# 📜 Notes

- This repository documents **active, exploratory research**—expect evolving code and experiments.
- Code is intentionally written for **clarity and ease of inspection** rather than heavy abstraction or modularity.
- Some duplication and hard-coded parameters are deliberate choices to enable **rapid prototyping and iteration**.

Feedback, suggestions, and collaborations are welcome!

