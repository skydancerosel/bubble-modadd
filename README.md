# Low-Dimensional Execution Manifolds in Transformer Learning Dynamics

We show that transformer training dynamics on structured tasks collapse onto low-dimensional **execution manifolds** in weight space. In controlled modular-addition experiments, attention parameters converge to a 2--3 dimensional subspace, with >92% of SGD non-commutativity confined to orthogonal directions. A random-subspace baseline confirms the geometric alignment is structured (exec/random ratio 2--10x), not a dimensionality artifact. Attention bubbling, circuit formation, and robustness emerge as geometric consequences of this collapse.

**Paper**: [Low-Dimensional Execution Manifolds in Transformer Learning Dynamics](https://arxiv.org/abs/2602.10496)

---

## Key Findings

1. **Collapse onto a Low-Dimensional Execution Manifold**
   In attention-only transformers, the attention parameters (`W_Q`, `W_K`, `W_V`, `W_O`) rapidly collapse during training onto a **2--3 dimensional subspace**.

   This collapse is:
   - consistent across layers
   - stable across random seeds in intrinsic dimension
   - robust for moderate task difficulty (e.g. `m <= 6`)
   - seed-dependent in orientation but invariant in dimension and dynamical role

   The phenomenon reflects a deep constraint on **learning dynamics**, not the result of explicit regularization or pruning.

<p align="center">
  <img src="plots/figure1_intrinsic_dim.png" width="70%" alt="PCA of attention weights showing collapse to low dimension">
</p>

2. **Attention Bubbling as Geometric Saturation**
   Sharp attention concentration ("**attention bubbles**") emerges naturally as **saturation** along a routing coordinate within the reduced execution manifold.

   Bubbling is **not** a discrete architectural quirk, but rather the continuous projection of movement along a low-dimensional learning trajectory.

<p align="center">
  <img src="plots/figure2_entropy_bubbling.png" width="70%" alt="Attention entropy and bubbling">
</p>

3. **Non-Integrability of SGD and Commutator Analysis**
   Despite the dramatic dimensional collapse, SGD updates remain **strongly non-commutative** in the full high-dimensional parameter space.

   We quantify this via **SGD commutators** `theta_AB - theta_BA` (computed from sequential gradient steps on independent minibatches).

   Key observations:
   - Normalized commutator defect `D = ||delta|| / (||eta*g_A|| * ||eta*g_B||)` **grows throughout training**, even after loss convergence
   - Defect spikes reaching 100--175x step magnitude indicate persistent non-commutativity

   **Crucially**: The execution subspace captures a small but **geometrically structured** fraction of commutator energy:
   - Projection fraction rho_exec ~ 0.02--0.13, with **>92% of non-commutativity perpendicular** to the execution manifold
   - A **random-subspace baseline** confirms this is not a dimensionality artifact: exec/random ratio reaches **9.7x early** in training and **2.1x late**
   - The decreasing ratio indicates non-commutativity progressively **rotates out** of the execution manifold as the model converges

<p align="center">
  <img src="plots/figure3_commutator_raw.png" width="70%" alt="Commutator defect over training">
</p>

4. **Localization of Noncommutativity with Random Baseline Control**
   Decomposing commutators into components **within** and **orthogonal** to the execution subspace, and comparing against random K-dimensional baselines:

   - **>92%** of commutator energy is perpendicular to execution directions throughout training, rising to **>98%** late
   - The execution basis captures **2--10x more** commutator energy than a random subspace of equal dimension
   - This ratio **decreases over training** (9.7x early -> 2.1x late), indicating residual non-commutativity is progressively expelled from execution directions

   This suggests a fundamental geometric role for overparameterization: extra dimensions **absorb optimization interference** without disrupting the core execution computation.

<p align="center">
  <img src="plots/figure4_comm_ratio.png" width="70%" alt="Commutator ratio">
</p>
<p align="center">
  <img src="plots/figure5_comm_decomposition.png" width="70%" alt="Commutator decomposition">
</p>

5. **Sparse Autoencoders as Supporting Evidence**
   We train sparse autoencoders (SAEs) on intermediate activations to probe internal representations.

   - A small number of SAE latents correlate with task structure (e.g. marker count `m`)
   - Ablation of these latents causes small, **stage-dependent** accuracy drops
   - SAE latents **do not isolate execution itself** --- execution remains **distributed** across the low-dimensional manifold

   SAEs highlight a clean separation between **execution geometry** and **auxiliary routing / staging structure**.

<p align="center">
  <img src="plots/figure6_sae_ablation.png" width="60%" alt="SAE ablation">
</p>

6. **Architectural Sensitivity**
   Adding MLP layers **disrupts** the low-dimensional geometric picture:
   - Attention parameters no longer collapse to low dimension
   - Training dynamics remain **higher-dimensional**
   - Generalization degrades under comparable training budgets

   This indicates a tight coupling between **attention-based routing** and low-dimensional execution dynamics.

7. **Path Dependence and Forgetting**
   Under strict curriculum learning (progressively increasing `m`):
   - Performance on **smaller m** degrades as training advances to larger `m`
   - Despite the model having **sufficient capacity**

   This behavior is consistent with **path-dependent** movement along a low-dimensional solution manifold, rather than independent memorization of each task variant.

### Generalization to Modular Multiplication

Preliminary experiments confirm the framework extends beyond addition:
- Modular multiplication converges to slightly higher dimensional execution manifolds (~4--6D vs 2--3D for addition)
- Dimensionality scales predictably with computational complexity
- Same geometric principles apply: commutator localization, preferential alignment, and curriculum effects

---

## Task: Marker-Based Modular Addition

- Sequences of length T = 32
- m marked positions (non-adjacent), indicated by marker tokens
- Each marker is followed by a value token
- The label is the sum of the marked values modulo C
- Remaining tokens are i.i.d. distractors

Task difficulty is controlled via number of markers m, number of classes C, and curriculum vs mixed-data sampling.

## Models

We study transformer variants with:
- **Attention-only** models (no MLP layers)
- **Standard** transformer blocks (attention + MLP)

This allows direct comparison of how architectural expressivity affects training geometry.

## Running Experiments

Dependencies are standard PyTorch / NumPy / matplotlib.

```bash
python train.py
python experiment_full.py
```

Analysis is run offline on saved checkpoints:
```bash
python analysis_subspace.py
python analysis_alignment.py
```

## Citation

```bibtex
@article{xu2026execution,
  title={Low-Dimensional Execution Manifolds in Transformer Learning Dynamics},
  author={Xu, Yongzhong},
  year={2026},
  eprint={2602.10496},
  archivePrefix={arXiv},
  url={https://arxiv.org/abs/2602.10496}
}
```
