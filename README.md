# **bubble-modadd** 

# *Learning Geometry and Training Dynamics*
> 
> ### Main Results (TL;DR)
>
> - In controlled modular-addition tasks, transformer training dynamics collapse onto a low-dimensional execution manifold (≈2–3D).  
> - SGD commutators are preferentially aligned with the execution subspace (up to 10× random baseline) early in training, with >92% of non-commutativity confined to orthogonal staging directions.
> - A random-subspace baseline control confirms the geometric alignment is structured, not a dimensionality artifact (exec/random ratio 2–10×).  
> - Attention “bubbling,” circuit formation, and robustness emerge as geometric consequences of this collapse.  
> - Sparse autoencoders capture auxiliary routing structure but do not isolate execution itself.

### Summary

This repository explores the **training dynamics** of transformer models through a carefully controlled, **marker-based modular addition** task.

The primary focus is not on achieving state-of-the-art performance, but on understanding the **geometric structure** of how learning actually happens in overparameterized networks.

We investigate:

- How overparameterized transformers discover and encode **structured computation**
- The emergence of interpretable **attention patterns** and modular **circuits**
- The surprising behavior of **stochastic optimization** (SGD) along — and especially *across* — the learned family of solutions

**Core discovery**  
During training, the high-dimensional parameter trajectories rapidly **collapse** onto a low-dimensional **execution subspace** (≈2–3 dimensions in this task). Nearly all meaningful progress in learning the task occurs inside this tiny subspace, while stochasticity, task interference, and most SGD noise are absorbed in the vast orthogonal complement.

This geometric collapse appears to underlie several intriguing phenomena including attention "bubbling", robust circuit formation, approximate integrability of SGD along execution directions, and the limitations of current sparse autoencoder-based interpretability approaches.

# 🔍 Motivation

Transformer models often exhibit:
- sharp attention concentration (“bubbles”),
- grokking-like generalization transitions,
- interpretable circuit structure.
- surprising robustness despite noisy SGD dynamics.

These phenomena are frequently studied in isolation.

This work proposes a unifying geometric explanation:**

**-Training dynamics rapidly collapse onto a low-dimensional execution manifold.**

**-Many of the most intriguing empirical observations — attention bubbling, abrupt grokking-style generalization, circuit formation, and robustness to noise — are natural consequences (i.e. projections) of this severe dimensional collapse.**

From this perspective, the central object of study is no longer the final trained network viewed as a static function, but rather **the trajectory of learning itself** — the geometry and dynamics of the path through parameter space.

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
- **Attention-only** models (no MLP layers)
- **Standard** transformer blocks (attention + MLP)

This allows direct comparison of how architectural expressivity affects training geometry.

# 📊 Key Findings

1. **Collapse onto a Low-Dimensional Execution Manifold**  
   In attention-only transformers, the attention parameters  (`W_Q`, `W_K`, `W_V`, `W_O`) rapidly collapse during training onto a **2–3 dimensional subspace**.  

   This collapse is:  
   - consistent across layers  
   - the intrinsic dimension is stable across random seeds  
   - robust for moderate task difficulty (e.g. `m ≤ 6`)
   - While the orientation of this subspace in raw parameter coordinates is seed-dependent, its dimension and dynamical role are invariant across runs  

   The phenomenon reflects a deep constraint on **learning dynamics**, not the result of explicit regularization or pruning.
<!-- Optional: centered version with caption -->
<p align="center">
  <img src="plots/figure1_intrinsic_dim.png" width="70%" alt="PCA of attention weights showing collapse to low dimension">
  <br>
</p>

2. **Attention Bubbling as Geometric Saturation**  
   Sharp attention concentration (“**attention bubbles**”) emerges naturally as **saturation** along a routing coordinate within the reduced execution manifold.  

   Bubbling is **not** a discrete architectural quirk, but rather the continuous projection of movement along a low-dimensional learning trajectory.
<!-- Optional: centered version with caption -->
<p align="center">
  <img src="plots/figure2_entropy_bubbling.png" width="70%" alt="PCA of attention weights showing collapse to low dimension">
  <br>
</p>
   

3. **Non-Integrability of SGD and Commutator Analysis**
   Despite the dramatic dimensional collapse, SGD updates remain **strongly non-commutative** in the full high-dimensional parameter space.

   We quantify this via **SGD commutators**
   `θ_AB - θ_BA` (computed from sequential gradient steps on independent minibatches).

   Key observations:
   - Normalized commutator defect `D = ||δ|| / (||ηg_A|| · ||ηg_B||)` **grows throughout training**, even after loss convergence
   - Defect spikes reaching 100–175× step magnitude indicate persistent non-commutativity

   **Crucially**: The execution subspace (built from **PCA of the weight trajectory**) captures a small but **geometrically structured** fraction of commutator energy:
   - Projection fraction ρ_exec ≈ 0.02–0.13, with **>92% of non-commutativity perpendicular** to the execution manifold
   - A **random-subspace baseline** confirms this is not a dimensionality artifact: exec/random ratio reaches **9.7× early** in training and **2.1× late**
   - The decreasing ratio indicates non-commutativity progressively **rotates out** of the execution manifold as the model converges
   <!-- Optional: centered version with caption -->
 <p align="center">
   <img src="plots/figure3_commutator_raw.png" width="70%" alt="PCA of attention weights showing collapse to low dimension">
   <br>
 </p>

4. **Localization of Noncommutativity with Random Baseline Control**
   Decomposing commutators into components **within** and **orthogonal** to the execution subspace, and comparing against random K-dimensional baselines:

   - **>92%** of commutator energy is perpendicular to execution directions throughout training, rising to **>98%** late
   - The execution basis captures **2–10× more** commutator energy than a random subspace of equal dimension
   - This ratio **decreases over training** (9.7× early → 2.1× late), indicating residual non-commutativity is progressively expelled from execution directions

   This suggests a fundamental geometric role for overparameterization:
   extra dimensions **absorb optimization interference** without disrupting the core execution computation.
   <p align="center">
   <img src="plots/figure4_comm_ratio.png" width="70%" alt="PCA of attention weights showing collapse to low dimension">
   <br>
  </p>
    <p align="center">
    <img src="plots/figure5_comm_decomposition.png" width="70%" alt="PCA of attention weights showing collapse to low dimension">
    <br>
   </p>

5. **Sparse Autoencoders as Supporting Evidence**  
   We train sparse autoencoders (SAEs) on intermediate activations to probe internal representations.  

   Results:  
   - A small number of SAE latents correlate with task structure (e.g. marker count `m`)  
   - Ablation of these latents causes small, **stage-dependent** accuracy drops  
   - Largest sensitivity occurs **mid-training** and diminishes late  

   **Most importantly**:  
   - SAE latents **do not isolate execution itself**  
   - Execution remains **distributed** across the low-dimensional manifold  

   → SAEs highlight a clean separation between **execution geometry** and **auxiliary routing / staging structure**.
   <p align="center">
       <img src="plots/figure6_sae_ablation.png" width="60%" alt="PCA of attention weights showing collapse to low dimension">
   <br>
   </p>

7. **Architectural Sensitivity**  
   Adding MLP layers **disrupts** the low-dimensional geometric picture:  
   - Attention parameters no longer collapse to low dimension  
   - Training dynamics remain **higher-dimensional**  
   - Generalization degrades under comparable training budgets  

   This indicates a tight coupling between **attention-based routing** and low-dimensional execution dynamics.

8. **Path Dependence and Forgetting**  
   Under strict curriculum learning (progressively increasing `m`):  
   - Performance on **smaller m** degrades as training advances to larger `m`  
   - Despite the model having **sufficient capacity**  

   This behavior is consistent with **path-dependent** movement along a low-dimensional solution manifold, rather than independent memorization of each task variant.
### Generalization to Modular Multiplication

Preliminary experiments confirm the framework extends beyond addition:

**Key observations:**
- Modular multiplication converges to slightly higher dimensional execution 
  manifolds (~4-6D vs 2-3D for addition)
- Dimensionality scales predictably with computational complexity
- All attention matrices (W_Q, W_K, W_V, W_O) operate in low-dimensional 
  subspaces (4-7D out of d=128)
- Same geometric principles apply: commutator localization to orthogonal
  directions, preferential alignment with execution subspaces relative to
  random baselines, and training curriculum effects

This suggests execution manifold dimensionality may be predictable from 
task structure, with implications for understanding how complexity scales 
in neural networks.

## Core Geometric Principle: The Execution Manifold as an Integrable Subspace

Despite operating in high-dimensional parameter space (d=128), transformers 
converge to a 2-3D "execution manifold" where:

1. **Dimensional Collapse**: Attention parameters collapse onto a 
   low-dimensional subspace with consistent intrinsic dimensionality (2–3) but no canonical alignment in parameter space, for moderate task 
   difficulties (m ≤ 6)

2. **Structured Projection with Random Baseline Control**: When SGD
   commutators [θ_AB - θ_BA] are projected onto the execution manifold
   (built from PCA of the weight trajectory), the execution basis captures
   2–10× more energy than a random subspace of equal dimension

   → >92% of non-commutativity is perpendicular to execution directions
   → The exec/random ratio decreases over training (9.7× → 2.1×) as
     non-commutativity rotates out of the execution manifold

3. **Geometric Role of Overparameterization**: The vast orthogonal space 
   absorbs optimization interference without disrupting core computation

## Theoretical Interpretation

This empirical picture suggests a novel view of neural network learning:

**Overparameterization enables parallel micro-task learning**: High 
dimensionality allows the model to explore many local solutions 
simultaneously in "staging" directions without interference.

**Grokking as projection onto the execution manifold**: The phase 
transition corresponds to discovering the low-dimensional integrable 
subspace where compositional operations commute.

**Mixed vs Curriculum learning**: Mixed training forces discovery of the 
unified execution manifold. Curriculum learning gets trapped in 
task-specific regions of the staging space, leading to catastrophic 
forgetting.

This framework connects to dynamical systems theory: the execution manifold 
is an attractor in the learning dynamics, analogous to stable solutions in 
nonlinear PDEs.

## Implications for Interpretability and Safety

1. **When to interpret**: Sparse features (SAE-recoverable) matter early; 
   geometric structure matters late
   
2. **Where to interpret**: Focus on the execution manifold, not the full 
   parameter space
   
3. **How to train**: Mixed-task training may be crucial for discovering 
   compositional structure
   
4. **Scaling hypothesis**: If this pattern holds for complex tasks, 
   powerful models may also use low-dimensional execution manifolds - 
   offering hope for interpretability at scale

## Training Curriculum Determines Solution Geometry

**Mixed learning** (all m values simultaneously):
- Converges to unified 2-3D execution manifold
- No forgetting
- Enables compositional generalization

**Curriculum learning** (progressive m=1→2→3→4):
- Catastrophic forgetting of earlier m values
- Fails to discover integrable structure

**Critical observation**: Model cannot learn m=4 from scratch, but 
curriculum leads to forgetting. Mixed learning resolves this by forcing 
discovery of shared compositional structure.


# 📐 Analysis Tools

This repository provides a suite of diagnostic tools to analyze training dynamics and representations:

- **Intrinsic dimensionality** via effective rank estimation
- **PCA subspace alignment** across layers, seeds, and tasks
- **Attention bubbling metrics** (concentration and saturation)
- **Attention entropy** and other distribution statistics
- **SGD commutator** computation & **projected-commutator** analysis — to separate integrable vs. non-integrable directions
- **Sparse autoencoder** training, probing, and targeted **ablations** — for dissecting routing vs. execution structure
<!--
- **Distance to final low-dimensional subspace** during training 
-->

All metrics are **basis-invariant** and computed independently per random seed for robust averaging.

# 🚀 Running Experiments

Dependencies are standard PyTorch / NumPy / matplotlib.

Typical Usage
```bash
python train.py
python experiment_full.py
```

# 📈 Analysis

Analysis is run offline on saved checkpoints:
```bash
python analysis_subspace.py
python analysis_alignment.py
```

# 🧭 Open Questions

This work opens several promising directions for future investigation:

- **Scaling of execution dimensionality** How does the intrinsic dimension of the learned manifold scale with architectural expressivity (e.g., deeper/wider models, larger embedding size)? How do these collapse and bubbling phenomena behave as model depth and width increase—does the effective dimension remain bounded, or grow slowly?
- **Predictive diagnostics early in training** Can early-training low-dimensional diagnostics (e.g., subspace distance or effective rank) reliably **predict** eventual grokking success?
- **Robustness beyond linear tasks** In nonlinear tasks, does learning still concentrate dynamics onto a low-dimensional execution manifold (locally or globally), and how should intrinsic dimension be measured in the absence of a global linear subspace?
- **Micro-task composition into stable circuits** How do multiple micro-tasks discovered during training interact and compose over time? Under what conditions do they merge into stable, reusable execution circuits versus remaining entangled or interfering?

# 📜 Notes

- This repository documents **active, exploratory research**—expect evolving code and experiments.
- Code is intentionally written for **clarity and ease of inspection** rather than heavy abstraction or modularity.
- Some duplication and hard-coded parameters are deliberate choices to enable **rapid prototyping and iteration**.

Feedback, suggestions, and collaborations are welcome!

