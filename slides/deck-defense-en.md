---
theme: apple-basic
title: Shrinkage Estimators for Cosmological Precision Matrices
info: |
  Andrés F. Movilla Obregón
  Universidad del Norte, April 24, 2026
layout: center
class: intro-slide text-center
highlighter: shiki
drawings:
  persist: false
mdc: true
transition: slide-left
fonts:
  sans: Inter
  serif: Source Serif Pro
  mono: Fira Code
---

# Implementation of Shrinkage Estimators for Cosmological Precision Matrices

<div class="pt-6 opacity-80">
Andrés F. Movilla Obregón<br/>
Tutor: Elias D. Niño-Ruiz, Ph.D.<br/>
Universidad del Norte · April 24, 2026
</div>

---

# Objectives

<div class="text-sm">

**General**

Evaluate and extend direct precision shrinkage estimators in the $n \sim N_e$ regime shared by the sequential EnKF and cosmological covariance analysis.

**Specific**

1. **Implement** four direct-precision targets (identity, eigenvalue, cosmological, and scaled identity) within the TEDA framework.
2. **Benchmark** against Ledoit–Wolf, NERCOME, and the standard EnKF and EnKF-Cholesky baselines on Lorenz-96 and BOSS cosmological mocks.
3. **Characterize** how target structure (diagonal vs. dense) affects the shrinkage coefficients, clamping, and the bias–variance tradeoff.
4. **Propose** a domain-agnostic scaled-identity target as a middle ground between the plain identity and the domain-specific cosmological target.

</div>

---

# How the EnKF Works

At each cycle $t$:

1. **Simulate** $N_e$ parallel forecasts (the ensemble) by propagating the model forward
2. **Estimate** a covariance matrix ($\mathbf{B}_t$) from those forecasts
3. **Invert** it to get a precision matrix ($\mathbf{B}_t^{-1}$)
4. **Use** the precision matrix to blend the forecast with real observations

The precision matrix is **re-estimated and re-used every cycle**.

---

# The Cosmological Parallel

For one analysis:

1. **Simulate** $N_{\text{mocks}}$ mock catalogs (synthetic realizations of the universe)
2. **Estimate** a covariance matrix ($\boldsymbol{\Sigma}$) from those mocks
3. **Invert** it to get a precision matrix ($\boldsymbol{\Sigma}^{-1}$)
4. **Use** the precision matrix in a Gaussian likelihood to fit parameters against the real measurement

The precision matrix is **estimated once and reused as fixed**.

---

# Same Matrix, Different Names

Step 2 in both workflows computes the same kind of matrix.

<div class="text-sm pt-2">

|                             | Data Assimilation               | Cosmology                        |
| --------------------------- | ------------------------------- | -------------------------------- |
| Samples                     | $N_e$ ensemble forecasts        | $N_{\text{mocks}}$ mock catalogs |
| Covariance matrix of …      | forecast error                  | power-spectrum measurement       |
| Name of the matrix          | $\mathbf{B}_t$                  | $\boldsymbol{\Sigma}$            |

</div>

**Same statistical object:** a sample covariance matrix estimated from $N$ samples of an $n$-dimensional vector.

From here on, call it $\mathbf{S}$: the _generic sample covariance matrix_.

---

# The Generic Sample Covariance Matrix

For $N$ samples $\mathbf{x}_1, \dots, \mathbf{x}_N$:

$$\mathbf{S} = \frac{1}{N - 1}\sum_{i=1}^{N}(\mathbf{x}_i - \bar{\mathbf{x}})(\mathbf{x}_i - \bar{\mathbf{x}})^T$$

**Correct on average**, but for any single estimate:

- **Very noisy when $N \sim n$**: noise scales as $n^2/N$
- **Not invertible when $N \leq n$**: the matrix is singular
- **Inversion blows up the noise**: small eigenvalues in $\mathbf{S}$ become huge in $\mathbf{S}^{-1}$

So in the small-$N$, large-$n$ regime, $\mathbf{S}$ is unusable. We need a better estimator.

---

# Bias vs. Variance

The total error of any estimator splits into two pieces:

$$\text{Total error} \ = \ \text{Bias}^2 \ + \ \text{Variance}$$

- **Bias**: systematic offset from the truth
- **Variance**: random jitter between different sample sets

$\mathbf{S}$ has **zero bias** but **huge variance**.

**The idea:** accept a small bias in exchange for much less variance.

---

# Shrinkage

Blend $\mathbf{S}$ with a structured target matrix $\mathbf{T}$:

$$\hat{\mathbf{S}} \ = \ (1 - \lambda)\,\mathbf{S} \ + \ \lambda\,\mathbf{T}$$

- $\mathbf{T}$: something we trust a priori (e.g., identity, or a domain-informed matrix)
- $\lambda \in [0, 1]$: how much to trust $\mathbf{T}$ vs. $\mathbf{S}$
  - $\lambda = 0$: pure $\mathbf{S}$
  - $\lambda = 1$: pure $\mathbf{T}$

The "better estimators" in the field all differ in **how they choose $\mathbf{T}$ and $\lambda$**.

---

# State of the Art

<div class="text-sm">

| Family                 | Representative                        | What it does                                   |
| ---------------------- | ------------------------------------- | ---------------------------------------------- |
| Linear covariance      | Ledoit–Wolf (2004)                    | Blends $\mathbf{S}$ with a scaled identity     |
| Target-free            | NERCOME (2016)                        | Averages many bootstrap eigenvalue estimates   |
| Nonlinear              | Ledoit–Wolf (2012, 2020)              | Uses a different $\lambda$ for each eigenvalue |
| Sparse precision       | GLasso, CLIME                         | Forces most entries of $\mathbf{S}^{-1}$ to 0  |
| **Direct precision**   | **Bodnar et al. (2016)**              | Blends $\mathbf{S}^{-1}$ directly (our basis)  |
| Cosmological target    | Pope–Szapudi (2008), Looijmans (2024) | Uses a domain-specific target matrix           |
| Localization           | Gaspari–Cohn; modified Cholesky       | Zeroes out long-range correlations in $\mathbf{S}$ |

</div>

### But each has a dominant weakness

---

# Critical Comparison

Four weaknesses across the seven families:

1. **Wrong object.** Most methods shrink $\mathbf{S}$, but we need $\mathbf{S}^{-1}$. Inverting a shrunk $\mathbf{S}$ amplifies small-eigenvalue errors.

2. **Too expensive.** NERCOME bootstraps ~200× per call. Fine once, prohibitive inside a 100-cycle EnKF loop.

3. **Wrong shape.** Diagonal targets fail when the true precision is dense (e.g., Lorenz-96); coefficients explode and clamp every cycle.

4. **Wrong metric.** Methods minimize one-shot error. But the EnKF reuses the estimator every cycle, so errors compound.

---

# The Research Gap

No existing method delivers **all four** at once:

1. Shrinks the **precision** matrix (not just the covariance matrix)
2. **Cheap enough** to run every cycle inside an EnKF
3. **Stable** across sequential cycles (low variance)
4. **Flexible target**: works without domain knowledge


---

# Proposed Method

Start from Bodnar et al. (2016). Instead of shrinking $\mathbf{S}$ and then inverting, **shrink $\mathbf{S}^{-1}$ directly** against a target precision matrix $\boldsymbol{\Pi}_0$:

$$\boldsymbol{\Pi}_{\text{LS}} \ = \ \alpha^*\,\mathbf{S}^{-1} \ + \ \beta^*\,\boldsymbol{\Pi}_0$$

- $\alpha^*$, $\beta^*$: analytical weights, derived so the expected error of $\boldsymbol{\Pi}_{\text{LS}}$ is minimized (closed-form, recomputed from the data each time)
- $\boldsymbol{\Pi}_0$: a target precision matrix; our choice

This **fixes weakness #1** from the previous slide: we are now shrinking the object we actually need.

---

# Contribution 1: Sequential Embedding

**Embed direct precision shrinkage inside the EnKF loop.** At each cycle $t$:

1. Take the current ensemble
2. Compute $\mathbf{S}$ and recompute $\alpha^*, \beta^*$ **from scratch**
3. Form $\boldsymbol{\Pi}_{\text{LS}}$ and feed it into the Kalman gain

**No existing paper does this for direct precision shrinkage.** \
Bodnar et al. (2016) and Looijmans et al. (2024) apply it only to one-shot cosmological estimation.

---

# Four Target Choices Evaluated

| Target              | $\boldsymbol{\Pi}_0$                                  | Notes                                  |
| ------------------- | ----------------------------------------------------- | -------------------------------------- |
| Identity            | $\mathbf{I}$                                          | generic default                        |
| Eigenvalue          | uses eigenvalues of $\mathbf{S}^{-1}$                 | adapts to the data's spectrum          |
| Cosmological        | $\text{diag}(N_\ell / 2C_\ell^2)$                     | plugs in domain knowledge              |
| **Scaled identity** | $\text{diag}(2 / \mathbf{S}_{ii})$                    | **new in this thesis**                 |

One short slide per target follows.

---

# Target: Identity

$$\boldsymbol{\Pi}_0 = \mathbf{I}$$

- The **generic default**. No assumption about the data
- Treats every variable as if on the same scale
- **Fails** when variables have very different magnitudes (e.g., power spectrum at low vs. high $\ell$)

<div class="text-xs opacity-60 pt-3">Baseline target in Bodnar et al. (2016); analogue of the scaled identity in Ledoit–Wolf (2004).</div>

---

# Target: Eigenvalue

$$\boldsymbol{\Pi}_0 = \mathbf{U}\,\text{diag}(\lambda_i)\,\mathbf{U}^T$$

where $\mathbf{U}$, $\lambda_i$ are the eigenvectors and eigenvalues of $\mathbf{S}^{-1}$.

- **Adapts to the data's own spectrum**. No external parameters
- Keeps the diagonal part of $\mathbf{S}^{-1}$ in its own eigenbasis
- Self-referential: uses $\mathbf{S}^{-1}$ to build its own target

<div class="text-xs opacity-60 pt-3">Introduced by Bodnar et al. (2016) as an alternative target for direct precision shrinkage.</div>

---

# Target: Cosmological

$$\boldsymbol{\Pi}_0 = \text{diag}\!\left(\frac{N_\ell}{2\,C_\ell^2}\right)$$

- $C_\ell$: theoretical power spectrum at multipole $\ell$, from cosmological perturbation theory and reference parameters
- $N_\ell \approx (2\ell+1)\,f_{\text{sky}}$: mode count from survey geometry
- Assumes the underlying density field is **Gaussian**

**Requires a fixed cosmology, a specific survey, and a Gaussian approximation.** \
Doesn't transfer to other domains.

<div class="text-xs opacity-60 pt-3">Pope & Szapudi (2008) first used it for covariance shrinkage in cosmology; Looijmans et al. (2024) applied it with Bodnar's direct precision framework.</div>

---

# A Gap in the Target Menu

Each existing target has a weakness:

- **Identity**: ignores the scale of each variable
- **Eigenvalue**: **self-referential** and **operationally heavy** (uses $\mathbf{S}^{-1}$ to build its own target; doubles per-cycle cost)
- **Cosmological**: **requires domain knowledge** (a specific cosmology and survey)

Missing: a target that is **scale-aware**, **simple**, and **domain-agnostic**.\
Built from a cheap data summary, not from the full noisy inverse.

---

# Contribution 2: Scaled Identity Target

$$(\boldsymbol{\Pi}_0)_{ii} \ = \ \frac{2}{\mathbf{S}_{ii}}$$

- **Scale-aware**: per-variable variances from the diagonal of $\mathbf{S}$ itself
- **Domain-agnostic**: no external parameters, no cosmology, no survey
- **Simple**: only reads the diagonal of $\mathbf{S}$; $\mathcal{O}(n)$ per cycle
- Factor of 2 mirrors the cosmological $2/N_\ell$ scaling

**Fills the gap:** scale-aware + domain-agnostic + simple.

---

# Why It Works in Practice

Consequences of the scale-aware + domain-agnostic + simple design:

- **Magnitude-matched target**: diagonal $2/\mathbf{S}_{ii}$ is already in the right order of magnitude as $\mathbf{S}^{-1}$, so $\alpha^*$ and $\beta^*$ stay in $[0, 1]$ far more often than with plain identity → **lower clamping rate**.
- **Data-driven but cheap**: unlike the eigenvalue target (doubles per-cycle cost) or cosmological (needs pre-analysis setup), it's a one-line diagonal read.

Result: **10–30% improvement** over plain identity on both testbeds (RMSE in DA, Frobenius loss in cosmology).

---

# Testbed 1: Lorenz-96

A chaotic toy model of atmospheric dynamics. State dimension $n = 40$.

- Forcing $F = 8$ (strongly chaotic), 4th-order Runge-Kutta integration
- Observations: 32 of 40 variables observed, Gaussian noise $\sigma_o = 0.01$
- Inflation factor $\rho = 1.05$
- Assimilation window: 100 cycles
- Repetitions: **30 runs** per configuration

Neighboring variables are coupled through the dynamics → **true precision matrix is dense off-diagonal**.\
The hard case for diagonal targets.

---

# Testbed 2: AR(1) Cosmological Model

A simplified dynamical model whose stationary statistics match BOSS DR12 cosmological mocks. State dimension $d = 18$ (9 wavenumbers × 2 multipoles).

- Persistence $\phi = 0.95$ (smooth time evolution), calibrated to 2048 Patchy mocks
- Stationary covariance equals the BOSS mock covariance by construction
- Observations: 14 of 18 variables observed, Gaussian noise $\sigma_o = 100$
- Inflation factor $\rho = 1.02$
- Repetitions: **30 runs** per configuration

Cosmological modes are nearly independent → **true precision matrix is near-diagonal**.\
The easy case for diagonal targets.

---

# Why Both Testbeds?

The two systems have **opposite precision structure** by design:

- **Lorenz-96**: dense off-diagonal precision (hard case)
- **AR(1) cosmological**: near-diagonal precision (easy case)

Running the **same methods** on both lets us separate two effects:

- Does the **method** itself work?
- Does the **target match the truth**?

A single testbed would conflate the two.

---

# Eight Methods Compared

Head-to-head on both testbeds:

- **Baselines**: EnKF, EnKF-Cholesky
- **Covariance shrinkage**: Ledoit–Wolf
- **Target-free**: NERCOME
- **Direct precision shrinkage** (our framework), with four targets:
  - Identity
  - Eigenvalue
  - Cosmological *
  - **Scaled identity** (new in this thesis)

<div class="text-xs opacity-60 pt-4">
* The cosmological target can't be built on Lorenz-96 since it relies on domain-specific cosmological quantities.<br/>
Evaluated only on the cosmological testbed.
</div>

---

# Ensemble Size: Lower Bound

**How small can $N_e$ be?** Two constraints:

- **$N_e > n + 2$**: needed for $\mathbf{S}$ to be invertible
- **$N_e > n + 4$**: needed for the Bodnar coefficients ($\alpha^*$, $\beta^*$) to be well-defined

For Lorenz-96 ($n = 40$), that's $N_e \geq 44$. Our smallest tested value is $N_e = 50$, six members above the bound.

Anything smaller falls into the **under-sampled regime** that this thesis doesn't cover.

---

# Ensemble Size: Upper Bound

**How large can $N_e$ be before shrinkage becomes pointless?**

As $N_e \gg n$:

- The noise in $\mathbf{S}$ shrinks (scales as $n^2 / N_e$)
- The shrinkage weight $\lambda$ drops toward $0$
- $\mathbf{S}$ is already good, the target stops mattering

Beyond $N_e \approx 3n$, shrinkage offers no meaningful gain over plain $\mathbf{S}$.

For Lorenz-96 we stop at $N_e = 100$ ($n / N_e = 0.4$), safely inside the interesting regime.

---

# Tested Ensemble Sizes: Lorenz-96

State dimension $n = 40$.

| $N_e$ | $n/N_e$ | Regime & role                                         |
| ----- | ------- | ----------------------------------------------------- |
| 50    | 0.80    | Ill-conditioned. Stress at the invertibility bound   |
| 60    | 0.67    | Moderately ill-conditioned                            |
| 80    | 0.50    | Moderately ill-conditioned                            |
| 100   | 0.40    | Moderate → well-sampled boundary                      |

Four values chosen to span the regime between the lower and upper bounds.

---

# Tested Ensemble Sizes: Cosmological

State dimension $d = 18$.

| $N_e$ | $n/N_e$ | Regime & role                                         |
| ----- | ------- | ----------------------------------------------------- |
| 24    | 0.75    | Ill-conditioned. Matches operational mock budgets    |
| 30    | 0.60    | Moderate (Looijmans et al. working point)             |
| 40    | 0.45    | Moderate                                              |
| 50    | 0.36    | Moderate → well-sampled boundary                      |

Validated empirically with a finer 12-value $N_e$ scan (in the appendix). \
No sharp transitions between these points.

---
layout: image-right
image: ./figures/rmse_vs_ne.png
---

# RMSE vs $N_e$ on Lorenz-96

<div class="text-sm">

- Direct precision methods: **erratic**, RMSE 0.5–1.8 × 10⁻²
- Ledoit–Wolf: **lowest** RMSE, stable across $N_e$
- EnKF-Cholesky: competitive via banded structure
- Standard EnKF: diverges at small $N_e$

**Cause:** dense precision + diagonal targets → severe coefficient miscalibration.

</div>

---

# Why Do Direct Precision Methods Fail on Lorenz-96?

Look at the shrinkage coefficients:

- $\alpha^*$ stays within its valid range, near the theoretical bound $1 - n/N_e$.
- $\beta^*$ **blows up to $10^8$–$10^9$**, far outside the valid range $[0, 1]$, so it gets clamped back to $1$.
- **Clamping happens > 80% of cycles** across all three diagonal targets.

Result: the precision matrix $\boldsymbol{\Pi}_{\text{LS}}$ is highly ill-conditioned (condition number $10^7$–$10^{11}$, vs. $10^2$–$10^3$ for Ledoit–Wolf).

**Root cause: dense precision + diagonal target = forced mismatch.** The Bodnar formula tries to correct by inflating $\beta^*$; clamping prevents the correction.

---
layout: image-right
image: ./figures/beta_vs_ne.png
---

# Unclamped $\beta^*$ Explodes

<div class="text-sm">

Unclamped $|\beta^*|$ vs $N_e$ (log scale). Dashed line: clamp threshold at $\beta^* = 1$ (valid range $[0, 1]$).

- **Identity target:** $10^7$–$10^9$ times too large
- **Scaled identity:** better, but still needs clamping
- **Eigenvalue target:** the only one that behaves normally here

</div>

---
layout: image-right
image: ./figures/cosmo_enkf_rmse.png
---

# Cosmological Domain

<div class="text-sm">

- All methods stable, **monotonic RMSE reduction** with $N_e$
- Direct precision with cosmological target → lowest Frobenius loss
- Scaled identity: second-best domain-agnostic method
- **Ledoit–Wolf wins in EnKF RMSE**

</div>

---

# The Central Finding

<div class="text-center text-3xl pt-8 font-bold">
Best standalone estimator ≠ best EnKF filter.
</div>

<div class="text-center text-lg pt-6 opacity-80">
A disconnect not previously documented in the shrinkage literature.
</div>

---

# Standalone vs Sequential (Cosmological Domain)

**Standalone precision estimation** (one-shot Frobenius loss):

- Direct precision with **cosmological target** wins, loss $\approx 0.47$
- Other direct precision targets: eigenvalue > scaled identity > identity (all $\approx 0.54$)
- Ledoit–Wolf worst, loss $\approx 0.93$–$0.96$

**EnKF filtering** (RMSE over 100 cycles):

- Ledoit–Wolf **wins**
- Direct precision with cosmological target: second
- Other direct precision targets: scaled identity > eigenvalue > identity \
(ours is best among domain-agnostic)


---

# Why the Ranking Flips

The EnKF applies the estimator at every cycle, so **variance compounds**.

- Ledoit–Wolf std across draws: **$\pm 0.01$**
- Direct precision std: **$\pm 0.13$–$0.14$** (order of magnitude larger)

Over 100 cycles, a 100-fold variance gap overwhelms the bias advantage.

**Principle:** low-variance beats low-bias when errors accumulate sequentially.

---

# Asymptotic Cost

<div class="text-xs">

| Method                                                     | Cost per cycle                   |
| ---------------------------------------------------------- | -------------------------------- |
| EnKF                                                       | $\mathcal{O}(n^2 N_e + m^2 N_e)$ |
| EnKF-Cholesky                                              | $\mathcal{O}(n \, r \, N_e)$     |
| Direct precision (identity / scaled identity / cosmo)      | $\mathcal{O}(n^2 N_e + n^3)$     |
| Direct precision (eigenvalue target)                       | $\mathcal{O}(n^2 N_e + 2n^3)$    |
| NERCOME                                                    | $\mathcal{O}(B(n^2N_e + n^3))$   |
| Ledoit–Wolf                                                | $\mathcal{O}(n^2 N_e)$           |

</div>

<div class="text-xs opacity-75">

$n$: state dimension · $N_e$: ensemble size · $m$: observations · $r$: Cholesky bandwidth · $B$: bootstrap repeats (~200)
</div>

---

# Cost Takeaways

- **Direct precision**: adds an $\mathcal{O}(n^3)$ inversion on top of what EnKF already does
- **NERCOME**: multiplies the cost of direct precision by $B$ (~200 bootstrap repeats)
- **EnKF-Cholesky**: the only method **linear in $n$**; only one that scales to operational dimensions

---

# Measured Wall-Clock

**Lorenz-96** ($n = 40$, $N_e = 80$, per 100 cycles):

- All shrinkage methods: 26–28 s
- NERCOME: **215 s**, 8× slower

**Cosmological** ($d = 18$, $N_e = 30$, per 100 cycles):

- Shrinkage methods: 0.20–0.28 s
- NERCOME: **26.7 s**, 190× slower
- EnKF baseline: 0.14 s

Ensemble propagation dominates on Lorenz-96; precision estimation dominates on cosmo.

---
layout: image-right
image: ./figures/timing_scaling.png
---

# Scaling to Operational Dimensions

<div class="text-sm">

Extrapolation from measured timings using asymptotic formulas:

- **$n \lesssim 100$:** cost not a differentiator
- **$n \sim 10^2$–$10^3$:** $n^3$ inversion dominates; NERCOME ruled out
- **$n \gtrsim 10^4$:** only EnKF-Cholesky (linear in $n$) survives

**Practical implication:** direct precision shrinkage is best suited to mid-scale testbeds or combined with localization at operational scale.

</div>

---

# Practical Recommendation

<div class="text-sm pt-2">

| Regime ($n/N_e$)        | Precision structure | Target known? | Recommended                                    |
| ----------------------- | ------------------- | ------------- | ---------------------------------------------- |
| $>1$ (under-sampled)    | any                 | n/a           | Localization + EnKF-Cholesky                   |
| $0.5$–$1$               | near-diagonal       | **yes**       | Direct precision + domain target               |
| $0.5$–$1$               | near-diagonal       | **no**        | **Scaled identity** (this work) or Ledoit–Wolf |
| $0.5$–$1$               | dense               | n/a           | Ledoit–Wolf; EnKF-Cholesky at $n \gtrsim 10^3$ |
| $<0.3$ (well-sampled)   | any                 | n/a           | Plain EnKF (shrinkage weight $\lambda \to 0$)  |

</div>

<br/>

Accuracy drives the choice within the $\mathcal{O}(n^3)$ band; cost drives it at the extremes.

---

# Contributions

1. **First sequential embedding** of Bodnar direct precision shrinkage, coefficients recomputed every cycle inside the EnKF.

2. **Novel scaled identity target**, a data-driven middle ground between the plain identity and the domain-specific cosmological target.

3. **Dual-domain benchmark** on systems with opposite precision structure, separating method design from target-truth alignment.

4. **Standalone ≠ sequential**, the key finding: estimator variance matters as much as bias when errors accumulate.

5. **Systematic ensemble-size study**, documenting when and how the shrinkage coefficients fail.

6. **Open-source implementation** in the TEDA framework for all six estimators.

---

# Limitations

- **Under-sampled regime not covered**, when $N_e < n$, $\mathbf{S}$ is singular and all direct precision methods break. Would need localization.
- **Low-dimensional testbeds**, $n = 40$ and $d = 18$ are tiny compared to operational DA ($10^6$–$10^8$).
- **Simplified cosmological model**, AR(1) is linear; real surveys involve nonlinear structure growth.
- **Linear observation operators only**, no tests with nonlinear measurement models.
- **Heuristic factor of 2** in the scaled target, picked by analogy to cosmology; could be cross-validated instead.

---

# Future Work: Methodology

- **Shrinkage × localization**, combine the two regularization strategies
- **Intermediate-scale systems**, test scaling on quasi-geostrophic models ($n \sim 10^4$)
- **Adaptive target selection**, switch target based on the data's off-diagonal structure
- **Theoretical analysis of sequential accumulation**, derive bias/variance trade-off rigorously

---

# Future Work: Estimator Extensions

- **Principled scaling factor**, replace the heuristic 2 with a cross-validated value
- **Non-Gaussian samples**, extend Bodnar's derivation beyond Gaussian ensembles
- **Condition-number bound**, truncate tiny eigenvalues to prevent inversion blow-up
- **Time-adaptive target**, update the target from a running mean of past estimates

---
layout: center
class: intro-slide text-center
---

# Thank you


<div class="pt-6 opacity-70 text-sm">
Andrés F. Movilla Obregón · Universidad del Norte<br/>
Tutor: Elias D. Niño-Ruiz, Ph.D.
</div>

