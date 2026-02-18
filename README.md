# The Platonic Representation Hypothesis  
### A Small-Scale Empirical Study of Representational Convergence

This repository contains a controlled experimental study on **representational convergence** in neural networks, inspired by:

> Huh et al., *The Platonic Representation Hypothesis*, ICML 2024.

Developed for the course  
**Computer Vision and Intelligent Media Recognition**.

---

## Project Goal

We investigate the question:

> Do neural networks trained independently on the same task converge toward similar internal representations?

More formally:

\[
f_1(x) \approx f_2(x) \;\not\Rightarrow\; \Phi_1(x) \approx \Phi_2(x)
\]

We study whether latent representations converge under controlled training conditions.

---

## Experimental Setup

**Dataset**
- MNIST1D  
- 1000 test samples used for representation extraction  

**Models**
- SimpleCNN (1D convolutional network)
- SimpleMLP (fully connected network)

**Training Protocol**
- 40 random seeds
- 30 epochs
- Identical optimizer and hyperparameters
- Feature dimension \( d = 32 \)
- Representations extracted from layer `fc1`
- Representations evaluated on the **test set** (unseen data) to measure general representational similarity

**project structure**
.
├── model.py
│   ├── SimpleCNN
│   │   ├── 1D convolutional backbone (3 Conv1d + ReLU)
│   │   ├── fc1 → representation layer (used for CKA)
│   │   └── fc2 → classification layer (10 classes)
│   └── SimpleMLP
│       ├── fc1 → representation layer (used for CKA)
│       ├── fc2 → hidden layer
│       └── fc_out → classification layer
│
├── train.py
│   └── Training loop (multi-seed experiments)
│
├── extract.py
│   └── Feature extraction from fc1 layer
│
├── cka.py
│   ├── Kernel computation (linear kernel)
│   ├── Centering operation
│   ├── HSIC computation
│   └── CKA normalization
│
├── analyze_cka.py
│   ├── Pairwise CKA across seeds
│   ├── Statistical analysis (mean, std, skew, kurtosis)
│   └── Normality tests (Shapiro-Wilk)
│
├── mnist1d_dataset.py
│   └── MNIST-1D dataset loading and preprocessing
│
├── config.py
│   └── Experiment hyperparameters (batch size, seeds, feature_dim, etc.)
│
├── utils.py
│   ├── Seed control utilities
│   └── Helper functions
│
├── main.py
│   └── Entry point: full pipeline
│       ├── Train models (different random seeds)
│       ├── Extract representations
│       ├── Compute CKA
│       └── Aggregate statistics
│
└── README.md


---

## Methodology

For each experiment:

1. Train models independently with different seeds.
2. Extract latent representations:
   \[
   \Phi_s(X) \in \mathbb{R}^{n \times d}
   \]
3. Compute **Linear Centered Kernel Alignment (CKA)**.
4. Analyze the full distribution of pairwise similarities:
   - Mean
   - Standard deviation
   - Min / Max
   - Skewness
   - Kurtosis
   - Shapiro–Wilk normality test

---

## Experiments

### 1. Intra-Model (CNN)
CKA across different seeds of the same CNN architecture.

### 2. Intra-Model (MLP)
CKA across different seeds of the same MLP architecture.

### 3. Inter-Model (CNN vs MLP)
CKA across different architectures trained on the same task.

---

## Results

| Experiment | #Seeds | #Pairs | CKA Mean | CKA Std |
|------------|--------|--------|----------|---------|
| CNN (intra) | 40 | 780 | 0.9424 | 0.0231 |
| MLP (intra) | 40 | 780 | 0.9755 | 0.0052 |
| CNN vs MLP (inter) | 40 | 1600 | 0.7712 | 0.0316 |

**Observations**
- Intra-model alignment is high → representational stability across random initialization.
- Inter-model alignment is lower → architecture influences representation geometry.
- Distributions are concentrated but not perfectly Gaussian (normality rejected).

---

## Repository Structure

