# The Platonic Representation Hypothesis  
### A Small-Scale Empirical Study of Representational Convergence

This repository contains a controlled experimental study on **representational convergence** in neural networks, inspired by:

> Huh et al., *The Platonic Representation Hypothesis*, ICML 2024.

Developed for the course  
**Computer Vision and Intelligent Media Recognition**

---

## Project Goal

We investigate the question:

> Do neural networks trained independently on the same task converge toward similar internal representations?

More formally:

$$
f_1(x) \approx f_2(x) \;\not\Rightarrow\; \Phi_1(x) \approx \Phi_2(x)
$$

We study whether latent representations converge under controlled training conditions.

---

## Experimental Setup

### Dataset
- MNIST-1D  
- 1000 test samples used for representation extraction  

### SimpleCNN

```text
Input (1 x 40)
    ↓
Conv1D + ReLU (1 → 15, kernel=3, stride=2)
    ↓
Conv1D + ReLU (15 → 15, kernel=3, stride=2)
    ↓
Conv1D + ReLU (15 → 15, kernel=3, stride=2)
    ↓
Flatten (15 x 4 = 60)
    ↓
Fully Connected (fc1) ← representation layer (60 → feature_dim)
    ↓
ReLU
    ↓
Linear Classifier (fc2) (feature_dim → 10)
```

### SimpleMLP

```text
Input (1 x 40)
    ↓
Flatten (40)
    ↓
Linear (fc1) + ReLU ← representation layer (40 → feature_dim)
    ↓
Linear (fc2) + ReLU (feature_dim → feature_dim)
    ↓
Linear Output (fc_out) (feature_dim → 10)
```



### Training Protocol
- 40 random seeds  
- 30 epochs  
- Identical optimizer and hyperparameters  
- Feature dimension $d = 32$  
- Representations extracted from layer `fc1`  
- Representations evaluated on the **test set**

---

## Project Structure

```bash
.
├── model.py
│   ├── SimpleCNN
│   │   ├── 3 × Conv1d + ReLU backbone
│   │   ├── fc1 → representation layer (used for CKA)
│   │   └── fc2 → classification layer (10 classes)
│   └── SimpleMLP
│       ├── fc1 → representation layer (used for CKA)
│       ├── fc2 → hidden layer
│       └── fc_out → classification layer
│
├── train.py
│   └── Multi-seed training loop
│
├── extract.py
│   └── Feature extraction from fc1 layer
│
├── cka.py
│   ├── Linear kernel computation
│   ├── Kernel centering
│   ├── HSIC computation
│   └── CKA normalization
│
├── analyze_cka.py
│   ├── Pairwise CKA across seeds
│   ├── Statistical analysis
│   └── Shapiro–Wilk normality test
│
├── mnist1d_dataset.py
│   └── Dataset loading and preprocessing
│
├── config.py
│   └── Experiment hyperparameters
│
├── utils.py
│   ├── Seed control
│   └── Helper functions
│
├── main.py
│   └── Full experimental pipeline
│
└── README.md
```

## Methodology

For each experiment:

1. Train models independently with different random seeds.
2. Extract latent representations from the layer `fc1`
where:
- `N` = number of test samples used for extraction (e.g., 1000)
- `d` = feature dimension of `fc1` (e.g., 32)

3. Compute Linear Centered Kernel Alignment (CKA) between representations.
4. Analyze the distribution of pairwise similarities:
- Mean
- Standard deviation
- Min / Max
- Skewness
- Kurtosis
- Shapiro–Wilk normality test

---

## Experiments

### 1. Intra-Model (CNN)

Compare `fc1` representations across different seeds of the same `SimpleCNN` architecture.

### 2. Intra-Model (MLP)

Compare `fc1` representations across different seeds of the same `SimpleMLP` architecture.

### 3. Inter-Model (CNN vs MLP)

Compare `fc1` representations between `SimpleCNN` and `SimpleMLP` trained on the same task.

---

## Results

| Experiment            | #Seeds | #Pairs | CKA Mean | CKA Std |
|-----------------------|--------|--------|----------|---------|
| CNN (intra)           | 40     | 780    | 0.9424   | 0.0231  |
| MLP (intra)           | 40     | 780    | 0.9755   | 0.0052  |
| CNN vs MLP (inter)    | 40     | 1600   | 0.7712   | 0.0316  |

---

## Interpretation

- High intra-model CKA indicates representational stability across random initializations.
- Lower inter-model CKA suggests architectural inductive bias influences representation geometry.
- Similarity distributions are concentrated but not Gaussian (normality rejected by Shapiro–Wilk).

These results suggest partial representational convergence within architectures, but weaker convergence across different model classes.

---

## References

- Huh et al., *The Platonic Representation Hypothesis*, ICML 2024.


