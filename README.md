# The Platonic Representation Hypothesis

This repository contains a small-scale experimental study on **representational convergence** in neural networks, inspired by the paper  
**“The Platonic Representation Hypothesis”** (Huh et al., 2024).

The project is developed as part of the course  
**Computer Vision and Intelligent Media Recognition**.

---

## Project Goal

The objective of this project is to empirically investigate the following question:

> *Do neural networks with the same architecture, trained on the same task but with different random initializations, learn similar internal representations?*

To answer this, we:
- train multiple identical convolutional neural networks (CNNs) with different random seeds,
- extract intermediate feature representations from a fixed layer,
- measure their similarity using **Centered Kernel Alignment (CKA)**,
- report the mean and standard deviation of similarity across model pairs.

This setup represents a **controlled and simplified version** of the representational convergence phenomena discussed in the reference paper.

---

## Method Overview

The experimental pipeline is structured as follows:

1. **Dataset**
   - A standard vision dataset (e.g. CIFAR-10) is used.
   - The same training and test splits are shared across all models.

2. **Model**
   - A simple convolutional neural network architecture.
   - All hyperparameters are kept fixed.
   - Only the random seed is varied between runs.

3. **Training**
   - Multiple models are trained independently.
   - Identical training protocol for all runs.

4. **Representation Extraction**
   - Activations are extracted from a selected intermediate layer.
   - A fixed set of input images is used for all models.

5. **Similarity Measurement**
   - Linear **Centered Kernel Alignment (CKA)** is computed between representations.
   - Pairwise similarities are aggregated using mean and standard deviation.
  
---

## Repository Structure

src:
  model.py: CNN architecture
  train.py: Training loop
  extract.py: Feature extraction
  cka.py: CKA implementation
  utils.py: Utilities (seeds, helpers)

data: Datasets (ignored by git)
runs: Training outputs (ignored by git)

README.md: Project description
.gitignore: Git ignore rules

---

## Reference

Minyoung Huh, Brian Cheung, Tongzhou Wang, Phillip Isola  
**The Platonic Representation Hypothesis**, ICML 2024.

---

## Notes

- This repository is intended for **educational and experimental purposes**.
- The focus is on clarity, reproducibility, and conceptual understanding rather than large-scale performance.

