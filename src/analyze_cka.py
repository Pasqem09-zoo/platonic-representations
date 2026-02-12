import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys
import os

# --------------------------------------------
# Usage:
# python analyze_cka.py cka_cnn.npy
# python analyze_cka.py cka_mlp.npy
# python analyze_cka.py cka_inter.npy
# --------------------------------------------

if len(sys.argv) < 2:
    print("Usage: python analyze_cka.py <filename.npy>")
    sys.exit(1)

filename = sys.argv[1]
path = os.path.join("results", filename)

values = np.load(path)

# --------------------------------------------------
# Basic statistics
# --------------------------------------------------

n = len(values)
mu = np.mean(values)
sigma = np.std(values)
min_val = np.min(values)
max_val = np.max(values)
skewness = stats.skew(values)
kurtosis = stats.kurtosis(values)  # excess kurtosis
W, p_value = stats.shapiro(values)

print("File:", filename)
print("Number of samples:", n)
print("Mean:", mu)
print("Std:", sigma)
print("Min:", min_val)
print("Max:", max_val)
print("Skewness:", skewness)
print("Excess Kurtosis:", kurtosis)
print("Shapiro-Wilk statistic:", W)
print("p-value:", p_value)

os.makedirs("results", exist_ok=True)

# --------------------------------------------------
# 1️⃣ Histogram + Gaussian fit
# --------------------------------------------------

plt.figure(figsize=(6,4))

plt.hist(values, bins=20, density=True, alpha=0.6, label="Empirical")

x = np.linspace(min_val, max_val, 300)
plt.plot(x, stats.norm.pdf(x, mu, sigma), label="Gaussian fit")

plt.title(f"Histogram: {filename.replace('.npy','')}")
plt.xlabel("CKA")
plt.ylabel("Density")
plt.legend()

plt.tight_layout()
plt.savefig(f"results/{filename.replace('.npy','')}_hist.png", dpi=300)
plt.close()

# --------------------------------------------------
# 2️⃣ Q-Q plot
# --------------------------------------------------

plt.figure(figsize=(6,4))
stats.probplot(values, dist="norm", plot=plt)
plt.title(f"Q-Q Plot: {filename.replace('.npy','')}")
plt.tight_layout()
plt.savefig(f"results/{filename.replace('.npy','')}_qq.png", dpi=300)
plt.close()

# --------------------------------------------------
# 3️⃣ Kernel Density Estimate (KDE)
# --------------------------------------------------

plt.figure(figsize=(6,4))

kde = stats.gaussian_kde(values)
x = np.linspace(min_val, max_val, 300)

plt.plot(x, kde(x), label="KDE")
plt.title(f"KDE: {filename.replace('.npy','')}")
plt.xlabel("CKA")
plt.ylabel("Density")
plt.legend()

plt.tight_layout()
plt.savefig(f"results/{filename.replace('.npy','')}_kde.png", dpi=300)
plt.close()

print("Plots saved in /results folder.")
