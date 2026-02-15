# -----------------------------
# Experiment configuration
# -----------------------------

# Dataset
DATASET_NAME = "MNIST1D"
DATA_ROOT = "data"

# Training
BATCH_SIZE = 128
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3

# Seeds
SEEDS = list(range(40))   # scalable: e.g. range(10)

# Logging
USE_WANDB = True # True or False if set WANDB_MODE "disabled"
WANDB_PROJECT = "platonic-representations"
WANDB_MODE = "online"   # "offline" during development, "online" for logging to the cloud, "disabled" to turn off logging entirely

# Device
USE_MPS = True

# Models to compare
MODELS = ["cnn", "mlp"] # Options: ["cnn"], ["mlp"], ["cnn", "mlp"]

# Model feature dimension (shared)
FEATURE_DIM = 32 #64

# CKA analysis options
DO_INTRA_MODEL_CKA = False
DO_INTER_MODEL_CKA = True

# Which models to use for intra-model CKA
# Options: ["cnn"], ["mlp"], ["cnn", "mlp"]
INTRA_MODEL_TARGETS = ["cnn","mlp"]

