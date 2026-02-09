# -----------------------------
# Experiment configuration
# -----------------------------

# Dataset
DATASET_NAME = "CIFAR10"
DATA_ROOT = "data"

# Training
BATCH_SIZE = 128
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3

# Seeds
SEEDS = range(5)   # scalable: e.g. range(10)

# Model
ARCHITECTURE = "SimpleCNN"
FEATURE_LAYER = "fc1"   # layer used for CKA extraction

# Logging
USE_WANDB = True
WANDB_PROJECT = "platonic-representations"
WANDB_MODE = "online"   # "offline" during development

# Device
USE_MPS = True

# Models to compare
MODELS = ["cnn", "mlp"]

# Model feature dimension (shared)
FEATURE_DIM = 128

# CKA analysis options
DO_INTRA_MODEL_CKA = False
DO_INTER_MODEL_CKA = True

# Which models to use for intra-model CKA
# Options: ["cnn"], ["mlp"], ["cnn", "mlp"]
INTRA_MODEL_TARGETS = ["cnn"]

