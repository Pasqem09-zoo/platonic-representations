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
SEEDS = [0, 1]   # scalable: e.g. range(10)

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
MODELS = ["cnn"] # Options: ["cnn"], ["mlp"], ["cnn", "mlp"]

# Model feature dimension (shared)
FEATURE_DIM = 32 #64

# CKA analysis options
DO_INTRA_MODEL_CKA = True
DO_INTER_MODEL_CKA = False

# Which models to use for intra-model CKA
# Options: ["cnn"], ["mlp"], ["cnn", "mlp"]
INTRA_MODEL_TARGETS = ["cnn"]

