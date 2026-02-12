"""
Main entry point for the project.

This script orchestrates the full experiment:
- model initialization
- training
- feature extraction
- CKA computation
"""

import torch
from torch.utils.data import DataLoader

import wandb

import config

from mnist1d_dataset import load_mnist1d

from model import SimpleCNN, SimpleMLP
from train import train_model
from extract import extract_fc1_features
from cka import linear_cka
from utils import set_seed, get_device


#------------------------------
# CKA analysis
#------------------------------
def compute_intra_cka(representations): #computer cka for all pairs of representations across different seeds for the same model
    values = []
    n = len(representations)
    for i in range(n):
        for j in range(i + 1, n):
            phi_i = representations[i]
            phi_j = representations[j]
            cka_ij = linear_cka(phi_i, phi_j)
            values.append(cka_ij.item())
    return values


def compute_inter_cka(reps_a, reps_b): #compute cka for all pairs of representations across different seeds for two different models (e.g., cnn vs mlp)
    values = []
    for phi_a in reps_a:
        for phi_b in reps_b:
            cka = linear_cka(phi_a, phi_b)
            values.append(cka.item())
    return values


def summarize(name, values): #print mean and sd of cka values for a given name (e.g., "CNN intra-CKA", "MLP intra-CKA", "CNN vs MLP inter-CKA")
    num = len(values)
    mean = sum(values) / num
    std = torch.std(torch.tensor(values)).item()
    min_val = min(values)
    max_val = max(values)
    print(f"{name} | num pairs: {num} | mean ± std: {mean:.4f} ± {std:.4f} | min: {min_val:.4f} | max: {max_val:.4f}")


#compute and summarize CKA values for intra-model comparisons (e.g., CNN vs CNN, MLP vs MLP) and inter-model comparisons (e.g., CNN vs MLP)
def run_cka_analysis(all_representations):
    # intra-model CKA
    if config.DO_INTRA_MODEL_CKA:
        for model_name in config.INTRA_MODEL_TARGETS:
            if model_name in all_representations:
                cka_values = compute_intra_cka(all_representations[model_name])
                summarize(f"CKA intra-{model_name.upper()}", cka_values)
    # inter-model CKA
    if config.DO_INTER_MODEL_CKA:
        cka_cross = compute_inter_cka(
            all_representations["cnn"],
            all_representations["mlp"]
        )
        summarize("CKA CNN–MLP", cka_cross)


def main():

    # --------------------------------------------------
    # 1. Setup
    # --------------------------------------------------
    device = get_device()  # select the appropriate device (CPU or Apple MPS) for computations
    print(f"Using device: {device}")

    # --------------------------------------------------
    # 2. Dataset and DataLoader
    # --------------------------------------------------
    train_loader, test_loader = load_mnist1d(
    batch_size=config.BATCH_SIZE,
    n_samples=1000,
    path="./data/mnist1d_data.pkl"
)


    # --------------------------------------------------
    # 3. Train multiple models with different seeds but same architecture, same dataset, same training procedure
    # --------------------------------------------------

    models_available = {
    "cnn": SimpleCNN,
    "mlp": SimpleMLP
    } #dictionary that maps model names (e.g., "cnn", "mlp") to their corresponding model classes (SimpleCNN, SimpleMLP). By iterating over this dictionary, you can train and evaluate multiple models making it easier to compare their performance and representations


    models_to_test = {}

    for name in config.MODELS:
        models_to_test[name] = models_available[name]


    all_representations = {} #Basically, this dictionary will store the extracted features (representations) for each model type (e.g., "cnn", "mlp") across different random seeds. The keys of the dictionary are the model names, and the values are lists of feature tensors corresponding to each seed

    for model_name, model_class in models_to_test.items(): #iterate over models in models_to_test, where model_name is the key (e.g., "cnn") and model_class is the value (e.g., SimpleCNN)

        print(f"\nTraining model: {model_name}")
        representations = []

        for seed in config.SEEDS:

            wandb.init(
                project="platonic-representations",
                name=f"{model_name}_seed_{seed}",
                config={
                    "architecture": model_class.__name__,
                    "dataset": config.DATASET_NAME,
                    "epochs": config.NUM_EPOCHS, #number of times the entire training dataset will be passed through the model during training. A higher number of epochs can lead to better performance but also increases the risk of overfitting
                    "seed": seed
                },
                mode=config.WANDB_MODE if config.USE_WANDB else "disabled" # "online" for logging to the cloud, "offline" for local logging, "disabled" to turn off logging entirely. Using "offline" during development can help avoid issues with network connectivity and allows you to log data without an internet connection. You can later sync the offline logs to the cloud when you're ready.
            )

            set_seed(seed)
            model = model_class(feature_dim=config.FEATURE_DIM)
            train_model(model, train_loader, num_epochs=config.NUM_EPOCHS, device=device)

            phi = extract_fc1_features( #extract the activations from the fc1 layer of the trained model using the test_loader to pass the test images through the model and collect the features
                model,
                test_loader,
                device=device
            )

            print(f"{model_name} | seed {seed} | shape {phi.shape}")

            representations.append(phi)
            wandb.finish()

        all_representations[model_name] = representations #store the list of feature tensors for the current model type 

    # --------------------------------------------------
    # 4. CKA analysis (B3)
    # --------------------------------------------------
    run_cka_analysis(all_representations)



if __name__ == "__main__":
    main()