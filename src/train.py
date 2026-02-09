import torch
import torch.nn as nn
import torch.optim as optim

import wandb


def train_model(
    model,
    dataloader,
    num_epochs: int = 5, #how many times the entire training dataset will be passed through the model during training. A higher number of epochs can lead to better performance but also increases the risk of overfitting
    lr: float = 1e-3,
    device: str = "cpu"
):
    """
    Train a neural network model using cross-entropy loss.

    Parameters:
    - model: neural network to train
    - dataloader: training data loader
    - num_epochs: number of training epochs
    - lr: learning rate
    - device: cpu / cuda / mps
    """

    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss() #standard loss for classification. it messures the difference between the predicted class probabilities (outputs) and the true class labels (labels)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for images, labels in dataloader: #loop over the batch
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad() #clear the gradients of all optimized tensors before the backward pass. This is necessary because by default, gradients are accumulated in PyTorch
            outputs = model(images) #forward pass: compute predicted outputs by passing inputs to the model
            loss = criterion(outputs, labels) # mean loss for the batch 
            loss.backward() #backward pass: compute gradient of the loss with respect to parameters
            optimizer.step() #parameter update

            epoch_loss += loss.item() #accumulate loss for the epoch for monitoring purposes

        avg_loss = epoch_loss / len(dataloader) #average loss for epoch "epoch" by dividing the total loss by the number of batches in the dataloader
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}") #print the average loss for the epoch
        
        # Log training loss to Weights & Biases (wandb) for monitoring and visualization of the training process
        wandb.log({
            "train/loss": avg_loss,
            "epoch": epoch
        })