import torch


@torch.no_grad() #decorator that disables gradient calculation beacuse we are only interested in the activations of the fc1 layer and not in updating the model's parameters during this process. This can save memory and speed up computations since we don't need to track gradients for backpropagation
def extract_fc1_features(model, dataloader, device="cpu"):
    """
    Extract representations from the fc1 layer of the model.

    Parameters:
    - model: the neural network model from which to extract features, in this case we are interested in the activations of the fc1 layer
    - dataloader: data loader providing input images
    - device: cpu or cuda/mps

    Returns:
    - features: tensor of shape (N, D)
    """

    model.eval()

    all_features = []

    for images, _ in dataloader: #iteration over the dataloader: batches and labels (the underscore is a common convention in Python to indicate that the variable is intentionally unused, in this case, we don't need the labels for feature extraction)
        images = images.to(device)

        # Forward up to fc1
        x = model.features_extractor(images)
        x = model.fc1(x)

        all_features.append(x.cpu())

    features = torch.cat(all_features, dim=0)

    return features
