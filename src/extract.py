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
        if hasattr(model, "conv1"):
            # Forward pass up to fc1: CNN case
            x = model.conv1(images)
            x = torch.relu(x)
            x = model.pool(x)

            x = model.conv2(x)
            x = torch.relu(x)
            x = model.pool(x)

            x = x.view(x.size(0), -1)
            x = model.fc1(x)

        else:
            # Forward pass up to fc1: MLP case
            x = images.view(images.size(0), -1) #flatten the input images into a 2D tensor of shape (batch_size, 3*32*32) before feeding them into the fully connected layers of the MLP. This is necessary because the MLP expects a 2D input where each row corresponds to a flattened image in the batch
            x = model.fc1(x)

        all_features.append(x.cpu()) #append the extracted features to the list all_features

    features = torch.cat(all_features, dim=0) #concatenate all the feature tensors along the first dimension (dim=0) to create a single tensor of shape (N, D), where N is the total number of samples and D is the dimensionality of the features extracted from the fc1 layer
    return features
