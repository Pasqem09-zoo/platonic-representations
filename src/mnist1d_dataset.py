import torch
from torch.utils.data import TensorDataset, DataLoader
import mnist1d


def load_mnist1d(batch_size=64,
                 n_samples=1000,
                 path="./data/mnist1d_data.pkl"):
    """
    Load MNIST1D dataset and return PyTorch DataLoaders.

    Parameters:
        batch_size (int)
        n_samples (int): number of test samples (for CKA)
        path (str): where dataset file is stored

    Returns:
        train_loader, test_loader
    """

    args = mnist1d.data.get_dataset_args()

    data = mnist1d.data.get_dataset(
        args,
        path=path,
        download=True,
        regenerate=False
    )

    # Convert to torch tensors
    X_train = torch.tensor(data["x"], dtype=torch.float32)
    y_train = torch.tensor(data["y"], dtype=torch.long)

    X_test = torch.tensor(data["x_test"], dtype=torch.float32)
    y_test = torch.tensor(data["y_test"], dtype=torch.long)

    # Add channel dimension: (N, 40) → (N, 1, 40)
    X_train = X_train.unsqueeze(1)
    X_test = X_test.unsqueeze(1)

    # Subsample test set
    if n_samples is not None:
        X_test = X_test[:n_samples]
        y_test = y_test[:n_samples]

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True)

    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False)

    return train_loader, test_loader
