import random
import numpy as np
import torch


def set_seed(seed: int):
    """
    Set random seed for reproducibility.

    This fixes:
    - Python random
    - NumPy
    - PyTorch (CPU)
    """

    random.seed(seed) #seed for the built-in random module in Python, which is used for generating random numbers
    np.random.seed(seed) #seed for NumPy's random number generator, which is used for generating random numbers in NumPy
    torch.manual_seed(seed) #seed for PyTorch's random number generator, which is used for generating random numbers in PyTorch


    # # For deterministic behavior (if supported (nvidia))
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def get_device():
    """
    Select the appropriate device.
    Priority:
    - Apple MPS (if available)
    - CPU
    """

    if torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
