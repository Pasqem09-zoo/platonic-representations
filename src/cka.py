import torch


def center_kernel(K: torch.Tensor) -> torch.Tensor: #function that centers a kernel matrix K using the centering matrix H. The kernel matrix K is expected to be of shape (n x n), where n is the number of samples
    """
    Center a kernel matrix K using the centering matrix H.
    K is expected to be (n x n).
    H = I - (1/n) 11^T where I is identity matrix, 11^T is a matrix of ones, n is the number of samples, 1 is a vector of ones
    """
    n = K.size(0) # K appartiene a R^(nxn)
    device = K.device

    H = torch.eye(n, device=device) - torch.ones(n, n, device=device) / n # H = I - (1/n) * 11^T where I is identity matrix, 11^T is a matrix of ones, n is the number of samples, 1 is a vector of ones
    return H @ K @ H #center the kernel matrix K by multiplying it on both sides with the centering matrix H. necessary for computing CKA similarities correctly
    # @ is the matrix multiplication operator in PyTorch


def hsic(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """
    Hilbert-Schmidt Independence Criterion:
        HSIC(K, L) = Tr(K̄ L̄)
    (normalization constant omitted, as it cancels in CKA)
    """
    K_bar = center_kernel(K) #HKH, centered around the mean of data points
    L_bar = center_kernel(L) #HLH, centered around the mean of data points

    return torch.trace(K_bar @ L_bar) #computes the trace of the product of the centered kernels, which gives the HSIC value


def linear_cka(phi: torch.Tensor, psi: torch.Tensor) -> torch.Tensor: # from phi (n x d1) and psi (n x d2) to scalar CKA value
    """
    Compute linear CKA between two representations phi and psi.

    phi: (n x d1) representations of model A
    psi: (n x d2) representations of model B
    Returns: scalar CKA value
    """

    # Compute linear kernels
    K = phi @ phi.T #phi.T is phi transpose, with K_ij = <phi_i, phi_j>
    L = psi @ psi.T #psi.T is psi transpose, with L_ij = <psi_i, psi_j>

    # CKA normalization
    eps = 1e-10 #small constant to prevent division by zero in case of degenerate kernels
    return hsic(K, L) / (torch.sqrt(hsic(K, K) * hsic(L, L)) + eps)