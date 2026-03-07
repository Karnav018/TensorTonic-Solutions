import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    # Write code here


    X = np.asarray(X, dtype=float)

    if X.ndim !=2:
        return None

    N, D = X.shape

    if N < 2:
        return None
        
    aver = np.mean(X, axis=0)

    centered = X - aver

    cover = (1 / (N - 1)) * (centered.T @ centered)

    return cover