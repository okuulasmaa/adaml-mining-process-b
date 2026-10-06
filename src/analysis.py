import numpy as np


def pca(X, n, verbose=False):

    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    V = Vt.T
    loadings = V[:, 0:n]
    scores = X @ loadings

    if verbose:
        # Explained variances
        explained_variances = (S**2) / np.sum(S**2)
        print(f"Explained variance by PC1: {explained_variances[0]:.3f}")
        print(f"Explained variance by PC2: {explained_variances[1]:.3f}")
        print(f"Explained variance by PC1 and PC2: {(explained_variances[0] + explained_variances[1]):.3f}")

    return scores, loadings


def zscore(X: np.ndarray, mu=None, sig=None):
    
    if mu is None:
        mu = np.mean(X, axis=0)
    if sig is None:
        sig = np.std(X, axis=0)
    
    return (X - mu) / sig