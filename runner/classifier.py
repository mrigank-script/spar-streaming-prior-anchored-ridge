"""Closed-form one-vs-rest classifier with predict-before-update streaming.

Statistics are cumulative (A = sum phi phi^T, B = sum phi y^T) with effective
count n; the anchored solve is
  W = (A + n (gamma + ridge) I)^{-1} (B + n gamma G),
so the warm-start prior is retained and the regularizer keeps constant relative
strength over the stream.
"""

import numpy as np


def onehot(y, K):
    Y = np.zeros((len(y), K), dtype=np.float64)
    Y[np.arange(len(y)), np.asarray(y, dtype=int)] = 1.0
    return Y


def _anchored(A, B, n, gamma, ridge, G):
    D = A.shape[0]
    return np.linalg.solve(A + n * (gamma + ridge) * np.eye(D), B + n * gamma * G)


def fit_warm(X_train, y_train, K, ridge: float = 0.01, gamma: float = 0.0, G=None):
    n = float(len(X_train))
    A = X_train.T @ X_train
    B = X_train.T @ onehot(y_train, K)
    if G is None:
        G = np.linalg.solve(A + n * ridge * np.eye(A.shape[0]), B)
    return _anchored(A, B, n, gamma, ridge, G)


def stream(X_train, y_train, X, y, K, alpha: float = 1.0, gamma: float = 0.0,
           ridge: float = 0.01, G=None, reanchor_every: int = 1):
    """Predict-then-update over trials, retaining the warm-start prior."""
    A = X_train.T @ X_train
    B = X_train.T @ onehot(y_train, K)
    n = float(len(X_train))
    if G is None:
        G = np.linalg.solve(A + n * ridge * np.eye(A.shape[0]), B)
    prior = np.asarray(G, dtype=np.float64)
    W = _anchored(A, B, n, gamma, ridge, prior)
    Y_onehot = onehot(y, K)
    preds = np.zeros(len(y), dtype=int)
    k = max(1, reanchor_every)
    for i in range(len(y)):
        preds[i] = int(np.argmax(X[i] @ W))
        A = alpha * A + np.outer(X[i], X[i])
        B = alpha * B + np.outer(X[i], Y_onehot[i])
        n = alpha * n + 1.0
        if (i + 1) % k == 0:
            W = _anchored(A, B, n, gamma, ridge, prior)
    return preds
