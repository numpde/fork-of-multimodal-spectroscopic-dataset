

import numpy as np

def attention(Q: np.array, K: np.array, V: np.array):
    """
    Query Q: Matrix of shape (t, dk)
    Key K: Matrix of shape (t, dk)
    Value V: Matrix of shape (t, dv)

    Here, `t` is the "sequence length" and `dk` is the "key dimension".
    """

    assert Q.shape == K.shape
    assert Q.shape[-1] == V.shape[-1]

    # Get the sequence length and key dimension
    (t, dk) = Q.shape

    # Attention scores of shape (t, t)
    S = (Q @ K.T) / np.sqrt(dk)

    # Attention weights
    A = np.exp(S) @ np.diag(1 / np.sum(np.exp(S), axis=1))

    return A @ V

