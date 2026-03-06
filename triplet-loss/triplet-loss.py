import numpy as np

def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Compute Triplet Loss for embedding ranking.
    """
    # Write code here
    a = np.array(anchor)
    p = np.array(positive)
    n = np.array(negative)

    d_ap = np.sum(np.square(a - p), axis=-1)
    d_an = np.sum(np.square(a - n), axis =-1)

    losses = np.maximum(0, d_ap - d_an + margin)

    return np.mean(losses)
    