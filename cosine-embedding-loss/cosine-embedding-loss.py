import numpy as np

def cosine_embedding_loss(x1, x2, label, margin=0.0):
    """
    Compute cosine embedding loss for a pair of vectors.
    """

    x1 = np.array(x1)
    x2 = np.array(x2)

    dot = np.dot(x1,x2)

    nomr_x1 = np.linalg.norm(x1)
    nomr_x2 = np.linalg.norm(x2)

    cos = dot / (nomr_x1 * nomr_x2)

    if label == 1:
        return 1 - cos
    elif label == -1:
        return max(0, cos - margin)
    else:
        raise ValueError("")
