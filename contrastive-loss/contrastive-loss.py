import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean") -> float:
    """
    a, b: arrays of shape (N, D) or (D,)  (will broadcast to (N,D))
    y:    array of shape (N,) with values in {0,1}; 1=similar, 0=dissimilar
    margin: float > 0
    reduction: "mean" (default) or "sum"
    Return: float
    """
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    y = np.asarray(y)

    distance = np.linalg.norm(a-b, axis=1)

    loss_similar = y * np.square(distance)

    loss_dissimilar = (1-y) * np.square(np.maximum(0,margin - distance))

    total = loss_similar + loss_dissimilar

    if reduction == "mean":
        return float(np.mean(total))
    elif reduction == "sum":
        return float(np.sum(total))
    else:
        raise ValueError("Check mean and sum")

