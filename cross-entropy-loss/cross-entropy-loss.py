import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    # Write code here

    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    n = y_true.shape[0]

    corr_class_prob = y_pred[np.arange(n), y_true]
    
    loss = -np.mean(np.log(corr_class_prob))

    return loss