import numpy as np

def focal_loss(p, y, gamma=2.0):
    """
    Compute Focal Loss for binary classification.
    """
    # Write code here

    p = np.clip(np.asarray(p), 1e-15, 1-1e-15)
    y = np.asarray(y)

    sec_term = y * ((1 - p)**gamma) * np.log(p)

    first_term = p**gamma * (1 - y) * np.log(1-p) 

    fl = - (first_term + sec_term)

    return np.mean(fl)

    
    
    pass