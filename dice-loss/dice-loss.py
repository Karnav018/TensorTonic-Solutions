import numpy as np

def dice_loss(p, y, eps=1e-8):
    """
    Compute Dice Loss for segmentation.
    """
    # Write code here
    p = np.asarray(p).astype(np.float64).flatten()
    y = np.asarray(y).astype(np.float64).flatten()

    intersection = np.sum(p * y)
    
    sum_p = np.sum(p)
    sum_y = np.sum(y)

    dice_coffe = (2 * intersection + eps) / (sum_p + sum_y + eps)


    dice_loss = 1 - dice_coffe
    
    return dice_loss