import numpy as np

def kl_divergence(p, q, eps=1e-12):
    """
    Compute KL Divergence D_KL(P || Q).
    """
    # Write code here
    p = np.array(p)
    q = np.array(q)
    q_safe = q + eps
    mask = (p > 0)
    p_act = p[mask]
    q_act = q_safe[mask]
    ratio = p_act / q_act
    log_ratio = np.log(ratio)
    elementwise = p_act * log_ratio
    total = np.sum(elementwise)
    return total
    
    
    pass