import numpy as np

def adagrad_step(w: list, g: list, G: list, lr: float = 0.01, eps: float = 1e-8) -> dict:
    """
    Returns a dictionary with new_w and new_G.
    """
    # Write code here
    w = np.array(w)
    g = np.array(g)
    G = np.array(G)
    G = G + g ** 2
    w = w - g * lr / ( np.sqrt(G + eps)  ) 
    return {
         'new_w': w, 
        'new_G': G
    }
    
    