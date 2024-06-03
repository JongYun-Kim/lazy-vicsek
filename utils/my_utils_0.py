import numpy as np

def wrap_to_pi(angles):
    """
    Wraps *angles* to **[-pi, pi]**
    """
    return (angles + np.pi) % (2 * np.pi) - np.pi