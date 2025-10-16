import numpy as np

def smooth(y: np.ndarray, f: float = 0.05) -> np.ndarray:
    """Box filter of fraction f."""
    # Kernel size or the number of filter elements (must be odd)
    nf = round(len(y) * f * 2) // 2 + 1  # Note: a//b = floor(a/b)
    p = np.ones(nf // 2)  #  padding with size nf//2
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # pad y by its left most and right most elements
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")  # y-smoothed