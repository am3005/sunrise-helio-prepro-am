"""
Median filtering for eCallisto spectrograms.

Idea:
- Many RFI artifacts show up as bright, isolated pixels or tiny blobs.
- A median filter replaces each pixel with the median of a local window.
- This preserves edges better than a mean filter while suppressing outliers.

"""

import numpy as np
from scipy.ndimage import median_filter

def median_denoise(
        spectrogram: np.ndarray,
        size_freq: int = 3,
        size_time: int = 3,
) -> np.ndarray:
    
    """
    Apply a 2D median filter to suppress impulsive noise.

    Parameters
    ----------
    spectrogram : np.ndarray
        2D array (n_freqs, n_times) of intensities.
    size_freq : int
        Window size along frequency axis (odd integer).
    size_time : int
        Window size along time axis (odd integer).

    Returns
    -------
    filtered : np.ndarray
        Median-filtered spectrogram.
    """

    size = (size_freq, size_time)
    filtered = median_filter(
        spectrogram,
        size=size,
        mode="nearest",
    )
    return filtered