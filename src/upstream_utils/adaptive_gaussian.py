"""
Adaptive Gaussian background subtraction for eCallisto spectrograms.

Idea:
- Solar radio bursts are localized in time/frequency.
- The quiet-Sun background + slow instrumental drift is smooth.
- A wide 2D Gaussian blur approximates the smooth background.
- Subtracting this background enhances burst contrast.

"""

import numpy as np
from scipy.ndimage import gaussian_filter

def gaussian_background_subtract(
        spectrogram: np.ndarray,
        sigma_freq: float = 1.0,
        sigma_time: float = 20.0,
        clip_min: float | None = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    
    """
    Estimate and subtract a smooth background using a 2D Gaussian filter.

    Parameters
    ----------
    spectrogram : np.ndarray
        2D array (n_freqs, n_times) of intensities.
    sigma_freq : float
        Gaussian std dev in pixels along the frequency axis.
        Larger -> smoother background in frequency.
    sigma_time : float
        Gaussian std dev in pixels along the time axis.
        Larger -> smoother background in time (more aggressive).
    clip_min : float or None
        If not None, restrict values below clip_min to clip_min after subtraction.
        For radio data, clip_min=0.0 is a sensible default.

    Returns
    -------
    cleaned : np.ndarray
        Background-subtracted spectrogram.
    background : np.ndarray
        Estimated smooth background component.
    """

    background = gaussian_filter(
        spectrogram,
        sigma=(sigma_freq, sigma_time),
        mode="nearest",
    )

    cleaned = spectrogram - background

    if clip_min is not None:
        cleaned = np.maximum(cleaned, clip_min)

    return cleaned, background