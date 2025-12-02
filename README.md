# sunrise-helio-prepro-am

# one_day.py:

End-to-end daily data ingestion:

Downloads all .fit.gz files for a given station + day.

Decompresses FITS → NumPy arrays.

Chronologically sorts files using circular_sort().

Concatenates all spectrogram chunks into one continuous (freq × time) array.

Optionally aligns official burst labels to time indices.
Outputs: cleaned daily spectrogram .npy + optional label index .npy.

# adaptive_gaussian.py:

Implements adaptive Gaussian background subtraction:

Smooths the spectrogram with a 2D Gaussian (sigma_freq, sigma_time).

Subtracts smooth background → enhances bursts.

Clipping to enforce non-negative output.
Use case: remove slow instrumental drift + quiet Sun background.

# median_filtering.py:

Implements 2D median filtering:

Suppresses impulsive RFI spikes and single-pixel noise.

Preserves burst edges better than Gaussian smoothing.
Use case: remove pixel-level outliers.

# compute_snr.py:

Global SNR evaluator:

Collapses spectrogram → flux vs time (mean over frequencies).

Uses burst label intervals as “signal” and all other times as “noise.”

Computes SNR in decibels.

Optional visualization shading burst regions.
Use case: compare filter effectiveness quantitatively.

# compare_filters.py:

Runs all denoising methods head-to-head:

Raw

Adaptive Gaussian

Median filter
Computes SNR for each + displays comparative plots.
Use case: tune filter parameters and visually validate preprocessing.

