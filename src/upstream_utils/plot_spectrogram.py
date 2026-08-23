"""
This file is adapted from the Kasper-Heliophysics-MDP/Prepro-F25 repository.
Original implementation: Fall 2025 Kasper Heliophysics MDP cohort (Aashi Mishra listed as coauthor).

Adapted & refactored for personal preprocessing project work presentation:
Aashi Mishra
"""

import sys
import matplotlib.pyplot as plt
import numpy as np


def plot_spectrogram(big_array: np.ndarray, label_file=None, cmap="viridis"):
    """Plot a spectrogram: frequency (y) vs. time (x), intensity in color."""
    plt.figure(figsize=(12, 6))
    plt.imshow(
        big_array,
        aspect="auto",
        origin="lower",
        cmap=cmap
    )
    plt.colorbar(label="Intensity")
    plt.xlabel("Time index")
    plt.ylabel("Frequency bin")
    plt.title("Spectrogram")

    # optionally plot vertical lines at the burst start/end
    if label_file is not None:
        burst_labels = np.load(label_file, allow_pickle=True)
        print(burst_labels)
        for entry in burst_labels:
            start = entry["start_idx"]
            end = entry["end_idx"]
            plt.axvline(start, color="red", linestyle="--", alpha=0.7)
            plt.axvline(end, color="red", linestyle="--", alpha=0.7)

    plt.show()


def plot_spectrogram_comparison(raw, cleaned, method_name, label_file=None, cmap="viridis"):
    """
    Side-by-side comparison of a raw spectrogram vs. a cleaned/filtered version.
    Useful for visually demonstrating the effect of a denoising method.

    Parameters
    ----------
    raw : np.ndarray
        2D array (n_freqs, n_times) of the original, unfiltered spectrogram.
    cleaned : np.ndarray
        2D array (n_freqs, n_times) of the spectrogram after filtering.
    method_name : str
        Name of the filtering method applied, used in the plot title
        (e.g. "Adaptive Gaussian Background Subtraction").
    label_file : str, optional
        Path to a .npy file of burst labels; if provided, burst start/end
        indices are overlaid as vertical lines on both panels.
    cmap : str
        Matplotlib colormap to use for both panels.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure, in case the caller wants to save it directly
        (e.g. fig.savefig(...)) rather than only display it.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    im1 = ax1.imshow(raw, aspect="auto", origin="lower", cmap=cmap)
    ax1.set_title("Raw")
    ax1.set_xlabel("Time index")
    ax1.set_ylabel("Frequency bin")
    fig.colorbar(im1, ax=ax1, label="Intensity")

    im2 = ax2.imshow(cleaned, aspect="auto", origin="lower", cmap=cmap)
    ax2.set_title(f"After {method_name}")
    ax2.set_xlabel("Time index")
    fig.colorbar(im2, ax=ax2, label="Intensity")

    if label_file is not None:
        burst_labels = np.load(label_file, allow_pickle=True)
        for entry in burst_labels:
            for ax in (ax1, ax2):
                ax.axvline(entry["start_idx"], color="red", linestyle="--", alpha=0.7)
                ax.axvline(entry["end_idx"], color="red", linestyle="--", alpha=0.7)

    fig.suptitle(f"Spectrogram before/after {method_name}")
    plt.tight_layout()
    plt.show()
    return fig


if __name__ == "__main__":
    if len(sys.argv) != 2 and len(sys.argv) != 3:
        print("Usage: python plot_spectrogram.py <spectrogram_file_path> [labels_file_path]")
        sys.exit(1)

    spec_file_path = sys.argv[1]

    label_file_path = None
    if len(sys.argv) == 3:
        label_file_path = sys.argv[2]

    data = np.load(spec_file_path)
    plot_spectrogram(data, label_file_path)
    
    