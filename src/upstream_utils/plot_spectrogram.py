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

    #optionally plot vertical lines at the burst start/end
    if label_file is not None:
        burst_labels = np.load(label_file, allow_pickle=True)
        print(burst_labels)
        for entry in burst_labels:
            start = entry["start_idx"]
            end = entry["end_idx"]
            plt.axvline(start, color="red", linestyle="--", alpha=0.7)
            plt.axvline(end, color="red", linestyle="--", alpha=0.7)
            
    plt.show()

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
    
    