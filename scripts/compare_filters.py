"""
Compare denoising methods on a labeled eCallisto spectrogram.

Methods:
- No filtering (raw)
- Adaptive Gaussian background subtraction
- 2D median filtering

Metric:
- Global signal-to-noise ratio (SNR) in dB, using labeled burst intervals
  as signal and the rest as noise.

Usage
-----
From repo root:

    python scripts/compare_filters.py \
        data/spec-ALASKA-ANCHORAGE-05-13-2025.npy \
        data/labels-ALASKA-ANCHORAGE-05-13-2025.npy
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("src")

from upstream_utils.compute_snr import compute_snr
from upstream_utils.adaptive_gaussian import gaussian_background_subtract
from upstream_utils.median_filtering import median_denoise

def main(spec_path: str, labels_path: str) -> None:
    spectrogram = np.load(spec_path)
    burst_labels = np.load(labels_path, allow_pickle=True)

    print(f"Loaded spectrogram {spectrogram.shape} from {spec_path}")
    print(f"Loaded {len(burst_labels)} burst label entries from {labels_path}")


    # ---------- RAW ----------
    raw_snr_db, raw_signal, raw_noise = compute_snr(spectrogram, burst_labels)
    print("\n=== RAW SPECTROGRAM ===")
    print(f"Signal mean: {raw_signal:.3f}")
    print(f"Noise  mean: {raw_noise:.3f}")
    print(f"SNR        : {raw_snr_db:.2f} dB")

    # ---------- ADAPTIVE GAUSSIAN ----------
    gauss_spec, background = gaussian_background_subtract(
        spectrogram,
        sigma_freq=1.0,  
        sigma_time=20.0,
        clip_min=0.0,
    )
    gauss_snr_db, gauss_signal, gauss_noise = compute_snr(gauss_spec, burst_labels)
    print("\n=== GAUSSIAN BACKGROUND SUBTRACTION ===")
    print(f"Signal mean: {gauss_signal:.3f}")
    print(f"Noise  mean: {gauss_noise:.3f}")
    print(f"SNR        : {gauss_snr_db:.2f} dB")

    # ---------- MEDIAN FILTER ----------
    median_spec = median_denoise(
        spectrogram,
        size_freq=3,
        size_time=3,
    )
    median_snr_db, median_signal, median_noise = compute_snr(median_spec, burst_labels)
    print("\n=== MEDIAN FILTER ===")
    print(f"Signal mean: {median_signal:.3f}")
    print(f"Noise  mean: {median_noise:.3f}")
    print(f"SNR        : {median_snr_db:.2f} dB")

    # ---------- Summary table ----------
    print("\n=== SNR COMPARISON (dB) ===")
    print(f"Raw      : {raw_snr_db:.2f}")
    print(f"Gaussian : {gauss_snr_db:.2f}")
    print(f"Median   : {median_snr_db:.2f}")

    # ---------- Visualization for demo ----------
    time_axis = np.arange(spectrogram.shape[1])
    raw_flux = spectrogram.mean(axis=0)
    gauss_flux = gauss_spec.mean(axis=0)
    median_flux = median_spec.mean(axis=0)

    plt.figure(figsize=(14, 9))

    def overlay_bursts(ax):
        for entry in burst_labels:
            ax.axvspan(entry["start_idx"], entry["end_idx"],
                       color="red", alpha=0.15)

    ax1 = plt.subplot(3, 1, 1)
    ax1.set_title(f"Raw (SNR = {raw_snr_db:.2f} dB)")
    ax1.plot(time_axis, raw_flux)
    overlay_bursts(ax1)
    ax1.set_ylabel("Flux")

    ax2 = plt.subplot(3, 1, 2)
    ax2.set_title(f"Adaptive Gaussian (SNR = {gauss_snr_db:.2f} dB)")
    ax2.plot(time_axis, gauss_flux)
    overlay_bursts(ax2)
    ax2.set_ylabel("Flux")

    ax3 = plt.subplot(3, 1, 3)
    ax3.set_title(f"Median filter (SNR = {median_snr_db:.2f} dB)")
    ax3.plot(time_axis, median_flux)
    overlay_bursts(ax3)
    ax3.set_ylabel("Flux")
    ax3.set_xlabel("Time index")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/compare_filters.py <spectrogram_file> <labels_file>")
        sys.exit(1)

    spec_path = sys.argv[1]
    labels_path = sys.argv[2]
    main(spec_path, labels_path)