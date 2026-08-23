"""
Compare denoising methods on a labeled eCallisto spectrogram.

Methods:
- No filtering (raw)
- Adaptive Gaussian background subtraction
- 2D median filtering

Metrics:
- Global signal-to-noise ratio (SNR) in dB, using labeled burst intervals
  as signal and the rest as noise.
- Per-channel SNR, to check whether the global metric is diluting bursts
  that are strong in only a subset of frequency channels.

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

from upstream_utils.compute_snr import (
    compute_snr,
    compute_snr_per_channel,
    summarize_per_channel_snr,
)
from upstream_utils.adaptive_gaussian import gaussian_background_subtract
from upstream_utils.median_filtering import median_denoise


def report(name, snr_db, signal_mean, noise_mean, per_channel_summary):
    """Print a consistent block of global + per-channel SNR stats for one method."""
    print(f"\n=== {name} ===")
    print(f"Signal mean       : {signal_mean:.3f}")
    print(f"Noise  mean       : {noise_mean:.3f}")
    print(f"Global SNR        : {snr_db:.2f} dB")
    print(f"Per-channel mean  : {per_channel_summary['mean']:.2f} dB")
    print(f"Per-channel median: {per_channel_summary['median']:.2f} dB")
    print(f"Per-channel max   : {per_channel_summary['max']:.2f} dB "
          f"({per_channel_summary['n_valid_channels']}/{per_channel_summary['n_total_channels']} valid channels)")

    if per_channel_summary["max"] > snr_db:
        print(f"  -> Strongest channel ({per_channel_summary['max']:.2f} dB) exceeds "
              f"the global metric ({snr_db:.2f} dB): global averaging is diluting "
              f"burst strength for this method.")


def main(spec_path: str, labels_path: str) -> None:
    spectrogram = np.load(spec_path)
    burst_labels = np.load(labels_path, allow_pickle=True)

    print(f"Loaded spectrogram {spectrogram.shape} from {spec_path}")
    print(f"Loaded {len(burst_labels)} burst label entries from {labels_path}")

    # ---------- RAW ----------
    raw_snr_db, raw_signal, raw_noise = compute_snr(spectrogram, burst_labels)
    raw_per_channel = compute_snr_per_channel(spectrogram, burst_labels)
    raw_summary = summarize_per_channel_snr(raw_per_channel)
    report("RAW SPECTROGRAM", raw_snr_db, raw_signal, raw_noise, raw_summary)

    # ---------- ADAPTIVE GAUSSIAN ----------
    gauss_spec, background = gaussian_background_subtract(
        spectrogram,
        sigma_freq=1.0,
        sigma_time=20.0,
        clip_min=0.0,
    )
    gauss_snr_db, gauss_signal, gauss_noise = compute_snr(gauss_spec, burst_labels)
    gauss_per_channel = compute_snr_per_channel(gauss_spec, burst_labels)
    gauss_summary = summarize_per_channel_snr(gauss_per_channel)
    report("GAUSSIAN BACKGROUND SUBTRACTION", gauss_snr_db, gauss_signal, gauss_noise, gauss_summary)

    # ---------- MEDIAN FILTER ----------
    median_spec = median_denoise(
        spectrogram,
        size_freq=3,
        size_time=3,
    )
    median_snr_db, median_signal, median_noise = compute_snr(median_spec, burst_labels)
    median_per_channel = compute_snr_per_channel(median_spec, burst_labels)
    median_summary = summarize_per_channel_snr(median_per_channel)
    report("MEDIAN FILTER", median_snr_db, median_signal, median_noise, median_summary)

    # ---------- Summary table ----------
    print("\n=== GLOBAL SNR COMPARISON (dB) ===")
    print(f"Raw      : {raw_snr_db:.2f}")
    print(f"Gaussian : {gauss_snr_db:.2f}")
    print(f"Median   : {median_snr_db:.2f}")

    print("\n=== PER-CHANNEL MEAN SNR COMPARISON (dB) ===")
    print(f"Raw      : {raw_summary['mean']:.2f}")
    print(f"Gaussian : {gauss_summary['mean']:.2f}")
    print(f"Median   : {median_summary['mean']:.2f}")

    # ---------- Visualization for demo ----------
    time_axis = np.arange(spectrogram.shape[1])
    raw_flux = spectrogram.mean(axis=0)
    gauss_flux = gauss_spec.mean(axis=0)
    median_flux = median_spec.mean(axis=0)

    def overlay_bursts(ax):
        for entry in burst_labels:
            ax.axvspan(entry["start_idx"], entry["end_idx"],
                       color="red", alpha=0.15)

    # --- Figure 1: flux time series comparison (unchanged from original) ---
    fig1 = plt.figure(figsize=(14, 9))

    ax1 = plt.subplot(3, 1, 1)
    ax1.set_title(f"Raw (Global SNR = {raw_snr_db:.2f} dB)")
    ax1.plot(time_axis, raw_flux)
    overlay_bursts(ax1)
    ax1.set_ylabel("Flux")

    ax2 = plt.subplot(3, 1, 2)
    ax2.set_title(f"Adaptive Gaussian (Global SNR = {gauss_snr_db:.2f} dB)")
    ax2.plot(time_axis, gauss_flux)
    overlay_bursts(ax2)
    ax2.set_ylabel("Flux")

    ax3 = plt.subplot(3, 1, 3)
    ax3.set_title(f"Median filter (Global SNR = {median_snr_db:.2f} dB)")
    ax3.plot(time_axis, median_flux)
    overlay_bursts(ax3)
    ax3.set_ylabel("Flux")
    ax3.set_xlabel("Time index")

    plt.tight_layout()

    # --- Figure 2: per-channel SNR comparison across methods ---
    fig2, ax4 = plt.subplots(figsize=(12, 5))
    ax4.plot(raw_per_channel, label=f"Raw (global {raw_snr_db:.2f} dB)", alpha=0.8)
    ax4.plot(gauss_per_channel, label=f"Gaussian (global {gauss_snr_db:.2f} dB)", alpha=0.8)
    ax4.plot(median_per_channel, label=f"Median (global {median_snr_db:.2f} dB)", alpha=0.8)
    ax4.set_xlabel("Frequency channel")
    ax4.set_ylabel("SNR (dB)")
    ax4.set_title("Per-channel SNR by denoising method")
    ax4.legend()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/compare_filters.py <spectrogram_file> <labels_file>")
        sys.exit(1)

    spec_path = sys.argv[1]
    labels_path = sys.argv[2]
    main(spec_path, labels_path)