import numpy as np
import matplotlib.pyplot as plt
import sys


def compute_snr(spectrogram, burst_labels):
    """
    Compute global SNR of a spectrogram given burst index ranges, collapsing
    across all frequency channels before comparing burst vs. non-burst regions.

    Parameters
    ----------
    spectrogram : np.ndarray
        2D array (n_freqs, n_times) of intensities.
    burst_labels : list of dicts
        Each dict has {"burst": str, "start_idx": int, "end_idx": int}.

    Returns
    -------
    snr_db : float
        Signal-to-noise ratio in dB.
    signal_mean : float
        Average signal level.
    noise_mean : float
        Average noise level.

    Notes
    -----
    Averaging over all frequency channels before computing SNR can dilute
    narrowband bursts that only appear strongly in a subset of channels.
    See `compute_snr_per_channel` for a per-channel alternative.
    """
    n_times = spectrogram.shape[1]
    flux_time = spectrogram.mean(axis=0)  # collapse freqs → flux vs time

    # Mask for burst times
    burst_mask = np.zeros(n_times, dtype=bool)
    for entry in burst_labels:
        start = max(0, entry["start_idx"])
        end = min(n_times - 1, entry["end_idx"])
        burst_mask[start:end + 1] = True

    inside_flux = flux_time[burst_mask]
    outside_flux = flux_time[~burst_mask]

    signal_mean = inside_flux.mean() if inside_flux.size > 0 else np.nan
    noise_mean = outside_flux.mean() if outside_flux.size > 0 else np.nan

    if signal_mean is None or noise_mean is None or np.isnan(signal_mean) or np.isnan(noise_mean):
        snr_db = np.nan
    elif noise_mean <= 0:
        # Negative or zero noise_mean can occur after background subtraction;
        # log10 of a non-positive value is undefined. Flag explicitly rather
        # than silently returning nan with no indication of why.
        print(f"Warning: non-positive noise_mean ({noise_mean:.4f}), SNR undefined for this segment")
        snr_db = np.nan
    else:
        snr_db = 10 * np.log10(signal_mean / noise_mean)

    return snr_db, signal_mean, noise_mean


def compute_snr_per_channel(spectrogram, burst_labels):
    """
    Compute SNR independently for each frequency channel, rather than
    collapsing to 1D first, to avoid diluting narrowband bursts that don't
    span the full frequency range.

    Parameters
    ----------
    spectrogram : np.ndarray
        2D array (n_freqs, n_times) of intensities.
    burst_labels : list of dicts
        Each dict has {"burst": str, "start_idx": int, "end_idx": int}.

    Returns
    -------
    snr_per_channel : np.ndarray
        1D array of length n_freqs; SNR in dB for each frequency channel.
        Entries are np.nan where a channel has no valid burst/non-burst
        samples, or where noise_mean is non-positive.
    """
    n_freqs, n_times = spectrogram.shape

    burst_mask = np.zeros(n_times, dtype=bool)
    for entry in burst_labels:
        start = max(0, entry["start_idx"])
        end = min(n_times - 1, entry["end_idx"])
        burst_mask[start:end + 1] = True

    snr_per_channel = np.full(n_freqs, np.nan)
    for ch in range(n_freqs):
        inside = spectrogram[ch, burst_mask]
        outside = spectrogram[ch, ~burst_mask]
        if inside.size == 0 or outside.size == 0:
            continue

        signal_mean, noise_mean = inside.mean(), outside.mean()
        if noise_mean > 0:
            snr_per_channel[ch] = 10 * np.log10(signal_mean / noise_mean)
        # noise_mean <= 0 left as np.nan intentionally, same reasoning as compute_snr

    return snr_per_channel


def summarize_per_channel_snr(snr_per_channel):
    """
    Summarize a per-channel SNR array, ignoring nan entries.

    Returns
    -------
    dict with keys: mean, median, max, n_valid_channels, n_total_channels
    """
    valid = snr_per_channel[~np.isnan(snr_per_channel)]
    return {
        "mean": valid.mean() if valid.size > 0 else np.nan,
        "median": np.median(valid) if valid.size > 0 else np.nan,
        "max": valid.max() if valid.size > 0 else np.nan,
        "n_valid_channels": valid.size,
        "n_total_channels": snr_per_channel.size,
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compute_snr.py <spectrogram_file_path> <labels_file_path>")
        sys.exit(1)

    # Example: load data
    big_array = np.load(sys.argv[1])
    burst_labels = np.load(sys.argv[2], allow_pickle=True)

    # ---------- Global SNR ----------
    snr_db, signal, noise = compute_snr(big_array, burst_labels)
    print("=== GLOBAL SNR (collapsed across frequency) ===")
    print(f"Signal mean: {signal:.3f}")
    print(f"Noise mean : {noise:.3f}")
    print(f"SNR        : {snr_db:.2f} dB")

    # ---------- Per-channel SNR ----------
    snr_per_channel = compute_snr_per_channel(big_array, burst_labels)
    summary = summarize_per_channel_snr(snr_per_channel)
    print("\n=== PER-CHANNEL SNR ===")
    print(f"Mean SNR across valid channels  : {summary['mean']:.2f} dB")
    print(f"Median SNR across valid channels: {summary['median']:.2f} dB")
    print(f"Max SNR (strongest channel)     : {summary['max']:.2f} dB")
    print(f"Valid channels: {summary['n_valid_channels']} / {summary['n_total_channels']}")

    if summary["max"] > snr_db:
        print(f"\nStrongest individual channel ({summary['max']:.2f} dB) exceeds the "
              f"global SNR ({snr_db:.2f} dB) — evidence the global metric is diluting "
              f"burst strength for at least some channels.")

    # ---------- Visualization ----------
    flux_time = big_array.mean(axis=0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(flux_time, label="Flux (mean over freqs)", color="blue")
    for entry in burst_labels:
        ax1.axvspan(entry["start_idx"], entry["end_idx"], color="red", alpha=0.2)
    ax1.set_xlabel("Time index")
    ax1.set_ylabel("Flux")
    ax1.set_title(f"Flux time series with bursts (Global SNR = {snr_db:.2f} dB)")
    ax1.legend()

    ax2.plot(snr_per_channel, color="darkorange")
    ax2.axhline(snr_db, color="blue", linestyle="--", label=f"Global SNR ({snr_db:.2f} dB)")
    ax2.set_xlabel("Frequency channel")
    ax2.set_ylabel("SNR (dB)")
    ax2.set_title("Per-channel SNR vs. global SNR")
    ax2.legend()

    plt.tight_layout()
    plt.show()