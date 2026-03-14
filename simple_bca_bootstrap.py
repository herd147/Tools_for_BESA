import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.stats import bootstrap


def apply_bandpass_filter(data, lowcut=1, highcut=30, fs=1000, order=2):
    """
    Applies a Butterworth bandpass filter to the data.

    Parameters:
    - data: np.array, shape (n_samples, n_channels, n_subjects)
    - lowcut/highcut: Frequency boundaries in Hz
    - fs: Sampling rate in Hz
    - order: Polynomial order of the filter
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    
    # Filter along axis 0 (time samples)
    filtered = filtfilt(b, a, data, axis=0)
    return filtered


# --- Configuration ---
SAMPLING_RATE = 1000
N_RESAMPLES = 1000
LOW_FREQ = 1
HIGH_FREQ = 30

# Find all .npy files in the current directory
data_files = glob.glob("*.npy")

# --- Processing Loop ---
for file_path in data_files:
    # Skip files that are already bootstrap results to avoid infinite loops
    if "_bootstrap" in file_path:
        continue

    print(f"Processing: {file_path}")
    data = np.load(file_path)  # Expected shape: (samples, channels, subjects)
    
    # 1. Filtering
    print(f"Applying {LOW_FREQ}-{HIGH_FREQ} Hz bandpass filter...")
    data_filtered = apply_bandpass_filter(
        data, lowcut=LOW_FREQ, highcut=HIGH_FREQ, fs=SAMPLING_RATE
    )
    
    # 2. Hemisphere Averaging
    # Reducing from (samples, 2, subjects) to (samples, subjects)
    print("Averaging across hemispheres...")
    data_mean = data_filtered.mean(axis=1) 
    
    # 3. Bootstrap Resampling
    # We bootstrap across subjects (axis 1) to get the distribution of the mean
    print(f"Running bootstrap with {N_RESAMPLES} resamples...")
    res = bootstrap(
        (data_mean,), 
        np.mean, 
        axis=1, 
        n_resamples=N_RESAMPLES, 
        method='BCa'
    )

    # 4. Save Results
    output_name = file_path.replace(".npy", "_bootstrap.npy")
    np.save(output_name, res.bootstrap_distribution)

    # 5. Visualization for Quality Control
    plt.figure(figsize=(10, 6))
    plt.title(f"Bootstrap Samples: {file_path}")
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")

    # Plot a subset (e.g., 50) of the bootstrap distributions for visual check
    # Note: bootstrap_distribution shape is typically (n_resamples, n_samples)
    subset_dist = res.bootstrap_distribution[:50, :]
    
    for sample in subset_dist:
        plt.plot(sample, color='gray', alpha=0.1)
    
    # Plot the grand mean in a distinct color
    plt.plot(np.mean(subset_dist, axis=0), color='red', linewidth=1, label='Grand Mean')
    
    plt.legend()
    plt.show()

    print(f"Original shape: {data_mean.shape}")
    print(f"Bootstrap distribution shape: {res.bootstrap_distribution.shape}")
    print("-" * 30)
