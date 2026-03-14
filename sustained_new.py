import numpy as np
from scipy import stats, signal


def process_sustained_field(data, start_idx, end_idx, fs=1000):
    """
    Processes EEG/MEG data by averaging across hemispheres, 
    applying a low-pass filter, and extracting a specific time window.

    Parameters:
    - data: Input array (timepoints, hemispheres, subjects)
    - start_idx, end_idx: Indices for the time window of interest
    - fs: Sampling rate in Hz (default 1000Hz)
    
    Returns:
    - final_avg: Mean value per subject for the specified window.
    """
    # Step A: Average over hemispheres (axis 1)
    # Resulting shape: (timepoints, subjects)
    data_hemi_avg = np.mean(data, axis=1)

    # --- Filter Design: Low-pass up to 5 Hz ---
    highcut = 5.0
    
    # Create a 4th order Butterworth filter
    sos = signal.butter(4, highcut, btype='low', fs=fs, output='sos')
    
    # Apply zero-phase filter (forward and backward) along the time axis (0)
    filtered_data = signal.sosfiltfilt(sos, data_hemi_avg, axis=0)

    # Step B: Select the time window (Sustained Field)
    window_data = filtered_data[start_idx:end_idx, :]
    
    # Step C: Average across the time window -> One value per subject
    final_avg = np.mean(window_data, axis=0)
    
    return final_avg


# --- 1. Load Data ---
# It is recommended to use relative paths or a dedicated data folder
data_h1 = np.load('.npy')
data_h2 = np.load('.npy')
data_i1 = np.load('.npy')
data_i2 = np.load('.npy')
data_r1 = np.load('.npy')
data_r2 = np.load('.npy')

# --- 2. Define Parameters & Constants ---
SAMPLING_FREQ = 1000

# Time window indices, in ms (adjust based on actual data and sampling rate)
IDX_POR, IDX_POR_END = 1700, 1900
IDX_PCR, IDX_PCR_END = 2250, 2600
IDX_RHO, IDX_RHO_END = 1050, 1250
IDX_RHO_II, IDX_RHO_II_END = 1850, 2000

# --- 3. Analysis Configuration ---
# Defining analysis sets as a list of tuples: (Label, Data1, Data2, Start, End)
analyses = [
    ("Huggins POR", data_h1, data_h2, IDX_POR, IDX_POR_END),
    ("Huggins PCR", data_h1, data_h2, IDX_PCR, IDX_PCR_END),
    ("IRN POR", data_i1, data_i2, IDX_POR, IDX_POR_END),
    ("IRN PCR", data_i1, data_i2, IDX_PCR, IDX_PCR_END),
    ("Rho I", data_r1, data_r2, IDX_RHO, IDX_RHO_END),
    ("Rho II", data_r1, data_r2, IDX_RHO_II, IDX_RHO_II_END)
]

# --- 4. Processing, Statistics & Export ---
print("Starting analysis and saving results...")

with open('sustained_field_results.txt', 'w') as f:
    f.write("SUSTAINED FIELD ANALYSIS RESULTS\n")
    f.write("================================\n\n")

    for label, d1, d2, start, end in analyses:
        # Process groups
        scores1 = process_sustained_field(d1, start, end, fs=SAMPLING_FREQ)
        scores2 = process_sustained_field(d2, start, end, fs=SAMPLING_FREQ)
        
        # Perform Independent t-test
        t_stat, p_val = stats.ttest_ind(scores1, scores2)
        
        # Write results to file
        f.write(f"Condition: {label}\n")
        f.write(f"  Group 1 Mean: {np.mean(scores1):.4f}\n")
        f.write(f"  Group 2 Mean: {np.mean(scores2):.4f}\n")
        f.write(f"  t-statistic: {t_stat:.3f}, p-value: {p_val:.4f}\n\n")

print("Done! Results saved to 'sustained_field_results.txt'.")