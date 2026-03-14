import numpy as np
import matplotlib.pyplot as plt

# --- 1. Configuration ---
# Update this path to your specific bootstrap result file
FILE_PATH = 'huggins_final_matrix_bootstrap.npy'
N_SAMPLES_TO_PLOT = 100
START_INDEX = 250  # Matches your previous plotting offset (-250ms)

# --- 2. Data Loading ---
try:
    # Expected shape: (timepoints, resamples)
    data = np.load(FILE_PATH)
except FileNotFoundError:
    print(f"Error: The file '{FILE_PATH}' was not found.")
    data = None

if data is not None:
    # 3. Sampling Logic
    # Select 100 random indices from the resamples (axis 1)
    n_resamples = data.shape[1]
    selected_indices = np.random.choice(
        n_resamples, 
        min(N_SAMPLES_TO_PLOT, n_resamples), 
        replace=False
    )
    subset = data[START_INDEX:, selected_indices]
    
    # 4. Statistical Calculations
    mean_curve = data[START_INDEX:, :].mean(axis=1)
    std_curve = data[START_INDEX:, :].std(axis=1)

    # 5. Time Axis Synchronization
    # Starting at -250ms to match the rest of your pipeline
    num_points = len(mean_curve)
    time_axis = np.linspace(-250, -250 + num_points - 1, num_points)

    # 6. Visualization
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot individual samples (thin and transparent)
    ax.plot(time_axis, subset, color='gray', linewidth=0.5, alpha=0.2)

    # Plot the mean (bold line)
    ax.plot(time_axis, mean_curve, color='red', linewidth=2.5, 
            label=f'Grand Mean (n={n_resamples})')

    # Plot the standard deviation cloud
    ax.fill_between(time_axis, mean_curve - std_curve, mean_curve + std_curve, 
                    color='red', alpha=0.1, label='Standard Deviation (1 $\sigma$)')

    # Formatting
    ax.set_title(f'Quality Control: Bootstrap Resamples and Mean ({FILE_PATH})', 
                 fontsize=14)
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Dipole Moment (nAm)', fontsize=12)
    ax.axhline(0, color='black', linewidth=1, linestyle='--')
    ax.axvline(0, color='black', linewidth=1)
    
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.show()
