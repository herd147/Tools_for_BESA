import numpy as np
import matplotlib.pyplot as plt

# --- 1. Global Configuration ---
# Starting index (Index 250 corresponds to -250ms pre-stimulus)
START_INDEX = 250
LINE_WIDTH = 2
ALPHA_FILL = 0.3
COLORS = ['coral', 'skyblue']
FILL_COLORS = ['lightcoral', 'lightblue']
LABELS = ["FRDA (n=14)", "Controls (n=14)"]

# File groupings for analysis
groups = {
    "Huggins": [
        "_bootstrap.npy", 
        "_bootstrap.npy"
    ],
    "IRN": [
        "_bootstrap.npy", 
        "bootstrap.npy"
    ],
    "Rho": [
        "_bootstrap.npy",
        "_bootstrap.npy"]
}

# --- 2. Plotting Logic ---
for title, files in groups.items():
    plt.figure(figsize=(14, 6))
    
    for idx, fname in enumerate(files):
        # Load bootstrap resamples
        try:
            # Data shape: (timepoints, resamples)
            data = np.load(fname)
        except FileNotFoundError:
            print(f"File not found: {fname}")
            continue
        
        # Calculate statistics across resamples
        mean_val = np.mean(data, axis=1)
        lower_ci = np.percentile(data, 2.5, axis=1)
        upper_ci = np.percentile(data, 97.5, axis=1)
        
        # Slicing: Keep data from the START_INDEX onwards
        mean_val = mean_val[START_INDEX:]
        lower_ci = lower_ci[START_INDEX:]
        upper_ci = upper_ci[START_INDEX:]
        
        # Synchronize Time Axis
        # Since index 500 is 0ms, our start at index 250 is exactly -250ms.
        num_points = len(mean_val)
        time_axis = np.linspace(-250, -250 + num_points - 1, num_points)
        
        # Determine Plot Limits and Annotations based on Condition
        if "Rho" in title:
            x_limit = (-250, 2250)
            stim_regions = [(0, 750, r'$\rho$ 1'), (750, 750, r'$\rho$ 2')]
        else:
            x_limit = (-250, 2750)
            stim_regions = [
                (0, 750, 'noise'), 
                (750, 750, 'pitch 1'), 
                (1500, 750, 'pitch 2')
            ]

        # Draw stimulus period bars once per figure
        if idx == 0:
            for start, duration, label in stim_regions:
                # Calculate center of the bar for text placement
                center = start + (duration / 2)
                plt.barh(-45, duration, height=3, left=start, 
                         color='gray', alpha=0.5, edgecolor='none')
                plt.text(center, -45, label, color='black', 
                         ha='center', va='center', fontweight='bold')

        # Plot Mean and 95% Confidence Interval
        plt.plot(time_axis, mean_val, color=COLORS[idx], 
                 linewidth=LINE_WIDTH, label=f"{LABELS[idx]} Mean")
        
        plt.fill_between(time_axis, lower_ci, upper_ci, 
                         color=FILL_COLORS[idx], alpha=ALPHA_FILL, 
                         label=f"{LABELS[idx]} 95% CI")

        plt.xlim(x_limit)
        plt.ylim(-55, 55)

    # Final Styling per Figure
    plt.title(f"Bootstrap 95% Confidence Intervals: {title}", fontsize=16)
    plt.xlabel("Time (ms)", fontsize=12)
    plt.ylabel("Dipole Moment (nAm)", fontsize=12)
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')  # Baseline
    plt.axvline(0, color='black', linewidth=0.8)                 # Stimulus onset
    
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right', fontsize=10, frameon=True)
    plt.tight_layout()
    
    # Save option (uncomment to use)
    # plt.savefig(f"bootstrap_plot_{title.lower()}.png", dpi=300)
    plt.show()
