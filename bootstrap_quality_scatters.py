import numpy as np
import matplotlib.pyplot as plt

# Define specific time slices for the bootstrap analysis
POR_irn = slice()
#continue

# --- 1. Window Configuration ---
# Fine-tune the analysis windows for each plot and group
WINDOW_CONFIGS = {
    "Huggins": {
        "POR": {"pat": POR_hp_pt, "ctrl": POR_hp_ctrl, "stim_start": 1250},
        "PCR": {"pat": PCR_hp_pt, "ctrl": PCR_hp_ctrl, "stim_start": 2000}
    },
    "IRN": {
        "POR": {"pat": POR_irn, "ctrl": POR_irn, "stim_start": 1250},
        "PCR": {"pat": PCR_irn, "ctrl": PCR_irn, "stim_start": 2000}
    },
    "Rho": {
        "CR":  {"pat": CR_rho_pt, "ctrl": CR_rho_ctrl, "stim_start": 1250}
    },
    "Rho PM": {
        "CR":  {"pat": CR_rho_pm_pt, "ctrl": CR_rho_pm_ctrl, "stim_start": 1250}
    },
    "Rho MP": {
        "CR":  {"pat": CR_rho_mp_pt, "ctrl": CR_rho_mp_ctrl, "stim_start": 1250}
    }
}

# --- 2. Data Groups ---
FILE_GROUPS = {
    "Huggins": ["_bootstrap.npy", "bootstrap.npy"],
    "IRN": ["bootstrap.npy", "bootstrap.npy"],
    "Rho": ["bootstrap.npy", "bootstrap.npy"],
    "Rho PM": ["bootstrap.npy", "bootstrap.npy"],
    "Rho MP": ["rbootstrap.npy", "_bootstrap.npy"]
}

# --- 3. Visualization ---
for condition, files in FILE_GROUPS.items():
    try:
        boot_pat = np.load(files[0])
        boot_ctrl = np.load(files[1])
    except FileNotFoundError as e:
        print(f"Skipping {condition}: {e}")
        continue

    config = WINDOW_CONFIGS[condition]
    num_regions = len(config)
    
    # Create subplots for each region (e.g., POR and PCR)
    fig, axes = plt.subplots(
        1, num_regions, figsize=(7 * num_regions, 6), squeeze=False
    )
    
    for ax, (region_name, cfg) in zip(axes[0], config.items()):
        stim_offset = cfg["stim_start"]
        
        # --- Process FRDA (Patients) ---
        win_p = cfg["pat"]
        data_p = boot_pat[win_p, :]
        y_min_p = np.min(data_p, axis=0)
        # Convert index to relative latency (ms)
        x_ms_p = (np.argmin(data_p, axis=0) + win_p.start) - stim_offset
        
        # --- Process Controls ---
        win_c = cfg["ctrl"]
        data_c = boot_ctrl[win_c, :]
        y_min_c = np.min(data_c, axis=0)
        x_ms_c = (np.argmin(data_c, axis=0) + win_c.start) - stim_offset
        
        # 4. Create Scatter Plots
        ax.scatter(x_ms_c, y_min_c, alpha=0.4, s=25,
                   label="Control (n=15)", color="skyblue")
        ax.scatter(x_ms_p, y_min_p, alpha=0.4, s=25,
                   label="Patients (n=15)", color="coral")
        
        # 5. Add Mean Vertical Lines for Latency
        mean_p = np.mean(x_ms_p)
        mean_c = np.mean(x_ms_c)
        
        ax.axvline(mean_p, color='coral', linestyle='--', alpha=0.6, 
                   label=f'Mean Pts: {mean_p:.1f}ms')
        ax.axvline(mean_c, color='skyblue', linestyle='--', alpha=0.6, 
                   label=f'Mean Ctrl: {mean_c:.1f}ms')
        
        # Formatting
        ax.set_title(f"{region_name} Stability ({condition})")
        ax.set_xlabel("Latency (ms post-stimulus)")
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(fontsize='small', loc='best')
        
    axes[0, 0].set_ylabel("Min. Amplitude (nAm)")
    plt.suptitle(f"Bootstrap Peak Stability Analysis: {condition}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
