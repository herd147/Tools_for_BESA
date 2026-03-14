import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Global default line width for plots (slightly thicker)
DEFAULT_LINEWIDTH = 2.1

def plot_mean(np_array, name):
    """
    Function to plot the mean of all subjects in the 3rd dimension (axis 0).

    Parameters:
    np_array (np.ndarray): Data to plot.
    name (str): Name of condition and plot.
    """
    
    # Apply 1-30Hz zero-phase bandpass filter to the data
    # Filtering parameters
    fs = 1000  
    lowcut = 1.0
    highcut = 30.0
    order = 2

    def butter_bandpass(lowcut, highcut, fs, order=2):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def zero_phase_filter_3d(data3d, lowcut, highcut, fs, order=2):
        b, a = butter_bandpass(lowcut, highcut, fs, order)
        filtered = np.empty_like(data3d)
    
        # Iterate over hemispheres and swf files
        for hemi in range(data3d.shape[1]):
            for swf in range(data3d.shape[2]):
                filtered[:, hemi, swf] = filtfilt(b, a, data3d[:, hemi, swf])
    
        return filtered
    
    # Apply the filter
    np_array = zero_phase_filter_3d(np_array, lowcut, highcut, fs, order)
    
    # Calculate the mean across the 3rd dimension (axis 0)
    mean_array = np.mean(np_array, axis=2)

    plt.figure(figsize=(14, 6))
    time_samples = np.linspace(-500,3500, mean_array.shape[0])  # Adjust the x-axis to start from -500ms
    if "_pt" in name:
        plt.plot(time_samples, mean_array[:, 0], color='red', label='left hemisphere', linewidth=DEFAULT_LINEWIDTH)
        plt.plot(time_samples, mean_array[:, 1], color='coral', label='right hemisphere', linewidth=DEFAULT_LINEWIDTH)
    else:
        plt.plot(time_samples, mean_array[:, 0], color='blue', label='left hemisphere', linewidth=DEFAULT_LINEWIDTH)
        plt.plot(time_samples, mean_array[:, 1], color='skyblue', label='right hemisphere', linewidth=DEFAULT_LINEWIDTH)
    plt.legend()
    plt.xlabel("Time (ms)")
    plt.ylabel('dipole moment (nAm)') 
    plt.grid()
    plt.xlim(-50, 3000)

    bar_height = 3

    plt.barh(-60, 750, height=bar_height, left=0, color='gray', alpha=0.7)
    plt.text(375, -60 , 'noise', color='white', ha='center', va='center')

    plt.barh(-60, 750, height=bar_height, left=750, color='darkgray', alpha=0.7)
    plt.text(1125, -60 , 'pitch 1', color='black', ha='center', va='center')

    plt.barh(-60, 750, height=bar_height, left=1500, color='lightgray', alpha=0.7)
    plt.text(1875, -60, 'pitch 2', color='black', ha='center', va='center')

    # Save the plot
    plt.savefig(f'{name}_mean.png')
    plt.show()

# Example usage
# plot_mean(huggins, "Huggins")

def plot_mean_rho(np_array, name):
    """
    Function to plot the mean of all subjects in the 3rd dimension (axis 0) for rho values.

    Parameters:
    np_array (np.ndarray): Data to plot.
    name (str): Name of condition and plot.
    """
    # Calculate the mean across the 3rd dimension (axis 0)
    mean_array = np.mean(np_array, axis=2)

    plt.figure(figsize=(14, 6))
    time_samples = np.linspace(-500, 2500, mean_array.shape[0])  # Adjust the x-axis to start from -500ms
    plt.plot(time_samples, mean_array[:, 0], color='blue', label='left hemisphere')
    plt.plot(time_samples, mean_array[:, 1], color='skyblue', label='right hemisphere', linewidth=DEFAULT_LINEWIDTH)
    plt.plot(time_samples, mean_array[:, 0], color='blue', label='left hemisphere', linewidth=DEFAULT_LINEWIDTH)
    plt.legend()
    plt.xlabel("Time (ms)")
    plt.ylabel('dipole moment (nAm)') 
    plt.grid()
    plt.xlim(-50, 2250)

    bar_height = 3

    plt.barh(-60, 750, height=bar_height, left=0, color='gray', alpha=0.7)
    plt.text(375, -60 , 'rho 1', color='white', ha='center', va='center')

    plt.barh(-60, 750, height=bar_height, left=750, color='darkgray', alpha=0.7)
    plt.text(1125, -60 , 'rho 2', color='black', ha='center', va='center')

    # Save the plot
    plt.savefig(f'{name}_mean_rho.png')
    plt.show()

# Example usage
# plot_mean_rho(rho_data, "Rho")

def plot_individual(np_array, name):
    """
    Function to plot the data for each subject with the mean of all subjects overplotted.

    Parameters:
    np_array (np.ndarray): Data to plot.
    name (str): Name of condition and plot.
    """
    # Plot the data for each subject with the mean of all subjects overplotted
    plt.figure(figsize=(14, 6))
    time_samples = np.linspace(-500,3500, np_array.shape[0])  # Adjust the x-axis to start from -500ms
    for i in range(np_array.shape[2]):
        plt.plot(time_samples, np_array[:, :, i], alpha=0.5, linewidth=1)
    plt.plot(time_samples, np.mean(np_array, axis=2), color='black', linewidth=1)
    plt.xlabel("Time (ms)")
    plt.ylabel('dipole moment (nAm)') 
    plt.grid()
    plt.xlim(-50, 3000)

    bar_height = 3

    plt.barh(-60, 750, height=bar_height, left=0, color='gray', alpha=0.7)
    plt.text(375, -60 , 'noise', color='white', ha='center', va='center')

    plt.barh(-60, 750, height=bar_height, left=750, color='darkgray', alpha=0.7)
    plt.text(1125, -60 , 'pitch 1', color='black', ha='center', va='center')

    plt.barh(-60, 750, height=bar_height, left=1500, color='lightgray', alpha=0.7)
    plt.text(1875, -60, 'pitch 2', color='black', ha='center', va='center')

    
    # Save the plot
    plt.savefig(f'{name}_individual.png')
    plt.show()

# Example usage
# plot_individual(huggins, "Huggins")

def plot_overcontrol(np_array, np_array_ctrl, name_pt, name_ctrl):
    """
    Function to plot the data for the mean of patients over the mean of controls in one plot.

    Parameters:
    np_array (np.ndarray): Data to plot.
    name_pt (str): Name of patient condition.
    name_ctrl (str): Name of control condition.
    """
    # Get the mean of all subjects for patients and controls
    mean_pt = np.mean(np_array, axis=2)
    mean_pt_bothhemi = np.squeeze(np.mean(mean_pt, axis=1))
    mean_ctrl = np.mean(np_array_ctrl, axis=2)
    mean_ctrl_bothhemi = np.squeeze(np.mean(mean_ctrl, axis=1))
    
    # Bandpass zero-phase filter the data
    # Filtering parameters 
    fs = 1000  # Sampling frequency
    lowcut = 1.0  # Low cutoff frequency
    highcut = 30.0  # High cutoff frequency
    order = 2  # Filter order
    def butter_bandpass(lowcut, highcut, fs, order=2):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a
    def zero_phase_filter_1d(data1d, lowcut, highcut, fs, order=2):
        b, a = butter_bandpass(lowcut, highcut, fs, order)
        filtered = filtfilt(b, a, data1d)
        return filtered

    # Apply the filter
    mean_pt_bothhemi = zero_phase_filter_1d(mean_pt_bothhemi, lowcut, highcut, fs, order)
    mean_ctrl_bothhemi = zero_phase_filter_1d(mean_ctrl_bothhemi, lowcut, highcut, fs, order)


    # Plot the same for pooled hemispheres
    plt.figure(figsize=(14, 6))
    time_samples = np.linspace(-500, 3500, mean_pt_bothhemi.shape[0])  # Adjust the x-axis to start from -500ms
    plt.plot(time_samples, mean_pt_bothhemi, color='coral', linewidth=DEFAULT_LINEWIDTH)
    plt.plot(time_samples, mean_ctrl_bothhemi, color='skyblue', linewidth=DEFAULT_LINEWIDTH)
    plt.xlabel("Time (ms)")
    plt.ylabel('dipole moment (nAm)') 
    plt.grid()
    plt.legend([f'FRDA (n=14)', f'controls (n=14)'])
    plt.xlim(-250, 2750)
    plt.ylim(-50, 50)


    bar_height = 3

    plt.barh(-45, 750, height=bar_height, left=0, color='gray', alpha=0.7)
    plt.text(375, -45 , 'noise', color='white', ha='center', va='center')
    plt.barh(-45, 750, height=bar_height, left=750, color='darkgray', alpha=0.7)
    plt.text(1125, -45 , 'pitch 1', color='black', ha='center', va='center')
    plt.barh(-45, 750, height=bar_height, left=1500, color='lightgray', alpha=0.7)
    plt.text(1875, -45, 'pitch 2', color='black', ha='center', va='center')

    
    # Save the plot
    plt.savefig(f'{name_pt}_{name_ctrl}_mean_bothhemi.png')
    plt.show()

# Example usage
# plot_overcontrol(huggins, huggins_ctrl, "Huggins", "Huggins Controls")

def plot_overcontrol_unfiltered(np_array, np_array_ctrl, name_pt, name_ctrl):
    """
    Function to plot the data for the mean of patients over the mean of controls in one plot.

    Parameters:
    np_array (np.ndarray): Data to plot.
    name_pt (str): Name of patient condition.
    name_ctrl (str): Name of control condition.
    """
    # Get the mean of all subjects for patients and controls
    mean_pt = np.mean(np_array, axis=2)
    mean_pt_bothhemi = np.squeeze(np.mean(mean_pt, axis=1))
    mean_ctrl = np.mean(np_array_ctrl, axis=2)
    mean_ctrl_bothhemi = np.squeeze(np.mean(mean_ctrl, axis=1))

    
    # Plot the same for pooled hemispheres
    plt.figure(figsize=(14, 6))
    time_samples = np.linspace(-500, 3500, mean_pt_bothhemi.shape[0])  # Adjust the x-axis to start from -500ms
    plt.plot(time_samples, mean_ctrl_bothhemi, color='skyblue', linewidth=DEFAULT_LINEWIDTH)
    plt.plot(time_samples, mean_pt_bothhemi, color='coral', linewidth=DEFAULT_LINEWIDTH)
    plt.xlabel("Time (ms)")
    plt.ylabel('dipole moment (nAm)') 
    plt.grid()
    plt.legend([f'FRDA (n=14)', f'controls (n=14)'])
    plt.xlim(-250, 2750)
    plt.ylim(-50, 50)


    bar_height = 3

    plt.barh(-45, 750, height=bar_height, left=0, color='gray', alpha=0.7)
    plt.text(375, -45 , 'noise', color='white', ha='center', va='center')

    plt.barh(-45, 750, height=bar_height, left=750, color='darkgray', alpha=0.7)
    plt.text(1125, -45 , 'pitch 1', color='black', ha='center', va='center')
    
    plt.barh(-45, 750, height=bar_height, left=1500, color='lightgray', alpha=0.7)
    plt.text(1875, -45, 'pitch 2', color='black', ha='center', va='center')

    
    # Save the plot
    plt.savefig(f'{name_pt}_{name_ctrl}_mean_bothhemi_unfiltered.png')
    plt.show()

# Example usage
# plot_overcontrol(huggins, huggins_ctrl, "Huggins", "Huggins Controls")

def plot_overcontrol_rho_unfiltered(np_array, np_array_ctrl, name_pt, name_ctrl):
    """
    Function to plot the data for the mean of patients over the mean of controls in one plot for rho values.

    Parameters:
    np_array (np.ndarray): Data to plot.
    name_pt (str): Name of patient condition.
    name_ctrl (str): Name of control condition.
    """
   
    # Get the mean of all subjects for patients and controls
    mean_pt = np.mean(np_array, axis=2)
    mean_pt_bothhemi = np.squeeze(np.mean(mean_pt, axis=1))
    mean_ctrl = np.mean(np_array_ctrl, axis=2)
    mean_ctrl_bothhemi = np.squeeze(np.mean(mean_ctrl, axis=1))


    # Plot the same for pooled hemispheres
    plt.figure(figsize=(14, 6))
    time_samples = np.linspace(-500, 2500, mean_pt_bothhemi.shape[0])  # Adjust the x-axis to start from -500ms
    plt.plot(time_samples, mean_ctrl_bothhemi, color='skyblue', linewidth=2.1)
    plt.plot(time_samples, mean_pt_bothhemi, color='coral', linewidth=2.1)
    plt.xlabel("Time (ms)")
    plt.ylabel('dipole moment (nAm)') 
    plt.grid()
    plt.legend([f'FRDA (n=15)', f'controls (n=15)'])
    plt.xlim(-250, 2250)
    plt.ylim(-50, 50)

    
    bar_height = 3

    plt.barh(-45, 750, height=bar_height, left=0, color='gray', alpha=0.7)
    plt.text(375, -45 , '\u03C1 1', color='white', ha='center', va='center')

    plt.barh(-45, 750, height=bar_height, left=750, color='darkgray', alpha=0.7)
    plt.text(1125, -45 , '\u03C1 2', color='black', ha='center', va='center')
    # Save the plot
    plt.savefig(f'{name_pt}_{name_ctrl}_mean_rho_bothhemi_unfiltered.png')
    plt.show()

def plot_overcontrol_rho(np_array, np_array_ctrl, name_pt, name_ctrl):
    """
    Function to plot the data for the mean of patients over the mean of controls in one plot for rho values.

    Parameters:
    np_array (np.ndarray): Data to plot.
    name_pt (str): Name of patient condition.
    name_ctrl (str): Name of control condition.
    """
   
    # Get the mean of all subjects for patients and controls
    mean_pt = np.mean(np_array, axis=2)
    mean_pt_bothhemi = np.squeeze(np.mean(mean_pt, axis=1))
    mean_ctrl = np.mean(np_array_ctrl, axis=2)
    mean_ctrl_bothhemi = np.squeeze(np.mean(mean_ctrl, axis=1))

    # Bandpass zero-phase filter the data
    # Filtering parameters
    fs = 1000  # Sampling frequency
    lowcut = 1.0  # Low cutoff frequency
    highcut = 30.0  # High cutoff frequency
    order = 2  # Filter order
    def butter_bandpass(lowcut, highcut, fs, order=2):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a
    def zero_phase_filter_1d(data1d, lowcut, highcut, fs, order=2):
        b, a = butter_bandpass(lowcut, highcut, fs, order)
        filtered = filtfilt(b, a, data1d)
        return filtered
    # Apply the filter
    mean_pt_bothhemi = zero_phase_filter_1d(mean_pt_bothhemi, lowcut, highcut, fs, order)
    mean_ctrl_bothhemi = zero_phase_filter_1d(mean_ctrl_bothhemi, lowcut, highcut, fs, order)

    # Plot the same for pooled hemispheres
    plt.figure(figsize=(14, 6))
    time_samples = np.linspace(-500, 2500, mean_pt_bothhemi.shape[0])  # Adjust the x-axis to start from -500ms
    plt.plot(time_samples, mean_pt_bothhemi, color='coral', linewidth=DEFAULT_LINEWIDTH)
    plt.plot(time_samples, mean_ctrl_bothhemi, color='skyblue', linewidth=DEFAULT_LINEWIDTH)
    plt.xlabel("Time (ms)")
    plt.ylabel('dipole moment (nAm)') 
    plt.grid()
    plt.legend([f'FRDA (n=15)', f'controls (n=15)'])
    plt.xlim(-250, 2250)
    plt.ylim(-50, 50)

    
    bar_height = 3

    plt.barh(-45, 750, height=bar_height, left=0, color='gray', alpha=0.7)
    plt.text(375, -45 , '\u03C1 1', color='white', ha='center', va='center')

    plt.barh(-45, 750, height=bar_height, left=750, color='darkgray', alpha=0.7)
    plt.text(1125, -45 , '\u03C1 2', color='black', ha='center', va='center')

    # Save the plot
    plt.savefig(f'{name_pt}_{name_ctrl}_mean_rho_bothhemi.png')
    plt.show()

# Example usage
# plot_overcontrol_rho(rho_data, rho_data_ctrl, "Rho", "Rho Controls")

def plot_single_hemi(np_array, hemi, name):
    """
    Function to plot the data for a single hemisphere.

    Parameters:
    np_array (np.ndarray): Data to plot.
    hemi (int): Hemisphere index (0 for left, 1 for right).
    name (str): Name of condition and plot.
    """
    # Filter the data zero phase 1-30Hz
    fs = 1000
    lowcut = 1.0
    highcut = 30.0
    order = 2
    def butter_bandpass(lowcut, highcut, fs, order=2):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a
    def zero_phase_filter_3d(data3d, lowcut, highcut, fs, order=2):
        b, a = butter_bandpass(lowcut, highcut, fs, order)
        filtered = np.empty_like(data3d)
        
        # Iterate over hemispheres and swf files
        for hemi in range(data3d.shape[1]):
            for swf in range(data3d.shape[2]):
                filtered[:, hemi, swf] = filtfilt(b, a, data3d[:, hemi, swf])
        
        return filtered
    
    # Apply the filter
    np_array = zero_phase_filter_3d(np_array, lowcut, highcut, fs, order)
    
    # Plot the data for the specified hemisphere
    plt.figure(figsize=(14, 6))
    time_samples = np.linspace(-500,-5000, np_array.shape[0])  # Adjust the x-axis to start from -500ms
    plt.plot(time_samples, np_array[:, hemi, :], label=f'Hemisphere {hemi + 1}', linewidth=DEFAULT_LINEWIDTH)
    # If multiple columns (subjects) are plotted, matplotlib will draw one line per column.
    plt.xlabel("Time (ms)")
    plt.ylabel('dipole moment (nAm)') 
    plt.grid()
    plt.xlim(-50, 3000)

    # Add a legend (IDs)
    plt.legend([f'Person {i + 1}' for i in range(np_array.shape[2])])

    
    # Save the plot
    plt.savefig(f'{name}_hemisphere_{hemi + 1}.png')
    plt.show()
