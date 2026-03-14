import numpy as np
import os
import re
import matplotlib
matplotlib.use("Qt5Agg")  # Set the backend before pyplot is imported
import matplotlib.pyplot as plt

def read_ascii_dataset(filename, num_rows):
    """
    Reads an ASCII dataset with a specified header and data format.

    Parameters:
        filename (str): Path to the ASCII file.
        num_rows (int): Number of data rows to read.

    Returns:
        tuple: (Npts, data)
            - Npts (int): Number of data points in each row.
            - data (ndarray): A NumPy array of shape (num_rows, Npts) containing the data.
    """
    try:
        with open(filename, 'r') as file:
            # Read and parse the header
            header_line = file.readline().strip()
            match = re.search(r'Npts=\s*(\d+)', header_line)
            if not match:
                raise ValueError("Npts value not found in header")
            Npts = int(match.group(1))  # Extract Npts value

            # Initialize a list to store the data
            data = []

            # Read the specified number of rows
            for _ in range(num_rows):
                line = file.readline().strip()

                # Skip lines that do not start with SD- or SC-
                if not line.startswith(('SD-', 'SC-')):
                    continue

                # Extract the numerical values after SD- or SC-
                values = line.split()[1:]  # Ignore the SD-/SC- prefix
                row_data = np.array(values, dtype=float)

                # Check that the row has the expected number of points
                if len(row_data) != Npts:
                    raise ValueError(f"Row does not match Npts={Npts}. Found {len(row_data)} points.")

                # Add to the data list
                data.append(row_data)

            # Convert data list to a NumPy array
            data = np.array(data)

            # Ensure we have the expected number of rows
            if data.shape[0] != num_rows:
                raise ValueError(f"Expected {num_rows} rows but found {data.shape[0]} valid rows.")

        return Npts, data

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None, None
    except Exception as e:
        print(f"Error reading file '{filename}': {e}")
        return None, None
