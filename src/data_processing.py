"""
Data loading and processing functions for EEG classification.
"""
import numpy as np
import os
import glob

def load_eeg_data(data_folder):
    """
    Load EEG data from text files in the data folder.
    
    Args:
        data_folder (str): Path to the folder containing the EEG data
        
    Returns:
        tuple: (all_data, all_names) - EEG data arrays and their corresponding folder names
    """
    # Initialize empty lists to store data and corresponding folder names
    all_data = []
    all_names = []
    
    # Loop through each folder in the data directory
    for folder in os.listdir(data_folder):
        folder_path = os.path.join(data_folder, folder)
        
        # Skip non-directories
        if not os.path.isdir(folder_path):
            continue
            
        # Loop through each text file in the current folder
        for file in glob.glob(os.path.join(folder_path, '*.txt')):
            # Load data from the text file using NumPy
            data = np.loadtxt(file)
            
            # Append the data and folder name to the respective lists
            all_data.append(data)
            all_names.append(folder)
    
    # Convert the list of data arrays into a NumPy array
    all_data = np.array(all_data)
    
    return all_data, all_names

def create_binary_labels(all_names):
    """
    Create binary labels from folder names:
    0 for healthy subjects (Z or O folders)
    1 for seizure activity (any other folder)
    
    Args:
        all_names (list): List of folder names
        
    Returns:
        list: Binary labels (0 for healthy, 1 for seizure)
    """
    # Create a list 'labels' initialized with zeros, having the same length as 'all_names'
    labels = [0 for i in range(len(all_names))]
    
    # Iterate through each index in the range of the length of 'all_names'
    for i in range(0, len(all_names)):
        # Check if the name at the current index in 'all_names' is either "Z" or "O"
        if all_names[i] == "Z" or all_names[i] == "O":
            # If true, set the corresponding element in 'labels' to 0 (healthy)
            labels[i] = 0
        else:
            # If false, set the corresponding element in 'labels' to 1 (seizure)
            labels[i] = 1
    
    return labels

def create_tri_labels(all_names):
    """
    Create three-class labels from folder names:
    0 for healthy subjects (Z or O folders)
    1 for mild seizure activity (F or N folders)
    2 for seizure activity (S folder)
    
    Args:
        all_names (list): List of folder names
        
    Returns:
        list: Three-class labels (0 for healthy, 1 for mild seizure, 2 for seizure)
    """
    # Create a list 'tri_labels' initialized with zeros
    tri_labels = [0 for i in range(len(all_names))]
    
    # Iterate and assign labels
    for i in range(0, len(all_names)):
        if all_names[i] == "Z" or all_names[i] == "O":
            tri_labels[i] = 0  # Healthy
        elif all_names[i] == "F" or all_names[i] == "N":
            tri_labels[i] = 1  # Mild Seizure
        else:
            tri_labels[i] = 2  # Seizure
    
    return tri_labels
