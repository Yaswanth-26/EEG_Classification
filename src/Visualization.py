"""
Visualization functions for EEG data and model results.
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def plot_eeg_signals(data, indices=None, labels=None, title="EEG Signals"):
    """
    Plot multiple EEG signals.
    
    Args:
        data (numpy.ndarray): EEG data with shape (n_samples, n_timepoints)
        indices (list, optional): Indices of signals to plot
        labels (list, optional): Labels for the signals
        title (str): Plot title
    """
    if indices is None:
        # If no indices provided, select first 5 samples
        indices = range(min(5, data.shape[0]))
    
    plt.figure(figsize=(12, 8))
    
    for i, idx in enumerate(indices):
        plt.subplot(len(indices), 1, i+1)
        plt.plot(data[idx])
        
        if labels is not None:
            label_text = f"Sample {idx} (Class: {labels[idx]})"
        else:
            label_text = f"Sample {idx}"
            
        plt.title(label_text)
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.9)
    plt.show()

def plot_eeg_by_class(data, labels, class_names, samples_per_class=2):
    """
    Plot EEG signals grouped by class.
    
    Args:
        data (numpy.ndarray): EEG data
        labels (list): Class labels
        class_names (list): Names of the classes
        samples_per_class (int): Number of samples to plot per class
    """
    unique_labels = np.unique(labels)
    
    plt.figure(figsize=(15, 10))
    
    for i, label in enumerate(unique_labels):
        # Find indices of samples with this label
        indices = np.where(np.array(labels) == label)[0]
        
        # Select a subset of samples
        selected_indices = indices[:samples_per_class]
        
        for j, idx in enumerate(selected_indices):
            plt.subplot(len(unique_labels), samples_per_class, i*samples_per_class + j + 1)
            plt.plot(data[idx])
            plt.title(f"{class_names[label]} (Sample {j+1})")
            plt.xlabel("Time")
            plt.ylabel("Amplitude")
    
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, feature_names=None):
    """
    Plot feature importance for models that support it.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names (list, optional): Names of the features
    """
    if not hasattr(model, 'feature_importances_'):
        raise AttributeError("Model does not have feature_importances_ attribute")
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(importances))]
    
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance in Model')
    plt.tight_layout()
    plt.show()

def visualize_data_clusters(data, labels, method='pca', class_names=None):
    """
    Visualize data clusters using dimensionality reduction.
    
    Args:
        data (numpy.ndarray): Data to visualize
        labels (list): Class labels
        method (str): Dimensionality reduction method ('pca' or 'tsne')
        class_names (list, optional): Names of the classes
    """
    # Reshape data if needed
    if len(data.shape) > 2:
        data_reshaped = data.reshape(data.shape[0], -1)
    else:
        data_reshaped = data
    
    # Apply dimensionality reduction
    if method.lower() == 'pca':
        reducer = PCA(n_components=2)
    elif method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    else:
        raise ValueError("Method must be 'pca' or 'tsne'")
    
    # Transform data
    data_2d = reducer.fit_transform(data_reshaped)
    
    # Set up class names
    if class_names is None:
        class_names = [f"Class {i}" for i in np.unique(labels)]
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    for i, label in enumerate(np.unique(labels)):
        mask = np.array(labels) == label
        plt.scatter(data_2d[mask, 0], data_2d[mask, 1], label=class_names[i])
    
    plt.title(f'Data Visualization using {method.upper()}')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
