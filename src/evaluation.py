"""
Functions for evaluating EEG classification models.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_binary_model(model, X_test, y_test):
    """
    Evaluate a binary classification model.
    
    Args:
        model (tf.keras.Model): Trained model
        X_test (numpy.ndarray): Test features
        y_test (numpy.ndarray): Test labels
        
    Returns:
        tuple: (accuracy, y_pred, report, conf_matrix) - Evaluation metrics
    """
    # Make predictions
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Generate classification report
    report = classification_report(y_test, y_pred, target_names=['Healthy', 'Seizure'])
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return accuracy, y_pred, report, conf_matrix

def evaluate_multiclass_model(model, X_test, y_test):
    """
    Evaluate a multi-class classification model.
    
    Args:
        model (tf.keras.Model): Trained model
        X_test (numpy.ndarray): Test features
        y_test (numpy.ndarray): Test labels
        
    Returns:
        tuple: (accuracy, y_pred, report, conf_matrix) - Evaluation metrics
    """
    # Make predictions
    y_pred = np.argmax(model.predict(X_test), axis=1)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Generate classification report
    report = classification_report(y_test, y_pred, target_names=['Healthy', 'Mild Seizure', 'Seizure'])
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return accuracy, y_pred, report, conf_matrix

def plot_confusion_matrix(conf_matrix, class_names):
    """
    Plot a confusion matrix.
    
    Args:
        conf_matrix (numpy.ndarray): Confusion matrix
        class_names (list): Names of classes
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
               xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

def plot_training_history(history):
    """
    Plot training and validation accuracy and loss over epochs.
    
    Args:
        history (tf.keras.callbacks.History): Training history
    """
    plt.figure(figsize=(10, 8))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def print_model_evaluation(accuracy, report, conf_matrix, class_names):
    """
    Print model evaluation metrics and display confusion matrix.
    
    Args:
        accuracy (float): Model accuracy
        report (str): Classification report
        conf_matrix (numpy.ndarray): Confusion matrix
        class_names (list): Names of classes
    """
    print(f"Test Accuracy: {accuracy}")
    print("\nClassification Report:\n", report)
    print("Confusion Matrix:\n", conf_matrix)
    
    # Plot confusion matrix
    plot_confusion_matrix(conf_matrix, class_names)
