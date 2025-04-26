"""
Main script for the EEG Classification Project.
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from src.data_processing import load_eeg_data, create_binary_labels, create_tri_labels
from src.preprocessing import preprocess_eeg_data, plot_eeg_signal
from src.models import (
    prepare_data_for_model, 
    create_lstm_model, 
    create_bilstm_model, 
    create_lstm_cnn_model,
    train_model
)
from src.evaluation import (
    evaluate_binary_model, 
    evaluate_multiclass_model, 
    plot_training_history,
    print_model_evaluation
)
from src.visualization import plot_eeg_by_class, visualize_data_clusters

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='EEG Classification')
    
    parser.add_argument('--data_path', type=str, default='data',
                        help='Path to the data directory')
    parser.add_argument('--mode', type=str, choices=['binary', 'multiclass'], default='binary',
                        help='Classification mode: binary or multiclass')
    parser.add_argument('--model_type', type=str, choices=['lstm', 'bilstm', 'cnn_lstm'], default='lstm',
                        help='Model type to use')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize data and results')
    
    return parser.parse_args()

def main():
    """Main function for the EEG Classification project."""
    # Parse arguments
    args = parse_arguments()
    
    print(f"EEG Classification - Mode: {args.mode}, Model: {args.model_type}")
    print("-" * 50)
    
    # Load data
    print("Loading data...")
    all_data, all_names = load_eeg_data(args.data_path)
    
    # Create labels based on the selected mode
    if args.mode == 'binary':
        labels = create_binary_labels(all_names)
        class_names = ['Healthy', 'Seizure']
    else:  # multiclass
        labels = create_tri_labels(all_names)
        class_names = ['Healthy', 'Mild Seizure', 'Seizure']
    
    # Preprocess data
    print("Preprocessing data...")
    processed_data = np.array([preprocess_eeg_data(data) for data in all_data])
    
    # Visualize data if requested
    if args.visualize:
        print("Visualizing data samples...")
        plot_eeg_by_class(processed_data, labels, class_names)
        visualize_data_clusters(processed_data, labels, method='pca', class_names=class_names)
    
    # Prepare data for model
    X_train, X_test, y_train, y_test = prepare_data_for_model(processed_data, labels)
    
    # Create model based on selected type
    print(f"Creating {args.model_type} model...")
    input_shape = (X_train.shape[1], 1)  # (timesteps, features)
    num_classes = 1 if args.mode == 'binary' else 3
    
    if args.model_type == 'lstm':
        model = create_lstm_model(input_shape, num_classes)
    elif args.model_type == 'bilstm':
        model = create_bilstm_model(input_shape, num_classes)
    else:  # cnn_lstm
        model = create_lstm_cnn_model(input_shape, num_classes)
    
    # Print model summary
    model.summary()
    
    # Train model
    print(f"Training {args.model_type} model...")
    model, history = train_model(model, X_train, y_train, X_test, y_test, 
                                args.epochs, args.batch_size)
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model
    print("Evaluating model...")
    if args.mode == 'binary':
        accuracy, y_pred, report, conf_matrix = evaluate_binary_model(model, X_test, y_test)
    else:  # multiclass
        accuracy, y_pred, report, conf_matrix = evaluate_multiclass_model(model, X_test, y_test)
    
    # Print evaluation results
    print_model_evaluation(accuracy, report, conf_matrix, class_names)
    
    print("Done!")

if __name__ == "__main__":
    main()
