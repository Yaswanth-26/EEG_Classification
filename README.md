# EEG Seizure Classification

This project implements deep learning models for classifying EEG signals to detect seizures.

## Project Overview

Electroencephalography (EEG) is a method for recording electrical activity in the brain, commonly used in diagnosis of epilepsy and other neurological disorders. This project builds classification models that can identify seizure patterns in EEG recordings, with models supporting both binary classification (healthy vs. seizure) and multi-class classification (healthy, mild seizure, seizure).

## Data

The project uses EEG datasets that contain recordings from epileptic patients. The data is organized into the following classes:
- Z, O: Healthy subjects
- F, N: Mild seizure activity
- S: Seizure activity

## Features

- Data loading and preprocessing of EEG signals
- Signal filtering, baseline correction, and normalization
- Multiple deep learning architectures:
  - Basic LSTM model
  - Bidirectional LSTM model
  - Combined CNN-LSTM model
- Support for both binary and multi-class classification
- Comprehensive evaluation metrics and visualizations

## Project Structure

```
eeg-seizure-classification/
??? README.md
??? data/                      # Data directory
??? notebooks/                 # Jupyter notebooks
?   ??? original_analysis.ipynb
??? src/                       # Source code
?   ??? __init__.py
?   ??? data_processing.py     # Data loading and label creation
?   ??? preprocessing.py       # Signal preprocessing
?   ??? models.py              # Model architectures
?   ??? evaluation.py          # Model evaluation
?   ??? visualization.py       # Data and result visualization
??? main.py                    # Main script
??? requirements.txt           # Required dependencies
??? setup.py                   # Package setup file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/eeg-seizure-classification.git
cd eeg-seizure-classification
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Alternatively, install the package:
```bash
pip install -e .
```

## Usage

### Running with Default Settings

```bash
python main.py --data_path path/to/data
```

### Command Line Arguments

- `--data_path`: Path to the data directory (default: 'data')
- `--mode`: Classification mode - 'binary' or 'multiclass' (default: 'binary')
- `--model_type`: Model type - 'lstm', 'bilstm', or 'cnn_lstm' (default: 'lstm')
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size for training (default: 32)
- `--visualize`: Flag to visualize data and results

### Examples

Run binary classification with LSTM model:
```bash
python main.py --data_path data --mode binary --model_type lstm
```

Run multi-class classification with CNN-LSTM model and data visualization:
```bash
python main.py --data_path data --mode multiclass --model_type cnn_lstm --visualize
```

## Results

The models are evaluated using:
- Accuracy
- Precision, recall, and F1-score
- Confusion matrix
- Training and validation curves

## License

[MIT License](LICENSE)