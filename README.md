# EEG Seizure Classification

This project implements deep learning models for classifying EEG signals to detect seizures.

## Project Overview

Electroencephalography (EEG) is a method for recording electrical activity in the brain, commonly used in diagnosis of epilepsy and other neurological disorders. This project builds classification models that can identify seizure patterns in EEG recordings, with models supporting both binary classification (healthy vs. seizure) and multi-class classification (healthy, mild seizure, seizure).

## Data

The project uses EEG datasets that contain recordings from epileptic patients:

## 1. CHB-MIT Scalp EEG Database

This dataset consists of EEG recordings from pediatric subjects with intractable seizures, collected at the Children's Hospital Boston. It contains recordings from 22 subjects (5 males, ages 3-22 and 17 females, ages 1.5-19).

- **Dataset Link**: [CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/)
- **Number of seizures**: 198 seizures (182 in the original set of 23 cases)
- **Sampling rate**: 256 samples per second with 16-bit resolution
- **Electrode placement**: International 10-20 system

## 2. Bonn EEG Dataset

This dataset comes from the University of Bonn, Germany, and is a widely used benchmark for epileptic seizure detection algorithms.

- **Dataset Link**: Available through various research repositories, typically referenced as University of Bonn dataset
- **Structure**: Contains five sets (A-E) with 100 single-channel EEG segments each
- **Classes**:
  - Set A (Z): EEG recordings from healthy subjects with eyes open
  - Set B (O): EEG recordings from healthy subjects with eyes closed 
  - Set C (N): EEG recordings from seizure-free intervals, recorded from the hippocampal formation of the opposite hemisphere of the brain
  - Set D (F): EEG recordings from seizure-free intervals, recorded from the epileptogenic zone
  - Set E (S): EEG recordings from seizure activity

Each EEG segment contains 4097 data points over 23.6 seconds with a sampling frequency of 173.61 Hz.

In our project, the data is organized according to the following classes:
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
├── README.md
├── data/                      # Data directory
├── notebooks/                 # Jupyter notebooks
│   └── original_analysis.ipynb
├── src/                       # Source code
│   ├── __init__.py
│   ├── data_processing.py     # Data loading and label creation
│   ├── preprocessing.py       # Signal preprocessing
│   ├── models.py              # Model architectures
│   ├── evaluation.py          # Model evaluation
│   └── visualization.py       # Data and result visualization
├── main.py                    # Main script
├── requirements.txt           # Required dependencies
└── setup.py                   # Package setup file
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
