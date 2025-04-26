"""
Models for EEG classification.
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Conv1D, MaxPooling1D, Flatten
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def prepare_data_for_model(X, y, test_size=0.2, random_state=42):
    """
    Prepare data for model training by splitting and standardizing.
    
    Args:
        X (numpy.ndarray): Input features (EEG data)
        y (numpy.ndarray): Target labels
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test) - Split and standardized data
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Convert to NumPy arrays if they aren't already
    X_train = tf.convert_to_tensor(X_train)
    X_test = tf.convert_to_tensor(X_test)
    y_train = tf.convert_to_tensor(y_train)
    y_test = tf.convert_to_tensor(y_test)
    
    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).reshape(X_train.shape)
    X_test = scaler.transform(X_test).reshape(X_test.shape)
    
    return X_train, X_test, y_train, y_test

def create_lstm_model(input_shape, num_classes=1):
    """
    Create an LSTM model for EEG classification.
    
    Args:
        input_shape (tuple): Shape of input data (samples, timesteps)
        num_classes (int): Number of output classes (1 for binary, >1 for multi-class)
        
    Returns:
        tf.keras.Model: LSTM model
    """
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape))
    
    if num_classes == 1:
        # Binary classification
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    else:
        # Multi-class classification
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

def create_bilstm_model(input_shape, num_classes=1):
    """
    Create a Bidirectional LSTM model for EEG classification.
    
    Args:
        input_shape (tuple): Shape of input data (samples, timesteps)
        num_classes (int): Number of output classes (1 for binary, >1 for multi-class)
        
    Returns:
        tf.keras.Model: Bidirectional LSTM model
    """
    model = Sequential()
    model.add(Bidirectional(LSTM(64), input_shape=input_shape))
    
    if num_classes == 1:
        # Binary classification
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    else:
        # Multi-class classification
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

def create_lstm_cnn_model(input_shape, num_classes=1):
    """
    Create a combined LSTM and CNN model for EEG classification.
    
    Args:
        input_shape (tuple): Shape of input data (samples, timesteps)
        num_classes (int): Number of output classes (1 for binary, >1 for multi-class)
        
    Returns:
        tf.keras.Model: Combined LSTM and CNN model
    """
    model = Sequential()
    model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(64, return_sequences=True))
    model.add(Flatten())
    
    if num_classes == 1:
        # Binary classification
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    else:
        # Multi-class classification with an additional hidden layer
        model.add(Dense(64, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs=10, batch_size=32):
    """
    Train a model on the given data.
    
    Args:
        model (tf.keras.Model): Model to train
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training labels
        X_test (numpy.ndarray): Test features
        y_test (numpy.ndarray): Test labels
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        
    Returns:
        tuple: (model, history) - Trained model and training history
    """
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test)
    )
    
    return model, history
