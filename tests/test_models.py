"""
Tests for model functionality.
"""
import unittest
import numpy as np
import tensorflow as tf

# Import modules from src
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models import create_lstm_model, create_bilstm_model, create_lstm_cnn_model

class TestModels(unittest.TestCase):
    """Test cases for model creation and basic functionality."""

    def test_lstm_model_creation(self):
        """Test that the LSTM model can be created."""
        input_shape = (4097, 1)
        num_classes = 1
        model = create_lstm_model(input_shape, num_classes)
        self.assertIsInstance(model, tf.keras.Model)
        # Check that the model has the expected output shape
        self.assertEqual(model.output_shape, (None, 1))
    
    def test_bilstm_model_creation(self):
        """Test that the BiLSTM model can be created."""
        input_shape = (4097, 1)
        num_classes = 1
        model = create_bilstm_model(input_shape, num_classes)
        self.assertIsInstance(model, tf.keras.Model)
        # Check that the model has the expected output shape
        self.assertEqual(model.output_shape, (None, 1))
    
    def test_lstm_cnn_model_creation(self):
        """Test that the LSTM+CNN model can be created."""
        input_shape = (4097, 1)
        num_classes = 1
        model = create_lstm_cnn_model(input_shape, num_classes)
        self.assertIsInstance(model, tf.keras.Model)
        # Check that the model has the expected output shape
        self.assertEqual(model.output_shape, (None, 1))
    
    def test_model_forward_pass(self):
        """Test that the model can perform a forward pass."""
        input_shape = (4097, 1)
        num_classes = 1
        model = create_lstm_model(input_shape, num_classes)
        
        # Create a small batch of dummy data
        batch_size = 2
        dummy_input = np.random.random((batch_size, *input_shape))
        
        # Perform a forward pass
        output = model(dummy_input, training=False)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, num_classes))

if __name__ == '__main__':
    unittest.main()
