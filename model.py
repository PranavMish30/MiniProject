import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# def data_generator(file_paths, batch_size):
#     """
#     Generator to yield data batches from multiple CSV files.

#     Args:
#     - file_paths: List of file paths to CSV files.
#     - batch_size: Number of samples per batch.

#     Yields:
#     - Batch of input data (X_batch).
#     """
#     while True:  # Infinite loop for continuous data loading
#         for file_path in file_paths:
#             # Read file in chunks
#             for chunk in pd.read_csv(file_path, chunksize=batch_size):
#                 X_batch = chunk.values  # Convert to numpy array
#                 yield X_batch, X_batch  # Autoencoder: input = output


# Define the input layer with 83 neurons
input_layer = Input(shape=(20,))

# Define the encoder
encoded = Dense(12, activation='relu')(input_layer)  # L1: 32 neurons
encoded = Dense(6, activation='relu')(encoded)      # L2: 24 neurons

# Define the decoder
decoded = Dense(12, activation='relu')(encoded)      # L3: 32 neurons
output_layer = Dense(20, activation='sigmoid')(decoded)  # Output layer to reconstruct input

# Build the autoencoder model
autoencoder = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
autoencoder.compile(optimizer='adam', loss='mse')

# Summary of the model
autoencoder.summary()

# # Assume you have prepared your data as `X_train`
# autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, validation_split=0.2)

# # List of CSV file paths
# file_paths = ['train_LDAP.csv', 'train_MSSQL.csv', 'train_NetBIOS.csv', 'train_UDP.csv']

# # Batch size
# batch_size = 64

# # Initialize the generator
# train_generator = data_generator(file_paths, batch_size)

# # Number of samples in your dataset
# total_samples = 5950604  # Replace with the actual number of rows in all files combined

# Train the autoencoder
autoencoder.fit(
    train_generator,
    steps_per_epoch=total_samples // batch_size,
    epochs=5
)

# Save the trained model
autoencoder.save('autoencoder_model.h5')
