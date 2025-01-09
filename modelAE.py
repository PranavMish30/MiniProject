import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

from sklearn.model_selection import train_test_split


# Load preprocessed data
data = pd.read_csv('train.csv')
# data = data.values
# Split data into training (80%) and testing (20%) sets
X_train, X_test = train_test_split(data, test_size=0.2, random_state=42)
X_train = X_train.drop(columns=['label'])
X_test = X_test.drop(columns=['label'])
X_train = X_train.values
X_test = X_test.values

# Define the input layer with 83 neurons
input_layer = Input(shape=(19,))

# Define the encoder
encoded = Dense(12, activation='relu')(input_layer)  # L1: 32 neurons
bottleneck = Dense(6, activation='relu')(encoded)      # L2: 24 neurons

# Define the decoder
decoded = Dense(12, activation='relu')(bottleneck)      # L3: 32 neurons
output_layer = Dense(19, activation='sigmoid')(decoded)  # Output layer to reconstruct input

# Build the autoencoder model
autoencoder = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
autoencoder.compile(optimizer='adam', loss='mse')

# Summary of the model
autoencoder.summary()


# Train the model
history = autoencoder.fit(
    X_train, X_train,  # AutoEncoder uses the same input for output
    epochs=50,
    batch_size=32,
    validation_data=(X_test, X_test),
    verbose=1
)


# Evaluate model on test data
loss = autoencoder.evaluate(X_test, X_test, verbose=1)
print(f"Test Loss: {loss}")


# Save the entire model
autoencoder.save('autoencoder_model.h5')
print("Model saved to 'autoencoder_model.h5'")
