
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import Bidirectional, LSTM, Dense

# Step 1: Load the saved AE model
autoencoder = load_model('autoencoder_model.h5',compile=False)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("AutoEncoder Model Loaded Successfully.")

# Step 2: Create a model that outputs the bottleneck layer
# Assuming the bottleneck layer is named 'bottleneck' in the AE model
encoder = autoencoder.get_layer('dense_1')  # Replace 'bottleneck' with your layer's name
bottleneck_model = tf.keras.Model(inputs=autoencoder.input, outputs=encoder.output)

# Step 3: Load your data
# Assuming your preprocessed data is in a CSV file

data = pd.read_csv('train.csv')  # Replace with your CSV file path
# X,y = train_test_split(data, test_size=0.2, random_state=42)

# X = data.drop(columns=['label'])  # Drop the target column to get features
# y_train = y['label']  # Extract the target column
X = data.drop(columns=['label'])
y_train = data['label'] 


# Step 4: Pass the data through the bottleneck model
bottleneck_output = bottleneck_model.predict(X)
print(f"Bottleneck Output Shape: {bottleneck_output.shape}")  # Shape: (samples, 6)

# Step 5: Reshape for BiLSTM
timesteps = 1  # Adjust as needed
features = bottleneck_output.shape[1]  # Should be 6 from bottleneck layer
X_reshaped = bottleneck_output.reshape(bottleneck_output.shape[0], timesteps, bottleneck_output.shape[1])
print(f"Reshaped Output Shape for BiLSTM: {X_reshaped.shape}")  # Shape: (samples, timesteps, features)



# Define the BiLSTM model
bilstm_model = Sequential()
bilstm_model.add(Bidirectional(LSTM(64, return_sequences=False), input_shape=(timesteps, features)))
bilstm_model.add(Dense(32, activation='relu'))
bilstm_model.add(Dense(1, activation='sigmoid'))  # Binary classification

# Compile the model
bilstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the BiLSTM
history = bilstm_model.fit(X_reshaped, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

# Save the BiLSTM model after training
bilstm_model.save('bilstm_model.h5')  # Save in HDF5 format
print("BiLSTM model saved successfully!")
