
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import Bidirectional, LSTM, Dense


# # Step 1: Load the saved AE model
# autoencoder = load_model('autoencoder_model.h5',compile=False)
# autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# print("AutoEncoder Model Loaded Successfully.")

# autoencoder.summary()
# # Print layer names and their indices
# for i, layer in enumerate(autoencoder.layers):
#     print(f"{i}: {layer.name}")

autoencoder = load_model('autoencoder_model.h5',compile=False)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("AutoEncoder Model Loaded Successfully.")

# Step 2: Create a model that outputs the bottleneck layer
# Assuming the bottleneck layer is named 'bottleneck' in the AE model
encoder = autoencoder.get_layer('dense_1')  # Replace 'bottleneck' with your layer's name
bottleneck_model = tf.keras.Model(inputs=autoencoder.input, outputs=encoder.output)

bottleneck_model.summary()