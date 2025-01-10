import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd

# Load models
autoencoder = load_model('autoencoder_model.h5', compile=False)
bilstm = load_model('bilstm_model.h5')

# Title
st.title("Real-Time DDoS Classification")

# Input data
uploaded_file = st.file_uploader("Upload CSV for Prediction", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)

    # Process input
    X = data.drop(columns=['label'])  # Ensure this matches your data
    y = data['label']

    encoder = autoencoder.get_layer('dense_1')  # Replace 'bottleneck' with your layer's name
    bottleneck_model = tf.keras.Model(inputs=autoencoder.input, outputs=encoder.output)
    bottleneck_output = bottleneck_model.predict(X)

    timesteps = 1  # Adjust as needed
    features = bottleneck_output.shape[1]  # Should be 6 from bottleneck layer
    X_reshaped = bottleneck_output.reshape(bottleneck_output.shape[0], timesteps, bottleneck_output.shape[1])

    predictions = bilstm.predict(X_reshaped)

    # Display predictions
    st.write("Predictions:", predictions)

    # Visualize results
    st.line_chart(predictions[:50])  # Display first 50 predictions for simplicity
