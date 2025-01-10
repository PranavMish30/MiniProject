import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load the model
autoencoder = load_model('autoencoder_model.h5',compile=False)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
bilstm = load_model('bilstm_model.h5',compile=False)
bilstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("Models loaded successfully.")



# Load data
data = pd.read_csv('train.csv')  # Replace with your test data file
X = data.drop(columns=['label'])  # Features
y = data['label']  # Ground truth labels
y_test = y.values

encoder = autoencoder.get_layer('dense_1')  # Replace 'bottleneck' with your layer's name
bottleneck_model = tf.keras.Model(inputs=autoencoder.input, outputs=encoder.output)
bottleneck_output = bottleneck_model.predict(X)

timesteps = 1  # Adjust as needed
features = bottleneck_output.shape[1]  # Should be 6 from bottleneck layer
X_reshaped = bottleneck_output.reshape(bottleneck_output.shape[0], timesteps, bottleneck_output.shape[1])

predictions = bilstm.predict(X_reshaped)


predictions = predictions.flatten()
# y_test = y.flatten()

# Select a subset (e.g., first 100 samples) for visualization if the dataset is large
# num_samples = 100
# predictions_subset = predictions[:num_samples]
# ground_truth_subset = y_test[:num_samples]
plt.figure(figsize=(12, 6))

# Plot predictions
plt.plot(predictions, label='Predictions', marker='o', linestyle='-', color='blue')

# Plot ground truth
plt.plot(y_test, label='Ground Truth', marker='*', linestyle='--', color='red')

# Add labels, legend, and title
plt.title('Predictions vs. Ground Truth')
plt.xlabel('Sample Index')
plt.ylabel('Values')
plt.legend()
plt.grid(True)

plt.savefig('predictions_vs_ground_truth.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()




predicted_classes = (predictions > 0.5).astype(int)  # Convert probabilities to binary classes

# Generate confusion matrix
cm = confusion_matrix(y, predicted_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.savefig('confusion_matrix_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()


plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='red', label='Ground Truth', alpha=0.6)
plt.scatter(range(len(predictions)), predictions, color='blue', label='Predictions', alpha=0.6)

# Add labels, legend, and title
plt.title('Predictions vs Ground Truth (Point Graph)', fontsize=14)
plt.xlabel('Sample Index', fontsize=12)
plt.ylabel('Values', fontsize=12)
plt.legend()
plt.grid(True)

plt.savefig('predictions_vs_ground_truth_scatter.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()