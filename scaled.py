
import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler

train_dir = "./"
file_path = os.path.join(train_dir,"encodedLDAP.csv")
train_encoded = pd.read_csv(file_path)
print("File read...")
# MinMax Normalization of the dataset.
# Training Dataset:
scaler = MinMaxScaler()
model=scaler.fit(train_encoded)
train_final=model.transform(train_encoded)
train_final_df = pd.DataFrame(train_final, columns=train_encoded.columns)
print("Training dataframe scaled...")

# Dataframe details.
print("No. of columns of training dataframe :",train_final_df.shape[1])

# Saving the preprocessed dataset.
# Save to a new CSV file
train_final_df.to_csv("trainLDAP.csv", index=False)
print("Scaled dataframe saved to trainLDAP.csv")