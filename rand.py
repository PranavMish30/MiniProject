
import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler

train_dir = "./"
file_path = os.path.join(train_dir,"train_UDP.csv")
train_encoded = pd.read_csv(file_path)
print("Files read...")

print("Shape of encoded dataframe :",train_encoded.shape)