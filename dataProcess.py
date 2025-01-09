import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler

# Construct the training Dataset.

# Directory containing the CSV files
train_dir = "./Dataset/train"
file_path = os.path.join(train_dir,"DrDoS_LDAP.csv")
train_Data = pd.read_csv(file_path)

# # Read all CSV files and combine them into one DataFrame
# train_dataframes = []
# for file in os.listdir(train_dir):
#     if file.endswith(".csv"):
#         file_path = os.path.join(train_dir, file)
#         df = pd.read_csv(file_path)
#         train_dataframes.append(df)

# # Concatenate all DataFrames into a single one
# train_Data = pd.concat(train_dataframes, ignore_index=True)
# print("Training dataframe created...")



# Drop unnecessary columns.
drop_cols = ["Unnamed: 0","Flow ID"," Source IP"," Destination IP"," Source Port"," Destination Port"," Timestamp","Flow Bytes/s"," Flow Packets/s","SimillarHTTP"]
# Training Dataset:
train_Data.drop(drop_cols,axis=1,inplace=True)
print("Unwanted columns dropped from training dataframe...")



# Replace null values with 0.
train_Data = train_Data.fillna(0)
print("Null values removed from training dataframe...")



# One Hot Encoding for categorical columns.
# Training Dataset: 
# categorical_columns = train_Data.select_dtypes(include=['object']).columns.tolist()
categorical_columns = [" Label"]
encoder = OneHotEncoder(sparse_output=False)

# Apply one-hot encoding to the categorical columns
one_hot_encoded = encoder.fit_transform(train_Data[categorical_columns])

#Create a DataFrame with the one-hot encoded columns
#We use get_feature_names_out() to get the column names for the encoded data
one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))

# Concatenate the one-hot encoded dataframe with the original dataframe
train_encoded = pd.concat([train_Data, one_hot_df], axis=1)

# Drop the original categorical columns
train_encoded = train_encoded.drop(categorical_columns, axis=1)
print("Training dataframe encoded...")


train_encoded.to_csv("encodedLDAP.csv", index=False)