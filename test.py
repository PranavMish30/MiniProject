import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler

# Construct the testing Dataset.

# Directory containing the CSV files
test_dir = "./Dataset/test"

# Read all CSV files and combine them into one DataFrame
test_dataframes = []
for file in os.listdir(test_dir):
    if file.endswith(".csv"):
        file_path = os.path.join(test_dir, file)
        df = pd.read_csv(file_path)
        test_dataframes.append(df)

# Concatenate all DataFrames into a single one
test_Data = pd.concat(test_dataframes, ignore_index=True)
print("Testing dataframe created...")

# Drop unnecessary columns.
drop_cols = ["Unnamed","Flow ID","Source IP","Destination IP","Source Port","Destination Port","Timestamp","Flow Bytes","Flow Packets","SimilarHTTP"]


# Testing Dataset:
test_Data.drop(drop_cols)
print("Unwanted columns dropped from testing dataframe...")


test_Data = test_Data.fillna(0)
print("Null values removed from testing dataframe...")

# Testing Dataset:

categorical_columns = test_Data.select_dtypes(include=['object']).columns.tolist()
encoder = OneHotEncoder(sparse_output=False)

# Apply one-hot encoding to the categorical columns
one_hot_encoded = encoder.fit_transform(test_Data[categorical_columns])

#Create a DataFrame with the one-hot encoded columns
#We use get_feature_names_out() to get the column names for the encoded data
one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))

# Concatenate the one-hot encoded dataframe with the original dataframe
test_encoded = pd.concat([test_Data, one_hot_df], axis=1)

# Drop the original categorical columns
test_encoded = test_encoded.drop(categorical_columns, axis=1)
print("Testing dataframe encoded...")


# Testing Dataset:

scaler = MinMaxScaler()
model=scaler.fit(test_encoded)
test_final=model.transform(test_encoded)
print("Testing dataframe scaled...")

print("No. of columns of testing dataframe :",test_final.shape[0])

# Save to a new CSV file
test_final.to_csv("test.csv", index=False)