import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler

# Construct the training Dataset.

# Directory containing the CSV files
train_dir = "./Data"
file_path = os.path.join(train_dir,"dataset.csv")
train_Data = pd.read_csv(file_path)


# Drop unnecessary columns.
drop_cols = ["src","dst","Protocol"]
# Training Dataset:
train_Data.drop(drop_cols,axis=1,inplace=True)
print("Unwanted columns dropped from training dataframe...")



# Replace null values with 0.
train_Data = train_Data.fillna(0)
print("Null values removed from training dataframe...")



# # One Hot Encoding for categorical columns.
# # Training Dataset: 
# # categorical_columns = train_Data.select_dtypes(include=['object']).columns.tolist()
# categorical_columns = [" Label"]
# encoder = OneHotEncoder(sparse_output=False)

# # Apply one-hot encoding to the categorical columns
# one_hot_encoded = encoder.fit_transform(train_Data[categorical_columns])

# #Create a DataFrame with the one-hot encoded columns
# #We use get_feature_names_out() to get the column names for the encoded data
# one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))

# # Concatenate the one-hot encoded dataframe with the original dataframe
# train_encoded = pd.concat([train_Data, one_hot_df], axis=1)

# # Drop the original categorical columns
# train_encoded = train_encoded.drop(categorical_columns, axis=1)
# print("Training dataframe encoded...")


# MinMax Normalization of the dataset.
# Training Dataset:
scaler = MinMaxScaler()
model=scaler.fit(train_Data)
train_final=model.transform(train_Data)
train_final_df = pd.DataFrame(train_final, columns=train_Data.columns)
print("Training dataframe scaled...")

# Dataframe details.
print("No. of columns of training dataframe :",train_final_df.shape[1])

# Saving the preprocessed dataset.
# Save to a new CSV file
train_final_df.to_csv("train.csv", index=False)
print("Scaled dataframe saved to train.csv")