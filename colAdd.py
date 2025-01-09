import dask.dataframe as dd

# Load your existing CSV file into a Dask DataFrame
df = dd.read_csv("train_NetBIOS.csv")
val = float(0)
# Add 4 columns with all rows set to 0
df = df.assign(
    # Label_DrDoS_NetBIOS=val,
     Label_DrDoS_MSSQL=val,
     Label_DrDoS_LDAP=val,
     Label_DrDoS_UDP=val,
    # Label_BENIGN=val
)

# Write the modified DataFrame back to a CSV
df.to_csv("train_NetBIOS.csv", index=False, single_file=True)
