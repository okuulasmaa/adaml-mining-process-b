import numpy as np
import pandas as pd

# Preprocessing data to float precision
data_df = pd.read_csv("./data/MiningProcess_Flotation_Plant_Database.csv", decimal=",")
data_df["date"] = pd.to_datetime(data_df["date"])
data_df["date"] = data_df["date"].astype(int)//1e9 # To seconds, source: https://stackoverflow.com/questions/54312802/pandas-convert-from-datetime-to-integer-timestamp
data_df = data_df.astype(float)

# Number of missing values
n_null = np.sum(data_df.isnull().to_numpy())
print(n_null)
