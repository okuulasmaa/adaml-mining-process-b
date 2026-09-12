import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Function that creates a df object incuding the data
def get_data_df():
    data_df = pd.read_csv("./data/MiningProcess_Flotation_Plant_Database.csv", decimal=",")
    data_df["date"] = pd.to_datetime(data_df["date"]).astype("datetime64[ns]")
    data_df["date"] = data_df["date"].astype(int) // 1e9 # To seconds, source: https://stackoverflow.com/questions/54312802/pandas-convert-from-datetime-to-integer-timestamp
    data_df = data_df.astype(float) # All values to same precision
   
    return data_df