import numpy as np
import pandas as pd


def get_data_df(data_path):
    '''
    Creates a df object incuding the data
    '''

    data_df = pd.read_csv(data_path, decimal=",")
    data_df["date"] = pd.to_datetime(data_df["date"]).astype("datetime64[ns]")
    data_df["date"] = data_df["date"].astype(int) // 1e9 # To seconds, source: https://stackoverflow.com/questions/54312802/pandas-convert-from-datetime-to-integer-timestamp
    data_df = data_df.astype(float) # All values to same precision
    data_df.sort_values(by=["date"], ascending=True) # Ensure that time series is sorted by time
   
    return data_df


def undersample_by_period(X: pd.DataFrame) -> np.ndarray:
    '''
    Undersamples data by taking the taking the average of every period of time
    '''

    times = X["date"].unique()
    X_bar = np.zeros((times.shape[0], X.shape[1]))

    for i, time in enumerate(times):
        X_bar[i, :] = X[X["date"] == time].mean().to_numpy()

    return X_bar

