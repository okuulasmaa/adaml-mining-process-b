import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


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


def split_data(X, y, train_size):
    '''
    Splits data into training, validation, and testing subsets.
    validation_size = train_size = (1 - train_size) / 2
    '''

    X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, train_size=train_size, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, train_size=0.5, shuffle=False)

    return X_train, X_val, X_test, y_train, y_val, y_test


def undersample_by_period(X: pd.DataFrame) -> pd.DataFrame:
    '''
    Undersamples data by taking the taking the average of every period of time
    '''

    times = X["date"].unique()

    rows = []
    for i, time in enumerate(times):
        rows.append(X[X["date"] == time].mean().to_list())

    return pd.DataFrame(rows, columns=X.columns)


def count_constant_values(data_df: pd.DataFrame):
    '''
    Counts the number of instances per variable where a variable has constant value between adjecent observations.
    '''

    return np.sum(np.isclose(data_df.diff(), 0.0), axis=0)
