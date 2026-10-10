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


def check_variables(data_df):

    # This function identifies timestamps at which variables have constant values
    # and the variables that remain constant at those timestamps.

    date_str = data_df.columns[0]
    
    # Number of rows in each timestamp group
    group_sizes = data_df.groupby(date_str).size()
    
    # Number of unique values of each variable at each time step
    nunique_df = data_df.groupby(date_str).nunique()
    
    # Constant only if the group has at least 2 rows
    is_constant = (nunique_df == 1) & (group_sizes >= 2).values[:, None]
    
    # Number of times each variable was constant
    how_many_dates = is_constant.sum(axis=0).to_numpy()
    
    # Number of constant variables at each timestamp
    how_many_variables = is_constant.sum(axis=1).to_numpy()
    
    # Variables that were constant at least once
    bad_variables = is_constant.columns[is_constant.any(axis=0)].tolist()
    
    return how_many_dates, how_many_variables, bad_variables


def count_constant_values(data_df: pd.DataFrame):
    '''
    Counts the number of instances per variable where a variable has constant value between adjecent observations.
    '''

    return np.sum(np.isclose(data_df.diff(), 0.0), axis=0)


def remove_dates(data_df, how_many_variables, limit):

    # This function removes timestamps where more than the specified limit of variables have constant values

    # Get the name of the date column 
    date_str = data_df.columns.tolist()[0]
    
    # Get unique timestamps
    times = data_df[date_str].unique()
    
    # Collect timestamps where the number of constant variables meets or exceeds the limit
    times_to_drop = [times[i] for i, val in enumerate(how_many_variables) if val > limit]
    
    # Return the dataframe excluding the rows with those timestamps
    return data_df[~data_df[date_str].isin(times_to_drop)]


def residual_sum_of_squares(y_true: np.ndarray, y_pred: np.ndarray):
    return np.sum((y_true - y_pred) ** 2)


