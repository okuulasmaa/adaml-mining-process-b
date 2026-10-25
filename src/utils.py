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


def residual_sum_of_squares(y_true: np.ndarray, y_pred: np.ndarray):
    return np.sum((y_true - y_pred) ** 2)


def pca(X, n, verbose=False):
    '''
    Perform PCA with n components.
    '''

    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    V = Vt.T
    loadings = V[:, 0:n]
    scores = X @ loadings

    if verbose:
        # Explained variances
        explained_variances = (S**2) / np.sum(S**2)
        for i in range(n):
            print(f"Explained variance by PC{i+1}: {explained_variances[i]:.3f}")
          
    return scores, loadings


def zscore(X: np.ndarray, mu=None, sig=None):
    '''
    Standardize data using z-score
    '''
    
    if mu is None:
        mu = np.mean(X, axis=0)
    if sig is None:
        sig = np.std(X, axis=0)
    
    return (X - mu) / sig


def explained_variance_table(pls, X, y):
    '''
    Percentage of the variance of X and y explained by each latent variable (LV).
    It is assumed that X and y are already standardized.
    '''

    T = pls.x_scores_ 
    P = pls.x_loadings_
    q = pls.y_loadings_[0]

    t_ss = np.sum(T**2, axis=0)

    var_x = 100 * t_ss * np.sum(P**2, axis=0) / np.sum(X**2)
    var_y = 100 * t_ss * q**2 / np.sum(y**2)

    return pd.DataFrame({
        "LV": np.arange(1, T.shape[1] + 1),
        "X (%)": var_x,
        "X cumulative (%)": np.cumsum(var_x),
        "y (%)": var_y,
        "y cumulative (%)": np.cumsum(var_y),
    }).set_index("LV")