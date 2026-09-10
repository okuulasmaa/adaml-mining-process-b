import numpy as np
import pandas as pd


def undersample_by_period(X: pd.DataFrame) -> np.ndarray:
    '''
    Undersamples data by taking the taking the average of every period of time
    '''

    times = X["date"].unique()
    X_bar = np.zeros((times.shape[0], X.shape[1]))

    for i, time in enumerate(times):
        X_bar[i, :] = X[X["date"] == time].mean().to_numpy()

    return X_bar

