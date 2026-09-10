import numpy as np
import pandas as pd

from src.utils import undersample_by_period
from src.visualizations import biplot, loading_plot


def main():

    # Preprocessing data to float precision
    data_df = pd.read_csv("./data/MiningProcess_Flotation_Plant_Database.csv", decimal=",")
    data_df["date"] = pd.to_datetime(data_df["date"])
    data_df["date"] = data_df["date"].astype(int) // 1e9 # To seconds, source: https://stackoverflow.com/questions/54312802/pandas-convert-from-datetime-to-integer-timestamp
    data_df = data_df.astype(float) # All values to same precision
    X = undersample_by_period(data_df.drop(columns=["% Silica Concentrate"]))
    y = data_df["% Silica Concentrate"].to_numpy()
    labels = data_df.columns

    # Statistics
    n_variables = X.shape[1]
    n_observations_original = data_df.shape[0]
    n_observations_undersampled = X.shape[0]
    n_null = np.sum(np.isnan(X))
    print(f"Number of variables: {n_variables}")
    print(f"Number of observations: {n_observations_original}")
    print(f"Number of observartions after undersampling: {n_observations_undersampled}")
    print(f"Number of missing values: {n_null}")

    # PCA
    Xc = (X - np.mean(X, axis=0)) / np.std(X, axis=0) # Standardization
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt.T
    loadings = V[:, 0:2]
    scores = Xc @ loadings

    biplot(scores, loadings, labels)
    loading_plot(loadings, labels)


if __name__ == "__main__":
    main()
