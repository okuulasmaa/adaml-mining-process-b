from pathlib import Path
import numpy as np
from src.utils import undersample_by_period, get_data_df
from src.visualizations import biplot


def main():

    data_path = Path("./data/MiningProcess_Flotation_Plant_Database.csv")

    # Loading the data
    data_df = get_data_df(data_path)
    X = undersample_by_period(data_df.drop(columns=["% Silica Concentrate"]))
    y = data_df["% Silica Concentrate"].to_numpy()
    
    # Statistics
    n_variables = X.shape[1]
    n_observations_original = data_df.shape[0]
    n_observations_undersampled = X.shape[0]
    n_null = np.sum(np.isnan(X))
    print(f"Number of variables: {n_variables}")
    print(f"Number of observations: {n_observations_original}")
    print(f"Number of observartions after undersampling: {n_observations_undersampled}")
    print(f"Number of missing values: {n_null}")
    print()

    # PCA
    Xc = (X - np.mean(X, axis=0)) / np.std(X, axis=0) # Standardization
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt.T
    loadings = V[:, 0:2]
    scores = Xc @ loadings

    # Explained variances
    explained_variances = (S**2) / np.sum(S**2)
    print(f"Explained variance by PC1: {explained_variances[0]:.3f}")
    print(f"Explained variance by PC2: {explained_variances[1]:.3f}")
    print(f"Explained variance by PC1 and PC2: {(explained_variances[0] + explained_variances[1]):.3f}")
    print()

    # Plotting
    time_colors = np.unique_values(data_df["date"] - data_df["date"].min()) 
    biplot(scores, loadings, time_colors=time_colors, title="Biplot")


if __name__ == "__main__":
    main()
