import pandas as pd
import numpy as np

from src.analysis import pca
from src.utils import undersample_by_period, count_constant_values
from src.visualizations import visualize_constant_variables, biplot


def biplot_undersampled(data_df: pd.DataFrame, var_names: list, show=False):
    # Center and standardize the data matrix for PCA
    X = data_df.to_numpy()
    Xc = (X - np.mean(X, axis=0)) / np.std(X, axis=0) 

    # PCA
    scores, loadings = pca(Xc, 2, verbose=True)

    # Create colors for the scatter points based on the timestamp
    temp = np.unique_values(data_df["date"] - data_df["date"].min()) 
    time_colors = temp[pd.factorize(data_df["date"])[0]]

    # Create biplot
    biplot(scores, loadings, "biplot_pretreated.pdf", var_names, time_colors=time_colors, title="Biplot of the pretreated data", show=show)


def pretreat_with_undersampling(data_df: pd.DataFrame, show=False):

    print("Pretreatment:")

    # Check that all the values in the data are positive
    all_positive = (data_df > 0).all()
    print("Truth table for variables to be positive")
    print(all_positive)

    # Undersample by date
    data_df = undersample_by_period(data_df)

    # Check variables that are consant at some timestamps
    n_constant = count_constant_values(data_df)

    # Visualize the constant-value problem
    visualize_constant_variables(n_constant, "n_constant.pdf", show)

    # Check are the timestamps evenly distributed
    time_diff = data_df["date"].diff()
    print(f"Even intervals: {np.allclose(time_diff , 3600)}") 

    # Visually identified that variables 2, 3, 12, 13 have constant periods so we drop those
    drop_idx = np.array([1, 2, 11, 12]);
    num_variables = data_df.shape[1]
    data_df = data_df.drop(columns=data_df.columns[drop_idx])
    var_names = [f"$f_{{{i}}}$" for i in range(1, num_variables + 1) if not i in drop_idx + 1]

    # Create the data matrix and the target vector
    y = data_df["% Silica Concentrate"].to_numpy()
    X = data_df.drop(columns=["% Silica Concentrate"]).to_numpy()

    print(f"Number of variables left: {X.shape[1]}")
    print(f"Number of observations left: {X.shape[0]}")

    biplot_undersampled(data_df.drop(columns=["% Silica Concentrate"]), var_names, show)

    print()

    return X, y, var_names