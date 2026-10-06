import pandas as pd
import numpy as np

from src.utils import undersample_by_period
from src.visualizations import visualize_time, biplot
from src.analysis import pca, zscore


def plot_initial_biplot(data_df, scores, loadings, show):

    # Create colors for the scatter points based on the timestamp
    temp = np.unique_values(data_df["date"] - data_df["date"].min()) 
    time_colors = temp[pd.factorize(data_df["date"])[0]]

    # We  choose at most  4 random obseravtions from each timestamp in the bibplot (to keep the plot clear)
    random_samples_no = 4
    sampled_df = data_df.groupby("date", group_keys=False).apply(lambda x: x.sample(n=min(len(x), random_samples_no)))
    indices = sampled_df.index
    scores_plot_sampled = scores[indices,:]
    time_colors_sampled = time_colors[indices]

    # Create variable labels for the biplot 
    variable_labels = [f"f{i+1}" for i in range(data_df.shape[1])]

    biplot(scores_plot_sampled, loadings, filename="exploratory_biplot.pdf", variable_labels=variable_labels,
        time_colors=time_colors_sampled, title="Exploratory biplot", show=show)


def explore_data(data_df: pd.DataFrame, show):

    print("Exploratory data analysis:")

    data_df = undersample_by_period(data_df)

    # X = data_df.drop(columns=["% Silica Concentrate"]).to_numpy()
    X = data_df.drop(columns=["% Silica Concentrate"]).to_numpy()

    # Statistics
    n_variables = X.shape[1]
    n_observations = data_df.shape[0]
    n_null = np.sum(np.isnan(X))
    print(f"Number of variables: {n_variables}")
    print(f"Number of observations: {n_observations}")
    print(f"Number of missing values: {n_null}")

    # Visualization
    variable_idx = (1, 2, 11, 12)
    visualize_time(data_df, variable_idx, "time.pdf", show)

    # PCA
    Xc = zscore(X)
    scores, loadings = pca(Xc, 2, verbose=True)

    # Plotting
    plot_initial_biplot(data_df, scores, loadings, show)

    print()
