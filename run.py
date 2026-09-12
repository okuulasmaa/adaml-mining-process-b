import numpy as np
import pandas as pd
from src.utils import undersample_by_period
from src.visualizations import biplot, loading_plot
import function_storage as fs
import matplotlib.pyplot as plt


def main():

    # Preprocessing data to float precision
    data_df = fs.get_data_df()
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

    # PCA
    Xc = (X - np.mean(X, axis=0)) / np.std(X, axis=0) # Standardization
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt.T
    loadings = V[:, 0:2]
    scores = Xc @ loadings/10

    # Initialize figure
    plt.figure(figsize=(10, 6))

    # Set a grid to the figure
    plt.grid()
    plt.gca().set_axisbelow(True) 

    # Colors for data points
    time_colors = np.unique_values(data_df["date"] - data_df["date"].min()) 
    plt.scatter(scores[:,0],scores[:,1],c=time_colors, cmap="gray",edgecolors="black",linewidths=0.25)

    # Colors for the features
    colors = list(plt.cm.tab20.colors) + list(plt.cm.tab10.colors[:3])

    # List for used colors
    used_colors = []

    # Plot the features
    for i in range(n_variables):
        temp1 = np.array([0.,V[i,0]])
        temp2 = np.array([0.,V[i,1]])
        if colors[i] in used_colors:
            plt.plot(temp1,temp2,color=colors[i],linestyle="--")
        else:
            plt.plot(temp1,temp2,color=colors[i])
        used_colors.append(colors[i])

    # Create labels
    labels = ["Xc"]
    for i in range(n_variables):
        labels.append("f"+str(i+1))

    # Create a color bar for the time 
    cbar  = plt.colorbar(location="left", pad=0.15)
    cbar.set_label("Time")
    
    # Settings
    plt.title("Biplot")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(labels,fontsize=8,bbox_to_anchor=(1.02, 1),loc="upper left")
    plt.tight_layout()
    plt.savefig("figures/biplot_loadings.pdf")
    plt.show()

    # Xc_normalized = Xc / np.linalg.norm(Xc, axis=0, keepdims=True)

    # C = Xc_normalized.T @ Xc_normalized # Feature correlation matrix 

    # plt.imshow(C, cmap="viridis")
    # plt.title("Feature Correlation")
    # plt.colorbar()
    # plt.savefig("figures/feature_correlation.pdf")
    # plt.show()


if __name__ == "__main__":
    main()
