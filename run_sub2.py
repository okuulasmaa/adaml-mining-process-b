from pathlib import Path
import numpy as np
import src.utils as su
import src.visualizations as sv
import pandas as pd
import sys


def main():

    # Import the data from the CSV file
    data_path = Path("./data/MiningProcess_Flotation_Plant_Database.csv")
    data_df = su.get_data_df(data_path)

    # Check are the timestamps evenly distributed
    time_diff = data_df["date"].diff().dropna()
    print(f"Even intervals: {np.allclose(time_diff , time_diff .iloc[0])}")

    # THe orginal number of columns
    orginal_col_no = data_df.shape[1]
    
    # Check that all the values in the data are positive
    all_positive = (data_df > 0).all()
    print("Truth table for variables to be positive")
    print(all_positive)

    # Check variables that are consant at some timestamps
    how_many_dates,how_many_variables,bad_variables = su.check_variables(data_df)

    # Visualize the constant-value problem
    sv.visualize_date_constant_variables(data_df,how_many_dates,how_many_variables)

    # If the silica concentration is constant across all rows at a given timestamp, we only keep the first row and remove the rest
    data_df = data_df[data_df["% Silica Concentrate"].ne(data_df["% Silica Concentrate"].shift())]

    # Remove the three variables that causes the most trouble 
    cols_to_drop = ['% Iron Feed', '% Silica Feed', '% Iron Concentrate']
    dropped_fi = [data_df.columns.get_loc(col)+1 for col in cols_to_drop] # This is for the variable labels in biplot
    data_df = data_df.drop(columns=cols_to_drop)

    # Check the variables again
    _,how_many_variables,bad_variables = su.check_variables(data_df)

    # Remove the remaining timestamps with constant values
    data_df = su.remove_dates(data_df,how_many_variables,0)

    # Check the variables again after filtering the data
    _,_,bad_variables = su.check_variables(data_df)
    if len(bad_variables) == 0:
        print("There are no more constant values at any timestamp.")
    print(f"After filtering there are {data_df.shape[0]} observations.")

    # Remove the target column
    data_df = data_df.drop(columns=["% Silica Concentrate"])

    # Create the data matrix
    data_df = data_df.reset_index(drop=True)
    X = data_df.to_numpy()

    # Center and standardize the data matrix
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0) 

    # PCA
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    V = Vt.T
    loadings = V[:, 0:2]
    scores = X @ loadings

    # Explained variances
    explained_variances = (S**2) / np.sum(S**2)
    print(f"Explained variance by PC1: {explained_variances[0]:.3f}")
    print(f"Explained variance by PC2: {explained_variances[1]:.3f}")
    print(f"Explained variance by PC1 and PC2: {(explained_variances[0] + explained_variances[1]):.3f}")
  
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
    variable_labels = []
    for i in range(orginal_col_no):
        label = i+1
        if label not in dropped_fi:
            variable_labels.append("f"+str(label))

    # Create biplot
    sv.biplot(scores_plot_sampled, loadings, "biplot_sub2",variable_labels,time_colors=time_colors_sampled ,title="Biplot")

    return

if __name__ == "__main__":
    main()
