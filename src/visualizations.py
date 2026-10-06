from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def visualize_distribution(data_df, variable_idx, bins=100):

    fig = plt.figure(figsize=(10, 6))

    for k, i in enumerate(variable_idx):
        ax = fig.add_subplot(1, len(variable_idx), k+1)
        ax.hist(data_df.iloc[:, i], bins=bins)
        ax.set_title(data_df.columns[k])
        
    plt.show()


def visualize_time(data_df, idx):

    time = np.arange(data_df.shape[0])

    fig = plt.figure(figsize=(10, 6))

    for k, i in enumerate(idx):
        ax = fig.add_subplot(1, len(idx), k+1)
        ax.plot(time, data_df.iloc[:, i])
        ax.set_title(data_df.columns[i])
        ax.set_ylabel(f"f_{i+1}")
        ax.set_xlabel("Time index")
        ax.grid()

    plt.show()
    

def biplot(scores, loadings, filename, variable_labels, time_colors=None, title=None, show=False):
    '''
    Plots a biplot with given scores and loadings
    '''

    save_path = Path(f"figures/{filename}.pdf")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Save number of variables
    n_variables = loadings.shape[0]

    # Initialize figure
    plt.figure(figsize=(10, 6))

    # Set a grid to the figure
    plt.grid()
    plt.gca().set_axisbelow(True) 

    # Plot the scores 
    scores_plot = scores[:, :2] / np.max(np.abs(scores[:, :2]))
    plt.scatter(scores_plot[:,0], scores_plot[:,1],c=time_colors, cmap="gray",edgecolors="black",linewidths=0.15,s=15)

    # Colors for the features
    colors = list(plt.cm.tab20.colors) + list(plt.cm.tab10.colors[:3])

    # List for used colors
    used_colors = []

    # Plot the loadings
    for i in range(n_variables):
        temp1 = np.array([0., loadings[i,0]])
        temp2 = np.array([0., loadings[i,1]])
        if colors[i] in used_colors:
            plt.plot(temp1,temp2,color=colors[i],linestyle="--")
        else:
            plt.plot(temp1,temp2,color=colors[i])
        used_colors.append(colors[i])

    # Create labels
    labels = ["Scores"]
    for label in variable_labels:
        labels.append(label)

    # Create a color bar for the time 
    cbar  = plt.colorbar(location="left", pad=0.15)
    cbar.set_label("Time")
    
    # Settings
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(labels,fontsize=8,bbox_to_anchor=(1.02, 1),loc="upper left")
    plt.tight_layout()

    # Saving the figure
    save_location = Path("figures/"+filename+".pdf")
    save_location.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    if show:
        plt.show()


def visualize_date_constant_variables(data_df, how_many_dates, how_many_variables, show=False):

    save_path = Path("figures/time_consant_variables.pdf")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize plot
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Bar chart (variables)
    bar_values = np.zeros(np.shape(how_many_dates))
    bar_values[how_many_dates  >  0] = np.log10(how_many_dates[how_many_dates  >  0])
    ax1.bar(range(len(how_many_dates)),bar_values)
    ax1.set_xlabel("Variables")
    ax1.set_ylabel("Number of timestamps where constant (log10)")
    ax1.set_xticks(range(len(how_many_dates)))
    ax1.set_xticklabels(["f" + str(i+2) for i in range(len(how_many_dates))], rotation=45)

    # 2. Line plot (unique timestamps on the horizontal axis)
    unique_dates = data_df["date"].unique()
    plot_values = np.zeros(np.shape(how_many_variables))
    plot_values[how_many_variables  >  0] = np.log10(how_many_variables[how_many_variables  >  0])
    ax2.plot(unique_dates, plot_values)
    ax2.set_xlabel("Timestamp")
    ax2.set_ylabel("Number of constant variables (log10)")

    plt.tight_layout()
    plt.savefig(save_path)
    if show:
        plt.show()