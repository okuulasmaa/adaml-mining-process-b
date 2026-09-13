from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def visualize_distribution(data_df, size=(5,5), bins=100):

    fig = plt.figure(figsize=(10, 6))

    k = 0
    for _ in range(1, size[0] + 1):
        for _ in range(1, size[1] + 1):
            ax = fig.add_subplot(size[0], size[1], k + 1)
            ax.hist(data_df.iloc[:, k], bins=bins)
            ax.set_title(data_df.columns[k])
            k += 1
            if k == data_df.shape[1]:
                break
        if k == data_df.shape[1]:
            break
        
    plt.show()


def visualize_time(data_df, size=(5,5)):

    time = np.arange(data_df.shape[0])

    fig = plt.figure(figsize=(10, 6))

    k = 1
    for _ in range(1, size[0] + 1):
        for _ in range(1, size[1] + 1):
            ax = fig.add_subplot(size[0], size[1], k)
            ax.plot(time, data_df.iloc[:, k])
            ax.set_ylabel(f"f_{k+1}")
            ax.set_xlabel("Time index")
            k += 1
            if k == data_df.shape[1]:
                break
        if k == data_df.shape[1]:
            break
        
    plt.show()
    


def biplot(scores, loadings, time_colors=None, title=None):
    '''
    Plots a biplot with given scores and loadings
    '''

    n_variables = loadings.shape[0]

    # Initialize figure
    plt.figure(figsize=(10, 6))

    # Set a grid to the figure
    plt.grid()
    plt.gca().set_axisbelow(True) 

    # Colors for data points
    scores_plot = scores[:, :2] / np.max(np.abs(scores[:, :2]))
    plt.scatter(scores_plot[:,0], scores_plot[:,1],c=time_colors, cmap="gray",edgecolors="black",linewidths=0.25)

    # Colors for the features
    colors = list(plt.cm.tab20.colors) + list(plt.cm.tab10.colors[:3])

    # List for used colors
    used_colors = []

    # Plot the features
    for i in range(n_variables):
        temp1 = np.array([0., loadings[i,0]])
        temp2 = np.array([0., loadings[i,1]])
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
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(labels,fontsize=8,bbox_to_anchor=(1.02, 1),loc="upper left")
    plt.tight_layout()

    # Saving the figure
    save_location = Path("figures/biplot_loadings.pdf")
    save_location.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig("figures/biplot_loadings.pdf")

    plt.show()