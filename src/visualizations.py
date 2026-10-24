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


def visualize_time(data_df, idx, filename, show):

    save_path = Path(f"figures/{filename}")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    time = np.arange(data_df.shape[0])

    fig = plt.figure(figsize=(10, 6))

    for k, i in enumerate(idx):
        ax = fig.add_subplot(1, len(idx), k+1)
        ax.plot(time, data_df.iloc[:, i])
        ax.set_title(data_df.columns[i])
        ax.set_xlabel("Time index")
        ax.grid()

    plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    

def biplot(scores, loadings, filename, variable_labels, time_colors=None, title=None, show=False):
    '''
    Plots a biplot with given scores and loadings
    '''

    save_path = Path(f"figures/{filename}")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Save number of variables
    n_variables = loadings.shape[0]

    # Initialize figure
    fig = plt.figure(figsize=(10, 6))

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
    else:
        plt.close(fig)


def visualize_constant_variables(n_constant, filename, show):

    save_path = Path(f"figures/{filename}")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    labels = [f"$f_{{{i}}}$" for i in range(1, len(n_constant)+1)]

    plot_values = np.zeros(np.shape(n_constant))
    plot_values[n_constant>0] = np.log(n_constant[n_constant>0])

    fig = plt.figure(figsize=(14, 5))
    ax = fig.add_subplot(111)
    ax.bar(labels,plot_values)
    ax.set_ylabel("Adjecent constant values (log)")
    ax.tick_params(axis="x", labelsize=8)

    plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return 


def plot_valid_metrics(press, Q2, R2, components, filename, show=False):

    save_path = Path(f"figures/{filename}")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure()

    # PRESS plot
    ax = fig.add_subplot(131)
    ax.plot(components, press)
    ax.axvline(components[np.argmin(press)], linestyle="--")
    ax.grid()
    ax.set_xlabel("LVs")
    ax.set_ylabel("PRESS")
    ax.set_title("PRESS")

    # Q2 plot
    ax = fig.add_subplot(132)
    ax.plot(components, Q2)
    ax.axvline(components[np.argmin(press)], linestyle="--")
    ax.grid()
    ax.set_xlabel("LVs")
    ax.set_ylabel("Q2")
    ax.set_title("Q2")

    # R2 plot
    ax = fig.add_subplot(133)
    ax.plot(components, R2)
    ax.axvline(components[np.argmin(press)], linestyle="--")
    ax.grid()
    ax.set_xlabel("LVs")
    ax.set_ylabel("R2")
    ax.set_title("R2")

    plt.tight_layout()
    plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_test_pred_vs_real(y_pred, y_real, dates, filename, show=False):

    save_path = Path(f"figures/{filename}")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100), 'k')
    sc = ax.scatter(y_real, y_pred, c=dates, cmap='viridis', linewidths=0.20, s=30)
    fig.colorbar(sc, ax=ax)
    ax.set_xlabel("True values")
    ax.set_ylabel("Predicted values")

    plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)


def visualize_residuals(y, filename, n_bins=100, show=False):

    save_path = Path(f"figures/{filename}")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure()

    # Time plot
    ax = fig.add_subplot(121)
    ax.plot(list(range(y.shape[0])), y)
    ax.set_ylabel("Residual")
    ax.set_xlabel("Time index")

    # Residual histogram
    ax = fig.add_subplot(122)
    ax.hist(y, bins=n_bins)
    ax.set_ylabel("Count")
    ax.set_xlabel("Residual")

    plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    

def visualize_coefficients(coefficients, var_names, filename, show=False):

    save_path = Path(f"figures/{filename}")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylabel("Regression coefficient")
    ax.bar(var_names, coefficients)

    plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)


def t1_t2_scatter_plot(pls,time_colors,indices,Xc_test):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    t1 = pls.x_scores_[:,indices[0]]
    t2 = pls.x_scores_[:,indices[1]]

    sc1 =ax1.scatter(t1,t2,c=time_colors[:len(t1)],linewidths=0.15,s=15)
    ax1.set_xlabel("T"+str(indices[0]+1))
    ax1.set_ylabel("T"+str(indices[1]+1))
    ax1.set_title("T scatter plot (train)")
 
    cbar = fig.colorbar(sc1, ax=ax1)
    cbar.set_label('Date')
    
    T_test = pls.transform(Xc_test)
    t1 = T_test[:,indices[0]]
    t2 = T_test[:,indices[1]]

    sc2 = ax2.scatter(t1,t2,c=time_colors[-len(t1):],linewidths=0.15,s=15)
    ax2.set_xlabel("T"+str(indices[0]+1))
    ax2.set_ylabel("T"+str(indices[1]+1))
    ax2.set_title("T scatter plot (test)")

    cbar = fig.colorbar(sc2, ax=ax2)
    cbar.set_label('Date')
 
    plt.savefig("figures/t_scatter_plot_t"+str(indices[0]+1)+"t"+str(indices[1]+1)+".pdf")
    plt.show()

    return




# Emman testi