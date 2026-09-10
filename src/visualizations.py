import numpy as np
import matplotlib.pyplot as plt


def biplot(scores, loadings, labels=None, title=None):
    '''
    Plots a biplot with given scores and loadings

    DISCLAIMER:
    Since there isn't as easily available biplot function for Python as there is for MATLAB,
    this code was created using Copilot to emulate the behavior of MATLAB's biplot function 
    with some additional improvements (coloring the time observations by passed time).
    '''
    
    plt.figure(figsize=(8, 6))

    scores_plot = scores[:, :2] / np.max(np.abs(scores[:, :2]))
    time_colors = np.arange(scores.shape[0])
    plt.scatter(scores_plot[:, 0], scores_plot[:, 1], c=time_colors, alpha=0.5)
    plt.colorbar(label='Time [seconds from the start]')

    for i in range(loadings.shape[0]):
        plt.arrow(0, 0,
                  loadings[i, 0],
                  loadings[i, 1],
                  color='r',
                  alpha=0.8)

        if labels is not None:
            plt.text(loadings[i, 0] * 1.1,
                     loadings[i, 1] * 1.1,
                     labels[i],
                     color='k')

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True)
    plt.title(title)
    plt.show()


def loading_plot(loadings, labels=None, title=None):
    '''
    Plots a loading plot with given loadings

    DISCLAIMER:
    Reused LLM generated code from biplot().
    '''

    plt.figure(figsize=(8, 6))

    for i in range(loadings.shape[0]):
        plt.arrow(0, 0,
                  loadings[i, 0],
                  loadings[i, 1],
                  color='r',)

        if labels is not None:
            plt.text(loadings[i, 0] * 1.1,
                     loadings[i, 1] * 1.1,
                     labels[i],
                     color='k')

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True)
    plt.title(title)
    plt.show()