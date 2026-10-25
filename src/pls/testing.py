from sklearn.cross_decomposition import PLSRegression
import numpy as np
import pandas as pd

from src.utils import residual_sum_of_squares
from src.visualizations import plot_test_pred_vs_real, visualize_coefficients, visualize_residuals,t1_t2_scatter_plot


def test_model(pls: PLSRegression, X_test, y_test, y_train, var_names, data_df, show=False):
    '''
    Test PLS1 model.
    '''

    print("Testing:")

    y_pred = pls.predict(X_test)

    dates_test =  X_test[:,0]

    # Metrics
    tss = residual_sum_of_squares(y_train, np.mean(y_train))
    press = residual_sum_of_squares(y_test, y_pred)
    Q2 = 1 - press/tss
    print(f"PRESS: {press:.3f}")
    print(f"Q2: {Q2:.3f}")
    
    # Visualizations
    plot_test_pred_vs_real(y_pred, y_test, dates_test, filename="initial_test_results.pdf", show=show)
    visualize_residuals(y_test - pls.predict(X_test), filename="residuals.pdf", n_bins=50, show=show)
    visualize_coefficients(pls.coef_[0], var_names[:-1], filename="coefficients.pdf", show=show)

    temp = np.unique_values(data_df["date"] - data_df["date"].min()) 
    time_colors = temp[pd.factorize(data_df["date"])[0]]
    t1_t2_scatter_plot(pls,time_colors,[0,1],X_test,show=show)
    
