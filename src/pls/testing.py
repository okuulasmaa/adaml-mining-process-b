from sklearn.cross_decomposition import PLSRegression
import numpy as np

from src.utils import residual_sum_of_squares
from src.visualizations import plot_test_pred_vs_real, visualize_coefficients, visualize_residuals


def test_model(pls: PLSRegression, X_test, y_test, y_train, var_names, show=False):

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

