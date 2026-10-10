import numpy as np
from sklearn.cross_decomposition import PLSRegression


from src.visualizations import plot_valid_metrics
from src.utils import residual_sum_of_squares



def calibrate(X_train: np.ndarray, X_val: np.ndarray, y_train: np.ndarray, y_val: np.ndarray, show=False):

    print("Calibration:")
    
    best_pls1 = None
    best_lvs = 0
    max_Q2 = 0
    max_R2 = 0
    min_press = np.inf
    min_components = 1
    max_components = X_train.shape[1]
    n_components_list = list(range(min_components, max_components + 1))

    tss = residual_sum_of_squares(y_train, np.mean(y_train))

    press_list = []
    Q2_list = []
    R2_list = []

    for n_components in n_components_list:

        pls1 = PLSRegression(n_components=n_components)

        # Fit pls1 model
        pls1.fit(X_train, y_train)

        # Validate the model
        y_pred, y_train_pred = pls1.predict(X_val), pls1.predict(X_train)
        press = residual_sum_of_squares(y_val, y_pred)
        rss = residual_sum_of_squares(y_train, y_train_pred)
        Q2 = 1 - press/tss
        R2 = 1 - rss/tss
        press_list.append(press)
        Q2_list.append(Q2)
        R2_list.append(R2)
        if press < min_press:
            best_pls1 = pls1
            min_press = press
            max_Q2 = Q2
            max_R2 = R2
            best_lvs = n_components

    print("Best model:")
    print(f"LVs: {best_lvs}")
    print(f"Q2: {max_Q2:.3f}")
    print(f"PRESS: {min_press:.3f}")
    print(f"R2: {max_R2:.3f}")

    plot_valid_metrics(press_list, Q2_list, R2_list, n_components_list, "calibration.pdf", show)

    print()

    return best_pls1

