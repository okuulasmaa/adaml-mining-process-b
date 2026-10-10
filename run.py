from pathlib import Path
import argparse
import numpy as np

from src.utils import get_data_df, split_data, residual_sum_of_squares
from src.exploartion import explore_data
from src.analysis import zscore
from src.pretreatment import pretreat_with_undersampling
from src.pls.calibration import calibrate
from src.visualizations import plot_test_pred_vs_real, plot_residuals


def build_parser():

    parser = argparse.ArgumentParser(description="Mining process quality estimation")
    parser.add_argument("-d", "--data", type=Path, help="Path to the data CSV")
    parser.add_argument("--show", action="store_true", help="Show figures.")

    return parser


def main():

    parser = build_parser()
    args = parser.parse_args()
    data_path = args.data
    show = args.show

    # Import the data from the CSV file
    data_df = get_data_df(data_path)

    # Exploratory analysis
    explore_data(data_df, show)

    # Perform data pretreatment
    X, y = pretreat_with_undersampling(data_df, show=show)

    # Split the time series into train, valid, and test splits.
    train_size = 0.7 
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, train_size)

    dates_test =  X_test[:,0]

    # Standardizing splits using z-score
    mean_X_train, mean_y_train = np.mean(X_train, axis=0), np.mean(y_train, axis=0)
    std_X_train, std_y_train = np.std(X_train, axis=0), np.std(y_train, axis=0)
    Xc_train, yc_train = zscore(X_train, mean_X_train, std_X_train), zscore(y_train, mean_y_train, std_y_train)
    Xc_val, yc_val = zscore(X_val, mean_X_train, std_X_train), zscore(y_val, mean_y_train, std_y_train)
    Xc_test, yc_test = zscore(X_test, mean_X_train, std_X_train), zscore(y_test, mean_y_train, std_y_train)

    # Calibration
    pls1 = calibrate(Xc_train, Xc_val, yc_train, yc_val, show=show)

    # TODO: Testing here
    print()
    print("Test begins")
    yc_test_pred = pls1.predict(Xc_test)

    sse = residual_sum_of_squares(yc_test,yc_test_pred)
    print(f"Sum of squared errors is {np.round(sse,3)}")

    plot_test_pred_vs_real(yc_test_pred,yc_test,dates_test,True)




    return 



if __name__ == "__main__":
    main()
