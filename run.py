from pathlib import Path
import argparse
from sklearn.model_selection import train_test_split

from src.utils import get_data_df
from src.pretreatment import pretreat
from src.pls.calibration import calibrate


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

    # Perform data pretreatment
    X, y = pretreat(data_df, show=show)

    # Split the time series into train, valid, and test splits.
    valtest_size = 0.3 # Half of this is used for validation and the other half for testing
    X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, test_size=valtest_size, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, train_size=0.5, shuffle=False)

    # Calibration
    pls1 = calibrate(X_train, X_val, y_train, y_val, show=show)

    # TODO: Testing here


if __name__ == "__main__":
    main()
