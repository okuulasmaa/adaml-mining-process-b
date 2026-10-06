from pathlib import Path
import argparse

from src.utils import get_data_df
from src.pretreatment import pretreat


def build_parser():

    parser = argparse.ArgumentParser(description="Mining process quality estimation")
    parser.add_argument("-d", "--data", type=Path, help="Path to the data CSV")
    parser.add_argument("-s", "--show", action="store_true", help="Show figures")

    return parser


def main():

    parser = build_parser()
    args = parser.parse_args()
    data_path = args.data
    show = args.show

    # Import the data from the CSV file
    data_df = get_data_df(data_path)

    X, y = pretreat(data_df, show=show)


if __name__ == "__main__":
    main()
