import argparse
from typing import NoReturn

import coseg
import yaml

config = yaml.safe_load(open("config.yml"))


def install_coseg(data_dir: str, ratio: float = 0.85, splits: str = None) -> NoReturn:
    """
    Downloads and installs the COSEG dataset.
    The script does the following:
        - Download the COSEG dataset
        - Split the data into a training and testing set
        - Transform the face labels to vertex labels

    :param data_dir: Directory where data will be stored and processed
    :param ratio: Percentage of data to be converted to training (remaining is test)
    """
    print("Downloading data...")
    coseg.download_data(args.data_dir)
    print("...done!")

    print("Splitting test and train data...")
    coseg.split_train_test(args.data_dir, args.ratio, args.splits)
    print("...done!")

    print("Converting face labels to vertex labels...")
    coseg.convert_labels(args.data_dir)
    print("...done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        default=config["data_dir"]["coseg"],
        required=False,
        help="Directory where data is stored",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.85,
        required=False,
        help="Ratio of train to test data",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default='preprocess/coseg/splits.yaml',
        required=False,
        help="yaml containing train/test splits",
    )
    args = parser.parse_args()

    install_coseg(args.data_dir, args.ratio, args.splits)
