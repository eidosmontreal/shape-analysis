import argparse
from os.path import join
from typing import NoReturn

import human_seg
import yaml

# config = yaml.safe_load(open("config.yml"))


def install_human_seg(data_dir: str) -> NoReturn:
    """
    Unzips and installs train/test splits of Human Segmentation.

    :param data_dir: Directory where data will be stored and processed
    """

    print("Unzipping data...")
    human_seg.unzip_data(args.data_dir)
    print("...done!")

    print("Splitting test/train data and converting face labels to vertex labels...")
    human_seg.split_train_test(join(args.data_dir, "HumanSegmentation"))
    print("...done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data", required=False, help="Directory where data is stored")

    args = parser.parse_args()

    install_human_seg(args.data_dir)
