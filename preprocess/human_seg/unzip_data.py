import argparse
import os
import shutil
from os.path import join

data_url = "https://www.dropbox.com/sh/cnyccu3vtuhq1ii/AADgGIN6rKbvWzv0Sh-Kr417a?dl=0&preview=human_benchmark_sig_17.zip"


def unzip_data(data_dir: str) -> None:
    """
    Downloads HumanSegmentation data and places it in ``data_dir``.

    :param data_dir: Directory where data will be soted
    """
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(join(data_dir, "HumanSegmentation"), exist_ok=True)
    if not os.path.exists(join(data_dir, "human_benchmark_sig_17.zip")):
        raise RuntimeError(f"Please download human_benchmark_sig_17.zip from {data_url} and move it to {data_dir}.")

    shutil.unpack_archive(join(data_dir, "human_benchmark_sig_17.zip"), data_dir)
    os.rename(join(data_dir, "sig17_seg_benchmark"), join(data_dir, "HumanSegmentation"))


if __name__ == "__main__":
    """
    Runs unzip_data to unzip HumanSegmentation data
    """
    parser = argparser.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="Directory where data is stored")
    args = parser.parse_args()

    unzip_data(args.data_dir)
