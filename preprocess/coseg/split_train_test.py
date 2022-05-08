import argparse
import os
import shutil
from os.path import join

import yaml

config = yaml.safe_load(open("config.yml"))


def split_train_test(data_dir: str, ratio: float = 0.85, splits: str = None) -> None:
    """
    Splits the COSEG data into a training and testing set with the set ``ratio``.
    The script is to be used with ``downlod_coseg.sh`` to download and process the datatset.

    Note that text files detailing the train/test splits are also produced.

    :param data_dir: Directory where data is stored
    :param ratio: (optional) Ratio of train to test data
    :param splits: (optional) Name of .yaml file containing train/test splits
    """
    if splits is None:
        splits_index = {}
        for category in config["classes"]["coseg"]:
            splits_index[category] = {}
            for mode in ["train", "test"]:
                splits_index[category][mode] = []
    else:
        splits_index = yaml.safe_load(open(splits))
        print("\t Using predefined train/test splits.")

    for category in config["classes"]["coseg"]:
        for mode in ["train", "test"]:
            for data in ["gt", "shapes"]:
                os.makedirs(join(data_dir, mode, category, data))
            if splits is not None:
                for idx in splits_index[category][mode]:
                    f = str(idx) + ".off"
                    g = str(idx) + ".seg"
                    old_shape_path = join(data_dir, category, "shapes", f)
                    old_gt_path = join(data_dir, category, "gt", g)

                    new_shape_path = join(data_dir, mode, category, "shapes", f)
                    new_gt_path = join(data_dir, mode, category, "gt", g)

                    shutil.copyfile(old_shape_path, new_shape_path)
                    shutil.copyfile(old_gt_path, new_gt_path)
                return

        files = sorted(os.listdir(join(data_dir, category, "shapes")))
        num_files = len(files)

        for i, f in enumerate(files):
            mode = "train" if (i + 1) / num_files <= ratio else "test"
            idx = f.split(".")[0]
            g = idx + ".seg"
            splits_index[category][mode] += [int(idx)]

            old_shape_path = join(data_dir, category, "shapes", f)
            old_gt_path = join(data_dir, category, "gt", g)

            new_shape_path = join(data_dir, mode, category, "shapes", f)
            new_gt_path = join(data_dir, mode, category, "gt", g)

            shutil.copyfile(old_shape_path, new_shape_path)
            shutil.copyfile(old_gt_path, new_gt_path)

    with open(join(data_dir, "splits.yaml"), "w") as f:
        yaml.dump(splits_index, f)


def main() -> None:
    """
    Run ``split_train_test`` with given ``data_dir`` and ``ratio``.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="Directory where data is stored")
    parser.add_argument("--ratio", type=float, required=False, default=0.85, help="Ratio of test and train data")
    parser.add_argument("--splits", type=str, required=False, default=None, help="yaml containing train/test splits")
    args = parser.parse_args()

    split_train_test(args.data_dir, args.ratio)


if __name__ == "__main__":
    main()
