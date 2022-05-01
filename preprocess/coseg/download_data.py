import argparse
import os
import shutil
from os.path import join

from requests import get


def wget(url: str) -> None:
    """
    Downloads the data given at ``url``

    :param url: URL from which to get data
    """
    with open(url.split("/")[-1], "wb") as f:
        f.write(get(url).content)


def download_data(data_dir: str) -> None:
    """
    Downloads COSEG data (meshes and ground truth).

    :param data_dir: Directory to download and store data into.
    """

    class_to_url = {
        "tele_aliens": [
            "http://irc.cs.sdu.edu.cn/~yunhai/public_html/ssl/data/Tele-aliens/shapes.zip",
            "http://irc.cs.sdu.edu.cn/~yunhai/public_html/ssl/data/Tele-aliens/gt.zip",
        ],
        "vases": [
            "http://irc.cs.sdu.edu.cn/~yunhai/public_html/ssl/data/Large-Vases/shapes.zip",
            "http://irc.cs.sdu.edu.cn/~yunhai/public_html/ssl/data/Large-Vases/gt.zip",
        ],
        "chairs": [
            "http://irc.cs.sdu.edu.cn/~yunhai/public_html/ssl/data/Large-Chairs/shapes.zip",
            "http://irc.cs.sdu.edu.cn/~yunhai/public_html/ssl/data/Large-Chairs/gt.zip",
        ],
    }

    for c in class_to_url.keys():
        class_dir = join(data_dir, c)
        os.makedirs(class_dir)
        wget(class_to_url[c][0])  # Download dataset
        wget(class_to_url[c][1])  # Download dataset
        shutil.unpack_archive("shapes.zip", class_dir)  # Unpack data into data_dir
        shutil.unpack_archive("gt.zip", class_dir)
        os.remove("shapes.zip")  # Remove .zips
        os.remove("gt.zip")


if __name__ == "__main__":
    """
    Runs download_data to download COSEG data
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="Directory where data is stored")
    args = parser.parse_args()
    download_data(args.data_dir)
