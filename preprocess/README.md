# Data Preprocessing
This folder contains links and scripts for downloading and processing the datasets used in our experiments.

The datasets used for segmentation were [Shape COSEG](http://irc.cs.sdu.edu.cn/~yunhai/public_html/ssl/ssd.htm) and Human Body Segmentation.

The script used for downsampling can be found in `meshutils/mesh_operations.py`, and is taken from [this repository](https://github.com/pixelite1201/pytorch_coma). 

**All scripts and commands should be run from the parent directory.**

## FAUST
We rely on the [FAUST dataset module](https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/datasets/faust.py) in *PyTorch Geometric* for handling the FAUST dataset.

Download the dataset from [here](http://faust.is.tue.mpg.de/) and move it to `../data` (which may need to be manually created).

## COSEG
To download and install the COSEG dataset, run the following command, which will download, split, and process labels of the COSEG dataset (respectively):
```bash
$ python install_coseg.py --data-dir '../data' --ratio 0.85
```
`--data-dir` indicates where to store the data and `--ratio` indicates ratio of train to test data to be split.

To downsample the meshes run `python decimate_coseg.py target_vertices`, where `target_vertices` indicates roughly the number of vertices to downsample each mesh until.

## Human Body Segmentation
After downloding the Human Body Segmentation dataset from this [link]('https://www.dropbox.com/sh/cnyccu3vtuhq1ii/AADgGIN6rKbvWzv0Sh-Kr417a?dl=0&preview=human_benchmark_sig_17.zip') and placing it in the `data/` directory, from the command line run `python install_human_seg.py` and then `python decimate_human_seg.py 1000`. Again, the `1000` refers to the number of vertices to roughly downsample each mesh until.
