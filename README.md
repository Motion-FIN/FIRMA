# Motion-Focused Interpolation Network(MoFIN): Integrating Region of Motion Loss and Self-Attention for Enhanced Video Frame Interpolation 

## Method Architecture

![framework](https://github.com/Motion-FIN/Motion-Focused-Interpolation-Network-MoFIN-/assets/150782727/ca3f34e5-03b1-4fd4-a013-774a3e7200ea)



## Requirements

1. The code has been developed with `torch1.12.1`.
2. Install other requirements as:
    ```bash
    pip install -r requirements.txt
    ```

## Download Datasets

Download [Vimeo90K dataset](http://toflow.csail.mit.edu/).

Download [SNU_FILM dataset](https://myungsub.github.io/CAIN/).

Download [UCF101 dataset](https://liuziwei7.github.io/projects/VoxelFlow).

Download [MiddleBury Other dataset](https://vision.middlebury.edu/flow/data/).

## Download Pretrained files

Download our [pretrained model-Setting1](https://drive.google.com/file/d/1ASviqlBU8VTN3WBTLINo93wr2mRW3-Yz/view?usp=sharing).
& [pretrained model-Setting4](https://drive.google.com/file/d/1NMH6y-F0TmE-e01tVlGKQOfhzoYMwYtV/view?usp=sharing).

## File Paths

Make your downloaded files structured like below:

```bash
.
├── configs
├── datas
├── datasets
│   ├── middlebury
│   │   ├── other-data
│   │   └── other-gt-interp
│   ├── snu_film
│   │   ├── test
│   │   ├── test-easy.txt
│   │   ├── test-extreme.txt
│   │   ├── test-hard.txt
│   │   └── test-medium.txt
│   ├── ucf101
│   │   ├── 1
│   │   ├── 1001
│   │   ...
│   │   ├── 981
│   │   └── 991
│   └── vimeo_triplet
│       ├── readme.txt
│       ├── sequences
│       ├── tri_testlist.txt
│       └── tri_trainlist.txt
├── pretrained
│   ├── Setting1.pth
│   └── Setting4.pth
├── experiments
├── losses
├── models
├── utils
├── validate
├── train.py
├── test.py
└── val.py
```
## Run

Run benchmarking by following commands:
```bash
python val.py --config configs/benchmarking/vimeo.yaml --gpu_id 0
python val.py --config configs/benchmarking/middlebury.yaml --gpu_id 0
python val.py --config configs/benchmarking/ucf101.yaml --gpu_id 0
python val.py --config configs/benchmarking/snu_film.yaml --gpu_id 0
```

To enable the augmented test (**"MoFIN-Set1-Aug"** & **"MoFIN-Set4-Aug"** in the paper), uncomment the `val_aug: [T,R]` line in the configuration files.

Augmentation Performance:

## Training

The model was trained on the Vimeo90K-triplet training split.

Run the following command for training:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port 9999 train.py --config configs/train.yaml
```

## Testing on a Custom Image Pair

First specify the path of the model weights in `configs/test.yaml`.

Then you can test the model on a customized image pair as:

```bash
python test.py --config configs/test.yaml --im0 <path to im0> --im1 <path to im1> --output_dir <path to output folder>
```
