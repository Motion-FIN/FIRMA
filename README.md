# Motion-Focused Interpolation Network(MoFIN): Integrating Region of Motion Loss and Self-Attention for Enhanced Video Frame Interpolation 

## Overview
In this paper, we present an innovative video frame interpolation approach that uniquely integrates Region of Motion(RoM) loss and self-attention scores.
This is the first instance of incorporating RoM loss in video frame interpolation, enabling our model to concentrate on key frame areas crucial for interpolation. This results in notably improved accuracy, particularly in videos with complex and non-linear object movements.
Additionally, our model employs self-attention scores on the features extracted from the Basic encoder and Contextnet, directing focus to specific motion areas in the frame for more accurate predictions.


<img src ="https://github.com/Motion-FIN/Motion-Focused-Interpolation-Network-MoFIN-/assets/150782727/ca3f34e5-03b1-4fd4-a013-774a3e7200ea" width="2000" height="400"/>

## Example of Demo

### Golf swing
<p float="left">
  <img src=figs/golf_Original.gif width=340 alt="Original" /> 
  <img src=figs/golf_MoFIN.gif width=340 alt="MoFIN" />
</p>

### Play on the soccer field
<p float="left">
  <img src=figs/sports_Original.gif width=340 alt="Original" /> 
  <img src=figs/sports_MoFIN.gif width=340 alt="MoFIN" />
</p>

## Requirements
- Pytorch 1.12.1
- python 3.11.3


## Installation

Download repository:
```bash
git clone https://github.com/Motion-FIN/Motion-Focused-Interpolation-Network-MoFIN-.git
```

Create conda environment:
```bash
conda create -n MoFIN python=3.11.3
conda activate MoFIN
pip install -r requirements.txt
```


    
## Download Pre-trained Models

Download pretrained models [Setting1](https://drive.google.com/file/d/1ASviqlBU8VTN3WBTLINo93wr2mRW3-Yz/view?usp=sharing).
& [Setting4](https://drive.google.com/file/d/1NMH6y-F0TmE-e01tVlGKQOfhzoYMwYtV/view?usp=sharing).


## Download Datasets

Download [Vimeo90K dataset](http://toflow.csail.mit.edu/).

Download [SNU_FILM dataset](https://myungsub.github.io/CAIN/).

Download [UCF101 dataset](https://liuziwei7.github.io/projects/VoxelFlow).

Download [MiddleBury Other dataset](https://vision.middlebury.edu/flow/data/).


## File Paths

The dataset folder names should be lower-case and structured as follows:

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

## Quick Usage (Testing on your pair of frames)

First specify the path of the model weights in `configs/test.yaml`.

Generate an intermediate frame on your pair of frames:

```bash
python test.py --config configs/test.yaml --im0 <path to im0> --im1 <path to im1> --output_dir <path to output folder>
```


## Evaluation

Run benchmarking by following commands:
```bash
python val.py --config configs/benchmarking/vimeo.yaml --gpu_id 0
python val.py --config configs/benchmarking/middlebury.yaml --gpu_id 0
python val.py --config configs/benchmarking/ucf101.yaml --gpu_id 0
python val.py --config configs/benchmarking/snu_film.yaml --gpu_id 0
```

To enable the augmented test (**"MoFIN-Set1-Aug"** & **"MoFIN-Set4-Aug"** in the paper), uncomment the `val_aug: [T,R]` line in the configuration files.


## Training

Run the following command for training:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port 9999 train.py --config configs/train.yaml
```

