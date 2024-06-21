# Table of Contents

- [General info](#general-info)
- [Methods pipeline](#methods-pipeline)
- [Contents](#contents)
- [Getting started](#getting-started)

# General Info

This Git repository contains python codes for implementing the NDNC-Net model. The NDNC-Net is a deep-learning approach which integrates the rotation dynamics model of the rotatable diametrically magnetized cylinder permanent magnet (RDPM) and is trained using synthetic data.

The mainly capability of the code is to accurately correct the rotational distortions in Optical Coherence Tomography (OCT) images which were sampled in the rat esophagus and the mouse colon. The current version includes the following models:

1. **Nonuniform Rotational Distortion Detection Network** (NDNet)

2. **Nonuniform Rotational Distortion Correction Network** (NCNet)

# Methods Pipeline

The methods for generating the dataset for the NDNC-Net model, the specific working principles, and the evaluation of the model's image restoration capabilities are all detailed in the paper. The diagram below illustrates the dataset generation method and the working principles of the NDNC-Net model.

![image](shematic.png)

# Contents

## Resample

This folder includes the python scripts that can generate the synthetic dataset with accurate pairs of resampling distance variation vector (RDVV) and distorted images for further NDNC-Net training.

## NDNet

This folder includes the python scripts that can train the NDNet model.

## NCNet

`ncnet.py` is the python script for inference of the NCNet model.

## OCT Restore

`oct_restore.py` is the python script that can restore the distorted images using the trained NDNet and NCNet models.

# Getting Started

## Setup

Python dependencies:

- pytorch
- opencv
- sklearn
- skimage

We provide a `requirements.txt` including all of the above dependencies. To create a new conda environment and install the dependencies, run:

```bash
conda create --name ndnc-net python=3.9
conda activate ndnc-net
pip install -r requirements.txt
```

## Initialize

Obtain the correction net checkpoint from [Google Drive](https://drive.google.com/file/d/1MupEM5652VPwYeARrCFa971LdnXfhOCX), and create a new directory named `weights` and place the checkpoint within.

## Prepare the Dataset

The dataset can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1H5xdALyERpqmABYI6VVqiCFCwWWD_ndI). You need to rename the folder to `images`(by default) and place it in the root directory.

To create the dynamics curves, run the following command:

```bash
python create_random.py --num_datasets 100 # replace 100 with the number of curves you want to generate
```

To create distorted images, run the following command:

```bash
python resample.py --num_samples 100 # replace 100 with the number of samples you want to generate
```

## Train NDNet

We provide a pre-trained NDNet model `best.pt` which can be found in the `src/yolo/train/weights` directory.

To train your own NDNet, prepare the dataset and `dataset.yaml` file, and run:

```bash
python train.py
```

After training, the best model will be saved in the `weights` directory.

If you want to evaluate the model, run:

```bash
python detect.py
```

Results will be saved in the `./runs/detect` directory.

## Inverse Resampling

Run

```bash
python oct_restore.py
```

to start inference. Results will be saved in the `./outputs`.

## Demonstration

A synthetic dataset for testing is available for download on [Google Drive](https://drive.google.com/drive/folders/1VD00LOpnTB4Pj4NNuMb9lZRDWIK5FiY2), which includes original distortion-free samples and the corresponding synthetic distorted images. This dataset can be used to compare the OCT images after correction with the original distortion-free images.


