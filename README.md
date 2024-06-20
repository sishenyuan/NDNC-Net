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

The methods for generating the dataset for the NDNC-Net model, the specific working principles and the evaluation of the model's image restoration capabilities can all be found in the paper. The dataset generation method and the working principles of the NDNC-Net model are illustrated in the diagram below.

![image](shematic.png)

# Contents

## Oct Dataset

This folder includes the python and matlab files that can generate the synthetic dataset with accurate pairs of resampling distance variation vector (RDVV) and distorted images for further NDNC-Net training. The specific setup for this part are shown in the README.md file in the oct-dataset folder.

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

## Initialization

Obtain the correction net checkpoint from [Google Drive](https://drive.google.com/file/d/1MupEM5652VPwYeARrCFa971LdnXfhOCX), and create a new directory named `weights` and place the checkpoint within.

## Data Preparation

Dataset can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1H5xdALyERpqmABYI6VVqiCFCwWWD_ndI). You need to rename the folder to `images`(by default) and place it in the root directory.

To create the dynamics curves, run the following command:

```bash
python create_random.py --num_datasets 100
```

To create distorted images, run the following command:

```bash
python create_distorted.py --num_samples 100
```

## Initiation

Run

```bash
python oct_restore.py
```

to start inference. The results will be saved in the `./outputs`.
