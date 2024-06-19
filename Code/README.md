# Setup

Python dependencies:

- pytorch
- opencv
- sklearn
- skimage

We provide a `requirements.txt` including all of the above dependencies. To create a new conda environment and install the dependencies, run:

    ```bash
    conda create --name nrnet python=3.9
    conda activate nrnet
    pip install -r requirements.txt
    ```

## Initialization

Obtain the correction net checkpoint from [Google Drive](https://drive.google.com/file/d/1MupEM5652VPwYeARrCFa971LdnXfhOCX), and create a new directory named `weights` and place the checkpoint within.

## Data Preparation

You can use our provided data to test the model.

## Initiation

Run

    ```bash
    python oct_restore.py
    ```

to start inference. The results will be saved in the `./outputs`.
