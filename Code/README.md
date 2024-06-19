
## Usage

### Requirements

* pytorch
* opencv
* sklearn
* skimage
* tqdm


### Initialization

Down the checkpoint of correction net from [Checkpoints](https://drive.google.com/file/d/1MupEM5652VPwYeARrCFa971LdnXfhOCX/view?usp=sharing), create a new folder named ```MegOCT/weights/``` and place the checkpoint in it.

### Data Preparation
We have prepared some demonstration data and you can down them from. Then just place the data in this folder.

## Start
Just run ```python oct_restore.py``` to start inferences. By default, the outputs are saved in ```./outputs```.
