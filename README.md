# Unsupervised Deep Video Denoiser


## Introduction



## Usage
### Installation
```shell
git clone https://github.com/crozier-del/UDVD-MF-Denoising
cd UDVD-MF-Denoising
conda env create -n denoise-HDR -f environment.yaml
```

### Running
```shell
conda activate denoise-HDR
python denoise_mf.py\
     --data path_to_tiff_file  
     --num-epochs 50
     --batch-size 1
     --image-size 256
```
### Arguments
* `data` (required): Full path to the `.tif` file containing the video to be denoised.
* `num-epochs` Number of training epochs(default: 50).
* `batch-size`: Number of images per batch for training (default: 1). Adjust based on available GPU memory.
* `image-size`: Size of the square image patches used for training (default: 256).

### Example

The provided `PtCeO2_6.tif` video can be denoised by running the following commands:

```shell
python denoise_mf.py --data ./examples/PtCeO2_6.tif --num-epochs 50 
```

### Citation

If you use this code, please cite our work:
