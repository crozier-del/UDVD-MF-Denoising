# Unsupervised Deep Video Denoiser


## Introduction



## Usage
### Installation
```shell
git clone https://github.com/crozier-del/UDVD-MF-Denoising
cd UDVD-MF-Denoising
pip install -r requirements.txt

```

### Command
```shell
python denoise_mf.py\
     --data path_to_tiff_file 
     --model UDVD 
     --num-epochs 50
     --batch-size 1
     --image-size 256
```
### Arguments
* `data` (required): Full path to the `.tif` or `.npy` file containing the video to be denoised. The `.npy` file should have shape of (frames, x, y).
* `model`: Model name. Options are `UDVD`, `N2N`, `N2S`, or `UDVD_sf`. Default is `UDVD`.
* `num-epochs` Number of training epochs(default: 50).
* `batch-size`: Number of images per batch for training (default: 1). Adjust based on available memory.
* `image-size`: Size of the square image patches used for training (default: 256). For N2N, a larger size, such as 512, is recommended to compensate the downsampling step.
* `save-format`: Select the format of denoised file (`tif` or `npy`). Default is `tif`.

### Example

The provided `PtCeO2_6.tif` video can be denoised by running the following commands:

```shell
python denoise_mf.py --data ./examples/PtCeO2_6.tif # denoise with the UDVD model
```

### Citation

If you use this code, please cite our work:
