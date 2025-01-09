# Unsupervised Deep Video Denoiser


## Denoisers:

```
* denoise.py: UDVD multi-frame
* denoise_sf.py: UDVD single-frame
* denoise_N2N.py: Neighbor2Neighbor
* denoise_N2S.py: Noise2Self
```

## Usage
### Installation
```shell
git clone https://github.com/crozier-del/UDVD-MF-Denoising
cd UDVD-MF-Denoising
pip install -r requirements.txt

```

### Command
```shell
python denoise.py\
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
python denoise.py --data ./examples/PtCeO2_6.tif # denoise with the UDVD model
```

### Citation

If you use this code, please cite our work:
## Configurations

In order to run the denoiser codes with different configurations the following arguments can be provided:

    * --data: this is the only argument that has to be always provided. It must be the full path to the .tif file that contains the video to be denoised.
    * --num-epochs: number of epochs through which the network must be trained. If it is not provided, the default value is 50 (500 for N2N). If the results are still noisy, providing a bigger value here may improve them.
    * --batch-size: number of images per batch on the training. The default value is 2 to keep CUDA memory usage low. It can be lowered to 1 if the training cannot run due to the lack of memory or increased if the resources used are higher.
    * --image-size: size of the (square) image crops used during the training. The default size is 256 (512 for N2N) and should be kept at this size for (1024, 1024) size frames
