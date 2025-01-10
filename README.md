# Unsupervised Deep Video Denoiser for Transmission Electron Microscopy


## Introduction
 This set of code is a fully unsupervised framework, namely **unsupervised deep video denoiser (UDVD)**, to train denoising models using exclusively real noisy data collected from a transmission electron microscope (TEM). The framework enables recovery of atomic-resolution information from TEM data, improving the signal-to-noise ratio (SNR) by a factor of 40 at a spatial resolution of 1 Å and time resolution near 10 ms.
 
 Assuming the data has minimal correlated noise, the denoiser will take a TEM movie in `.tif` format collected from a direct electron detector and generate the denoised result as a `.npy` file, which can be further converted to other file formats. It is recommended to run this denoiser on high-performance computers (hpc).

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
* `data` **(required)**: Full path to the `.tif` file containing the video to be denoised.
* `num-epochs` Number of training epochs (default: 50).
* `batch-size`: Number of images per batch for training (default: 1). Adjust based on available GPU memory.
* `image-size`: Size of the square image patches used for training (default: 256).

### Example

The provided `PtCeO2_6.tif` video can be denoised by running the following commands:

```shell
python denoise_mf.py --data "./examples/PtCeO2_6.tif" 
```

### Citation

If you use this code, please cite our work: https://doi.org/10.48550/arXiv.2011.15045
