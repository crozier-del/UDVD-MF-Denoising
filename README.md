# Unsupervised Deep Video Denoiser for Transmission Electron Microscopy
[![DOI](https://zenodo.org/badge/914030489.svg)](https://doi.org/10.5281/zenodo.14630448)

## Updates

### 2025.05.05
- More input type: `.npy`
- Support 4D dataset
- Adjustments to the blindspot algorithm: now also blind neighboring pixel by default
- More arguments

### 2025.03.31
- Fixed `environment.yaml` to ensure proper dependency installation and environment setup.

## Introduction
 This set of code is a fully unsupervised framework, namely **unsupervised deep video denoiser (UDVD)**, to train denoising models using exclusively real noisy data collected from a transmission electron microscope (TEM). The framework enables recovery of atomic-resolution information from TEM data, potentially improving the signal-to-noise ratio (SNR) by more than an order of magnitude.
 
 Assuming the data has minimal correlated noise, the denoiser will take a TEM dataset in `.tif` or `.npy` format collected from a direct electron detector and generate the denoised result as a `.npy` file, which can be further converted to other file formats. It is recommended to run this denoiser on high-performance computers (hpc). Check with hpc staff if necessary changes to installation and running commands are needed.

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
# all the arguments in [] are optional.
python denoise_mf.py --data path_to_file \
                    [--output-file path_to_output_file] [--save-model] 
                    [--num-epochs 50] [--batch-size 1] [--image-size 256] [--fourdim] 
                    [--include-neighbor] [--multiply 1]
```
### Arguments
Some arguments needs parameters. Replace the `$parameter` with the actual parameters. 
* **(required)**`--data $PATH_TO_FILE` : Full path to the `.tif` file containing the video to be denoised.
* *(new)*`--output-file $PATH_TO_OUTPUT_FILE` : Full path of the output file without filename extension. It will save at the same location with an extension of `_udvd_mf` by defualt.
* *(new)*`--fourdim` : Set if data is 4D.
* *(new)*`--include-neighbor` : Set if not blinding neighbors.
* *(new)*`--save-model` : Set if saving the model as a `.pth` file. The file name will be the same as the output file.
* `--num-epochs $NUM` Number of training epochs (default: 50).
* `--batch-size $NUM `: Number of images per batch for training (default: 1). Adjust based on available GPU memory.
* `--image-size $NUM`: Size of the square image patches used for training (default: 256).
* *(new)*`--multiply $NUM` : Multiply the data by an integer to manually normalize the data (default: 1).

### Examples

The example files can be downloaded from [here](https://www.dropbox.com/scl/fo/usoouapl9jd8uarwi7fkv/AOusqUYN-FeN7K-q1MqoCa0?rlkey=9evnykpkiadwwu4m5vl92omf4&st=jya48zgs&dl=0). There are two files in the folder: `PtCeO2_030303.tif` is the raw data, and `PtCeO2_030303_udvd_mf.tif` is the denoised result converted to `.tif` format.

To denoise the example data, run the following command:

```shell
python denoise_mf.py --data "PATH TO THE FILE/PtCeO2_030303.tif" 
```
Replacing the `PATH TO THE FILE` with the actual directory to the raw video file location. This line will denoise the TEM movieAfter the denoising process completed, the denoised result `PtCeO2_030303_udvd_mf.npy` can be found in the same directory as the input file.

## Citation

If you use this code, please cite our work: 

*Unsupervised Deep Video Denoising*\
D. Y. Sheth, S. Mohan, J. L. Vincent, R. Manzorro, P. A. Crozier, M. M. Khapra, E. P. Simoncelli, C. Fernandez-Granda; **Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)**, 2021, pp. 1759-1768\
[https://arxiv.org/abs/2011.15045](https://arxiv.org/abs/2011.15045)

*Evaluating Unsupervised Denoising Requires Unsupervised Metrics*\
A. Marcos Morales, M. Leibovich, S. Mohan, J. L. Vincent, P. Haluai, M. Tan, P. A. Crozier, C. Fernandez-Granda; **Proceedings of the 40th International Conference on Machine Learning (ICML)**, PMLR 2023 Vol. 202, pp. 23937-23957.\
[https://arxiv.org/abs/2210.05553](https://arxiv.org/abs/2210.05553)

## Support

If you encounter any issues or have questions about the project, please contact us at [CROZIER@asu.edu](mailto:CROZIER@asu.edu) or [ywan1240@asu.edu](mailto:ywan1240@asu.edu).

