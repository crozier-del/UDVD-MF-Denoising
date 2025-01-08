# video_denoisers


## Denoisers:

```
* denoise.py: UDVD multi-frame
* denoise_sf.py: UDVD single-frame
* denoise_N2N.py: Neighbor2Neighbor
* denoise_N2S.py: Noise2Self
```

## Configurations
In order to run the denoiser codes with different configurations the following arguments can be provided:

    * --data: this is the only argument that has to be always provided. It must be the full path to the .tif file that contains the video to be denoised.
    * --num-epochs: number of epochs through which the network must be trained. If it is not provided, the default value is 50 (500 for N2N). If the results are still noisy, providing a bigger value here may improve them.
    * --batch-size: number of images per batch on the training. The default value is 2 to keep CUDA memory usage low. It can be lowered to 1 if the training cannot run due to the lack of memory or increased if the resources used are higher.
    * --image-size: size of the (square) image crops used during the training. The default size is 256 (512 for N2N) and should be kept at this size for (1024, 1024) size frames
