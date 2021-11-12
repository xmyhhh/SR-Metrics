# Super-Resolution Performance Evaluation Code
The project covers common metrics for super-resolution performance evaluation.

## Metrics support
The scripts will calculate the values of the following evaluation metrics: 
[`'MA'`](https://github.com/chaoma99/sr-metric), 
[`'NIQE'`](https://github.com/csjunxu/Bovik_NIQE_SPL2013), 
[`'PI'`](https://github.com/roimehrez/PIRM2018), `'PSNR'`, 
[`'BRISQUE'`](http://live.ece.utexas.edu/research/quality/),
[`'SSIM'`](https://ece.uwaterloo.ca/~z70wang/research/ssim), `'MSE'`, `'RMSE'`, `'MAE'`, 
[`'LPIPS'`](https://github.com/richzhang/PerceptualSimilarity). 
Note that the `'SSIM'` values are calculated by `ssim.m`, the matlab code including the suggested downsampling process available in this [link](https://ece.uwaterloo.ca/~z70wang/research/ssim). 


## Highlights
- Breakpoint continuation support : The program can continue from where it was last interrupted by using `.xlsx` file
- Parallel computing  support : Programs can be re-scaled to take advantage of multi-core performance by using python`ThreadPoolExecutor`

## Dependencies
- Python 3 
- [PyTorch >= 1.0](https://pytorch.org/)
- Matlab 

## Instructions for use this code
Please ref [BLIND IMAGE QUALITY TOOLBOX](./metrics/README.md "BLIND IMAGE QUALITY TOOLBOX")

## Reference

The code is based on [SPSR](https://github.com/Maclory/SPSR)  and [BIQT](https://github.com/dsoellinger/blind_image_quality_toolbox). 
