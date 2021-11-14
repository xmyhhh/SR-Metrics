# Super-Resolution Performance Evaluation Code
The project covers common metrics for super-resolution performance evaluation.

## Metrics support
The scripts will calculate the values of the following evaluation metrics: 
[`'MA'`](https://github.com/chaoma99/sr-metric), 
[`'NIQE'`](https://github.com/csjunxu/Bovik_NIQE_SPL2013), 
[`'PI'`](https://github.com/roimehrez/PIRM2018), `'PSNR'`, `'PSNR-Y'`,
[`'SSIM'`](https://ece.uwaterloo.ca/~z70wang/research/ssim), `'SSIM-Y'`
[`'BRISQUE'`](http://live.ece.utexas.edu/research/quality/),
[`'LPIPS'`](https://github.com/richzhang/PerceptualSimilarity). 


## Highlights
- Breakpoint continuation support : The program can continue from where it was last interrupted by using `.xlsx` file
- Parallel computing  support : The Programs can be re-scaled to take advantage of multi-core performance by using python`ThreadPoolExecutor`
- Both RGB and YCbCr color space support 

## Dependencies
- Python 3 
- [PyTorch >= 1.0](https://pytorch.org/)
- Matlab (IMAGE QUALITY TOOLBOX required)

## Instructions for use this code
Please ref [BLIND IMAGE QUALITY TOOLBOX](./metrics/README.md "BLIND IMAGE QUALITY TOOLBOX")

## Reference
The code is based on [SPSR](https://github.com/Maclory/SPSR)  and [BIQT](https://github.com/dsoellinger/blind_image_quality_toolbox). 
