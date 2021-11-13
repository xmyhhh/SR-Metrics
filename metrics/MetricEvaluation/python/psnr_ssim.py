
import math

import cv2

import numpy as np


# copyed from data.util
def bgr2ycbcr(img, only_y=True):
    '''bgr version of rgb2ycbcr, following matlab version instead of opencv
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def calculate_psnr_ssim(img1, img2, crop_border=0):
    if crop_border == 0:
        cropped_img1 = img1
        cropped_img2 = img2
    else:
        cropped_img1 = img1[crop_border:-crop_border, crop_border:-crop_border]
        cropped_img2 = img2[crop_border:-crop_border, crop_border:-crop_border]
    psnr = calculate_psnr(cropped_img1 * 255, cropped_img2 * 255)
    ssim = calculate_ssim(cropped_img1 * 255, cropped_img2 * 255)

    if img2.shape[2] == 3:  # RGB image
        img1_y = bgr2ycbcr(img1, only_y=True)
        img2_y = bgr2ycbcr(img2, only_y=True)
        if crop_border == 0:
            cropped_img1_y = img1_y
            cropped_img2_y = img2_y
        else:
            cropped_img1_y = img1_y[crop_border:-crop_border, crop_border:-crop_border]
            cropped_img2_y = img2_y[crop_border:-crop_border, crop_border:-crop_border]
        psnr_y = calculate_psnr(cropped_img1_y * 255, cropped_img2_y * 255)
        ssim_y = calculate_ssim(cropped_img1_y * 255, cropped_img2_y * 255)
    else:
        psnr_y, ssim_y = 0, 0

    return psnr, ssim, psnr_y, ssim_y
