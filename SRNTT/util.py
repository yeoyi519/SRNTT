import math
import numpy as np
import cv2


####################
# metric
####################


def calculate_psnr(img1, img2, bits=8):
    # img1 and img2 have range [0, 2**bits-1]
    img1 = img1.astype(np.float64) # 数据类型很重要
    img2 = img2.astype(np.float64) # uint8 和 float32, float64 的差异明显
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10((2**bits-1) / math.sqrt(mse))


def ssim(img1, img2, bits=8):
    # img1 and img2 have range [0, 2**bits-1]
    C1 = (0.01 * (2**bits-1))**2
    C2 = (0.03 * (2**bits-1))**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    # filter2D 可以支持 (H,W,C) 的三维输入, 对每一通道做二维卷积
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2, bits=8):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 2**bits-1]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2 or img1.ndim == 3:
        return ssim(img1, img2, bits)
    else:
        raise ValueError('Wrong input image dimensions.')
