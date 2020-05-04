import math
import numpy as np
import cv2
import os.path as osp
from glob import glob
from scipy.misc import imread


####################
# patch
####################


def split_img(img, patch_size=128, stride=100, scale=4):
    h, w, _ = img.shape
    patches = []
    grids = []
    for ind_row in range(0, h - (patch_size - stride), stride):
        for ind_col in range(0, w - (patch_size - stride), stride):
            patch = img[ind_row:ind_row + patch_size, ind_col:ind_col + patch_size, :]
            if patch.shape != (patch_size, patch_size, 3):
                patch = np.pad(patch,
                                ((0, patch_size - patch.shape[0]), (0, patch_size - patch.shape[1]), (0, 0)),
                                'reflect')
            patches.append(patch)
            grids.append((ind_row * scale, ind_col * scale, patch_size * scale))
    grids = np.stack(grids, axis=0)
    patches = np.stack(patches, axis=0)
    return patches, grids


def recon_patch(img_input, grids, patch_files_1, patch_files_2=None, scale=4):
    h, w, _ = img_input.shape

    patch_size = grids[0, 2]
    h_l, w_l = grids[-1, 0] + patch_size, grids[-1, 1] + patch_size
    out_large_1 = np.zeros((h_l, w_l, 3), dtype=np.float32)
    if patch_files_2 is not None:
        out_large_2 = np.copy(out_large_1)
    counter = np.zeros_like(out_large_1, dtype=np.float32)

    for idx in range(len(grids)):
        out_large_1[
        grids[idx, 0]:grids[idx, 0] + patch_size,
        grids[idx, 1]:grids[idx, 1] + patch_size, :] += imread(patch_files_1[idx], mode='RGB').astype(np.float32)

        if patch_files_2 is not None:
            out_large_2[
            grids[idx, 0]:grids[idx, 0] + patch_size,
            grids[idx, 1]:grids[idx, 1] + patch_size, :] += imread(patch_files_2[idx], mode='RGB').astype(np.float32)

        counter[
        grids[idx, 0]:grids[idx, 0] + patch_size,
        grids[idx, 1]:grids[idx, 1] + patch_size, :] += 1

    out_large_1 /= counter
    out_1 = out_large_1[:h * scale, :w * scale, :]
    if patch_files_2 is not None:
        out_large_2 /= counter
        out_2 = out_large_2[:h * scale, :w * scale, :]
    else:
        out_2 = None
    return out_1, out_2


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
