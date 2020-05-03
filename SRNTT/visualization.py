import os.path as osp
import math
import logging

import numpy as np
import matplotlib.pyplot as plt


def visualize_feature(feature, tag, prefix, save_dir, data_format='channel_first',
                    every_image=64, cmap_level='global', vmin=None, vmax=None,
                    outlier=None, vmean=None, vstd=None, scale=0.01, verbose=1):
    """
        可视化单个特征的每个通道, 汇总在数张图中
    feature: numpy array, (B, C, H, W), B = 1
    tag: feature type
    prefix: before tag in save name
    data_format:
        channel_first, (B, C, H, W), B = 1
        channel_last, (B, H, W, C), B = 1
    every_image: every_image channels in single image
    cmap_level:
        global, all features in model, need to specify vmin and vmax, 查看模型整个周期特征间的演变
        local, current feature, 查看当前特征中最显著的特征图(H, W)
        self, single channel, 查看每个特征图(H, W)中最显著的部分
    outlier: 异常值(过小或过大)处理方式
        None, 不处理, 以 vmin and vmax 为边界
        Gaussian, 3 sigma 原则, cmap_level=global 时要指定 vmean and vstd
    scale: setting figure size to (W * scale * np.sqrt(every_image), H * scale * np.sqrt(every_image)) for visualization
    """
    feature = feature.copy().squeeze()
    if feature.ndim != 3:
        raise "Only support 3D feature (C, H, W)/(H, W, C)!"
    if data_format == 'channel_last':
        feature = np.transpose(feature, [2, 0, 1])
    C, H, W = feature.shape

    if verbose:
        logger = logging.getLogger('base')
    channel_mean = feature.mean(axis=(1, 2))
    channel_std = feature.std(axis=(1, 2))

    if cmap_level == 'global':
        if outlier == 'Gaussian':
            assert vmean != None and vstd != None, "Please specify vmean and vstd!"
            vmin = vmean - 3 * vstd
            vmax = vmean + 3 * vstd
            feature.clip(vmin, vmax)
        else:
            assert vmin != None and vmax != None, "Please specify vmin and vmax!"
    elif cmap_level == 'local':
        if outlier == 'Gaussian':
            vmean = feature.mean()
            vstd = feature.std()
            vmin = vmean - 3 * vstd
            vmax = vmean + 3 * vstd
            feature.clip(vmin, vmax)
        else:
            vmin = feature.min()
            vmax = feature.max()
    if cmap_level in ['global', 'local']:
        delta = (vmax - vmin) * 0.1
        vmin = vmin - delta # cmap min value
        vmax = vmax + delta # cmap max value

    n_row = n_col = int(np.sqrt(every_image))
    assert n_row * n_col == every_image
    n_img = int(np.ceil(C/(n_row*n_col)))
    for k in range(n_img):
        cur_feats = feature[k*every_image:min((k+1)*every_image, C)]
        cur_means = channel_mean[k*every_image:min((k+1)*every_image, C)]
        cur_stds = channel_std[k*every_image:min((k+1)*every_image, C)]

        cur_channels = cur_feats.shape[0]
        cur_n_row = int(np.ceil(cur_channels/n_col))
        plt.figure(figsize=(W * scale * n_col, H * scale * cur_n_row))
        for i in range(cur_n_row):
            for j in range(n_col):
                channel_id = i * n_col + j
                if channel_id >= cur_channels:
                    break

                plt.subplot(cur_n_row, n_col, channel_id + 1)

                if cmap_level == 'self':
                    if outlier == 'Gaussian':
                        vmean = cur_feats[i, j].mean()
                        vstd = cur_feats[i, j].std()
                        vmin = vmean - 3 * vstd
                        vmax = vmean + 3 * vstd
                        cur_feats[i, j].clip(vmin, vmax)
                    else:
                        vmin = cur_feats[i, j].min()
                        vmax = cur_feats[i, j].max()
                    delta = (vmax - vmin) * 0.1
                    vmin = vmin - delta # cmap min value
                    vmax = vmax + delta # cmap max value

                plt.imshow(cur_feats[channel_id], cmap='rainbow', vmin=vmin, vmax=vmax)
                plt.colorbar()
                plt.xlabel('Mean {:+.3f}, Std {:+.3f}'.format(cur_means[channel_id], cur_stds[channel_id]), fontsize='small')
                plt.title('Channel {}'.format(k*every_image+channel_id), fontsize='small')
                # plt.axis('off')

        plt.tight_layout(1.0)
        img_name = '{}_{}_{}_{}_C[{},{}).png'.format(prefix, tag, cmap_level, outlier, k*every_image, min((k+1)*every_image, C))
        img_path = osp.join(save_dir, img_name)
        plt.savefig(img_path)
        if verbose:
            logger.info('{} saved.'.format(img_name))

    plt.close('all')


def visualize_channel(feature, tag, prefix, save_dir, data_format='channel_first', verbose=1):
    """
        原尺寸保存单个特征的每个通道
    feature: numpy array, (B, C, H, W) or (B, N, C, H, W), B = 1
    tag: feature type
    prefix: before tag in save name
    data_format:
        channel_first, (B, C, H, W) or (B, N, C, H, W), B = 1
        channel_last, (B, H, W, C) or (B, N, H, W, C), B = 1
    """
    if verbose:
        logger = logging.getLogger('base')

    feature = feature.copy().squeeze()
    if feature.ndim == 3:
        if data_format == 'channel_last':
            feature = np.transpose(feature, [2, 0, 1])
        C, H, W = feature.shape
        for k in range(C):
            img_name = '{}_{}_C{}.png'.format(prefix, tag, k)
            img_path = osp.join(save_dir, img_name)
            plt.imsave(img_path, feature[k], cmap='rainbow')
            # if verbose:
            #     logger.info('{} saved.'.format(img_name))
    elif feature.ndim == 4:
        if data_format == 'channel_last':
            feature = np.transpose(feature, [0, 3, 1, 2])
        N, C, H, W = feature.shape
        for i in range(N):
            for j in range(C):
                img_name = '{}_{}_N{}_C{}.png'.format(prefix, tag, i, j)
                img_path = osp.join(save_dir, img_name)
                plt.imsave(img_path, feature[i, j], cmap='rainbow')
                # if verbose:
                #     logger.info('{} saved.'.format(img_name))
    else:
        raise "Only support 3D feature (C, H, W)/(H, W, C) or 4D feature (N, C, H, W)/(N, H, W, C)!"

    if verbose:
        logger.info('{}, {} channels saved.'.format(prefix, tag))


def visualize_weight(weight, tag, save_dir, data_format='channel_first', scale=1, verbose=1):
    """
        可视化权重
    weight: numpy array, (Co, Ci, kH, kW)
    tag: weight name
    data_format:
        channel_first, (Co, Ci, kH, kW)
        channel_last, (kH, kW, Ci, Co)
    scale: setting figure size
    """
    if verbose:
        logger = logging.getLogger('base')

    weight = weight.copy().squeeze()
    if weight.ndim == 4:
        if data_format == 'channel_last':
            weight = np.transpose(weight, [3, 2, 0, 1])
        Co, Ci, kH, kW = weight.shape
    elif weight.ndim == 2: # kH=kW=1
        if verbose:
            logger.warning("Won't visualize weight of {} with shape {}, because it's kernel size is 1.".format(tag, weight.shape))
        return
    else:
        raise "Only support 4D weight (Co, Ci, kH, kW)/(kH, kW, Ci, Co)!"

    channel_mean = weight.mean(axis=(2, 3))
    channel_std = weight.std(axis=(2, 3))

    # 一幅图至多64张子图
    if Co > 8:
        n_row = 8
        n_img_row = int(np.ceil(Co / 8))
    else:
        n_row, n_img_row = Co, 1
    if Ci > 8:
        n_col = 8
        n_img_col = int(np.ceil(Ci / 8))
    else:
        n_col, n_img_col = Ci, 1

    for i in range(n_img_row):
        for j in range(n_img_col):
            cur_n_row = n_row if i < n_img_row - 1 else Co - (n_img_row - 1) * n_row
            cur_n_col = n_col if i < n_img_col - 1 else Ci - (n_img_col - 1) * n_col
            cur_weight = weight[i*n_row:i*n_row+cur_n_row, j*n_col:j*n_col+cur_n_col]
            cur_mean = channel_mean[i*n_row:i*n_row+cur_n_row, j*n_col:j*n_col+cur_n_col]
            cur_std = channel_std[i*n_row:i*n_row+cur_n_row, j*n_col:j*n_col+cur_n_col]

            # plt.figure(figsize=(kW * scale * cur_n_col, kH * scale * cur_n_row))
            fig, ax = plt.subplots(cur_n_row, cur_n_col, figsize=(kW * scale * cur_n_col, kH * scale * cur_n_row))
            for p in range(cur_n_row):
                for q in range(cur_n_col):
                    # plt.subplot(cur_n_row, cur_n_col, p * cur_n_col + q + 1)
                    # plt.imshow(cur_weight[p, q], cmap='gray')
                    # plt.colorbar()
                    # plt.xlabel('Mean {:+.3f}, Std {:+.3f}'.format(cur_mean[p, q], cur_std[p, q]), fontsize='small')
                    # plt.title('Kernel {} Channel {}'.format(i*n_row+p, j*n_col+q), fontsize='small')
                    # plt.axis('off')
                    cur_ax = ax[p, q].matshow(cur_weight[p, q], cmap='gray')
                    fig.colorbar(cur_ax, ax=ax[p, q])
                    ax[p, q].set_xlabel('Mean {:+.3f}, Std {:+.3f}'.format(cur_mean[p, q], cur_std[p, q]), fontsize='small')
                    ax[p, q].set_title('Kernel {} Channel {}'.format(i*n_row+p, j*n_col+q), fontsize='small')
                    ax[p, q].xaxis.set_ticks_position('bottom') # 否则和title重叠
                    # ax[p, q].axis('off')

            plt.tight_layout(3.0)
            img_name = '{}_K[{},{})_C[{},{}).png'.format(tag, i*n_row, i*n_row+cur_n_row, j*n_col, j*n_col+cur_n_col)
            img_path = osp.join(save_dir, img_name)
            plt.savefig(img_path)
            if verbose:
                logger.info('{} saved.'.format(img_name))
            plt.close()

    # _tkinter.TclError: not enough free memory for image buffer, 所以即开即关
    # plt.close('all')

