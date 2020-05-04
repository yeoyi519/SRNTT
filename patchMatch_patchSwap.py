import tensorflow as tf
from SRNTT.tensorlayer import *
import numpy as np
from glob import glob
import os
import os.path as osp
from datetime import datetime
from SRNTT.model import *
from SRNTT.vgg19 import *
from SRNTT.swap import *
from SRNTT.util import *
import scipy.misc as scipy_misc
import argparse

tf.logging.set_verbosity(tf.logging.ERROR)

#### start from train.sh
## nohup python main.py --is_train False --save_dir experiments/training_srntt_CUFED --model_epoch 100 \
# --input_dir data_demo/test/CUFED5/CUFED5_0 --ref_dir data_demo/test/CUFED5/CUFED5_1 --clip_fea False \
# --noise_target swapped --noise_mean 0 --noise_sigma 0 \
# --result_dir results/testing_srntt_demo_CUFED5_0_swapped_m0_s0  --use_init_model_only False \
# &>testing_srntt_demo_CUFED5_0_swapped_m0_s0.logs &
# nohup python main.py --is_train False --save_dir experiments/training_srntt_CUFED --model_epoch 100 \
# --input_dir data_demo/test/CUFED5/CUFED5_0 --ref_dir data_demo/test/CUFED5/CUFED5_1 --clip_fea False \
# --noise_target hr --noise_mean 0 --noise_sigma 0 \
# --result_dir results/testing_srntt_demo_CUFED5_0_hr_m0_s0  --use_init_model_only False \
# &>testing_srntt_demo_CUFED5_0_hr_m0_s0.logs &
## nohup python patchMatch_patchSwap.py --swap_target ref_res --swap_size 3 --swap_stride 1 &>CUFED5_1_ref_res_3_1.logs &
## nohup python patchMatch_patchSwap.py --swap_target ref_img --swap_size 3 --swap_stride 1 &>CUFED5_1_ref_img_3_1.logs &
## nohup python patchMatch_patchSwap.py --swap_target ref_res --swap_size 5 --swap_stride 1 &>CUFED5_1_ref_res_5_1.logs &
## nohup python patchMatch_patchSwap.py --swap_target ref_res --swap_size 7 --swap_stride 1 &>CUFED5_1_ref_res_7_1.logs &
## nohup python patchMatch_patchSwap.py --swap_target ref_res --swap_size 7 --swap_stride 3 &>CUFED5_1_ref_res_7_3.logs &
## nohup python patchMatch_patchSwap.py --swap_target ref_res --swap_size 9 --swap_stride 1 &>CUFED5_1_ref_res_9_1.logs &
## nohup python patchMatch_patchSwap.py --swap_target ref_res --swap_size 7 --swap_stride 1 --ref CUFED5_2 &>CUFED5_2_ref_res_7_1.logs &
## nohup python patchMatch_patchSwap.py --swap_target ref_res --swap_size 7 --swap_stride 1 --ref CUFED5_3 &>CUFED5_3_ref_res_7_1.logs &
## nohup python patchMatch_patchSwap.py --swap_target ref_res --swap_size 7 --swap_stride 1 --ref CUFED5_4 &>CUFED5_4_ref_res_7_1.logs &
## nohup python patchMatch_patchSwap.py --swap_target ref_res --swap_size 7 --swap_stride 1 --ref CUFED5_5 &>CUFED5_5_ref_res_7_1.logs &
parser = argparse.ArgumentParser('patchMatch_patchSwap')
parser.add_argument('--data_folder', type=str, default='data_demo/test/CUFED5', help='')
parser.add_argument('--input', type=str, default='CUFED5_0', help='')
parser.add_argument('--ref', type=str, default='CUFED5_1', help='')
parser.add_argument('--swap_target', type=str, default='ref_res', choices=['ref_img', 'ref_res'], help='')
parser.add_argument('--swap_size', type=int, default=3, help='')
parser.add_argument('--swap_stride', type=int, default=1, help='')
args = parser.parse_args()

#### input size (LR)
### CUFED: 40
### DIV2K: 80
data_folder = args.data_folder
# if 'CUFED' in data_folder:
#     input_size = 40
# elif 'DIV2K' in data_folder:
#     input_size = 80
# else:
#     raise Exception('Unrecognized dataset!')

#### dir
### ref_dir:     data_demo/test/CUFED5/CUFED5_1
### input_dir:   data_demo/test/CUFED5/CUFED5_0
### result_dir:  data_demo/test/CUFED5/result_0-1
input_path = osp.join(data_folder, args.input)
ref_path = osp.join(data_folder, args.ref)
# matching_layer = ['relu3_1', 'relu2_1', 'relu1_1']
save_path = osp.join(data_folder, 'result-'+args.input+'-'+args.ref+'-{}'.format(datetime.now().strftime('%y%m%d-%H%M%S')))
tmp_path = osp.join(save_path, 'tmp')
if not osp.exists(save_path):
    os.makedirs(save_path)
if not osp.exists(tmp_path):
    os.makedirs(tmp_path)

#### image paths
input_files = sorted(glob(osp.join(input_path, '*.png')))
ref_files = sorted(glob(osp.join(ref_path, '*.png')))
n_files = len(input_files)
assert n_files == len(ref_files)

#### models: SRNTT(only content extractor), VGG19, Swap
### SRNTT input: [0, 255] -> [-1, 1]
### SRNTT ouput: tanh [-1, 1]
# vgg19_model_path = 'SRNTT/models/VGG19/imagenet-vgg-verydeep-19.mat'
# tf_input = tf.placeholder(dtype=tf.float32, shape=[1, input_size, input_size, 3])
# srntt = SRNTT(vgg19_model_path=vgg19_model_path)
# net_upscale, _ = srntt.model(tf_input / 127.5 - 1, is_train=False)
# net_vgg19 = VGG19(model_path=vgg19_model_path)
swaper = Swap(patch_size=args.swap_size, stride=args.swap_stride)

#### content extractor(pretrained SRGAN): SRNTT/SRNTT/models/SRNTT/upscale.npz
psnr_list, ssim_list = [], []
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    #### load SRNTT content extractor/pretrained SRGAN
    tf.global_variables_initializer().run()
    # model_path = osp.join(osp.dirname(osp.realpath(__file__)), 'SRNTT', 'models', 'SRNTT', 'upscale.npz')
    # if files.load_and_assign_npz(
    #         sess=sess,
    #         name=model_path,
    #         network=net_upscale) is False:
    #     raise Exception('FAILED load %s' % model_path)
    # else:
    #     print('SUCCESS load %s' % model_path)

    for i in range(n_files):
        #### input image: data/train/CUFED/input/*.png
        #### image map:   data/train/CUFED/map_321/*.npz
        #### input HQ -> read(scipy.misc) -> bicubic downsampling(scipy.misc) -> input LR(img_in_lr)[0, 255]
        ###           -> SR(SRNTT content extractor/pretrained SRGAN) -> input SR(img_in_sr)[0, 255]
        ###           -> VGG19 -> input SR relu3_1(map_in_sr)
        #### ref HQ -> read(scipy.misc) -> bicubic resize(scipy.misc) -> ref GT(img_ref)[0, 255]
        ###         -> bicubic downsampling(scipy.misc) -> ref LR(img_ref_lr)[0, 255]
        ###         -> SR(SRNTT content extractor/pretrained SRGAN) -> ref SR(img_ref_sr)[0, 255]
        ###         -> VGG19 -> ref SR relu3_1(map_ref_sr)
        #### ref GT(img_ref) -> VGG19 -> ref GT relu3_1/relu2_1/relu1_1(map_ref)
        #### patch matching and swapping on relu3_1, transfer to relu2_1/relu1_1
        #           ->  swapped features(maps),
        #               (weights), 与每个 input SR patch 最相关的 ref SR patch 的內积相似度 (on relu3_1)
        #               max_idx(correspondence), 与每个 input SR patch 最相似的 ref SR patch 的块索引 (on relu3_1)
        #### save, i.e.
        #   maps, list, [relu3_1, relu2_1, relu1_1], each (h, w, c) type
        #       relu3_1, (H//4, W//4, nf3)=(40, 40, 256)
        #       relu2_1, (H//2, W//2, nf2)=(80, 80, 128)
        #       relu1_1, (H, W, nf1)=(160, 160, 64)
        #   weights = None
        #   correspondence, numpy array, (ho, wo) (conv on relu3_1)
        img_name, img_ext = osp.splitext(osp.basename(input_files[i]))
        file_name = osp.join(save_path, img_name+'_SwapSR'+img_ext)
        if osp.exists(file_name):
            continue
        print('{:05d}/{:05d} {}'.format(i + 1, n_files, img_name))

        img_input = scipy_misc.imread(input_files[i], mode='RGB')
        h, w, _ = img_input.shape
        h = int(h // 4 * 4)
        w = int(w // 4 * 4)
        img_in_hr = img_input[0:h, 0:w, ::]
        img_in_lr = scipy_misc.imresize(img_in_hr, .25, interp='bicubic')
        h, w, _ = img_in_lr.shape
        img_in_lr_copy = np.copy(img_in_lr)
        if h * w * 16 > 2046 ** 2:  # avoid OOM
            # split img_input into patches
            img_in_lr, grids = split_img(img_in_lr)
        else:
            grids = None
            img_in_lr = np.expand_dims(img_in_lr, axis=0)

        img_ref = scipy_misc.imread(ref_files[i], mode='RGB')
        h2, w2, _ = img_ref.shape
        h2 = int(h2 // 4 * 4)
        w2 = int(w2 // 4 * 4)
        img_ref_hr = img_ref[0:h2, 0:w2, ::]
        # map_ref_hr = net_vgg19.get_layer_output(sess=sess, feed_image=img_ref_hr, layer_name=matching_layer)
        # other_style = []
        # for m in map_ref_hr[1:]:
        #     other_style.append([m])
        img_ref_lr = scipy_misc.imresize(img_ref_hr, .25, interp='bicubic')
        img_ref_sr = scipy_misc.imresize(img_ref_lr, 4., interp='bicubic')
        # img_ref_sr = (net_upscale.outputs.eval({tf_input: [img_ref_lr]})[0] + 1) * 127.5
        # map_ref_sr = net_vgg19.get_layer_output(sess=sess, feed_image=img_ref_sr, layer_name=matching_layer[0])

        for idx, patch_lr in enumerate(img_in_lr): # 同一张测试图的不同patch
            print('\tPatch {:03d}/{:03d}'.format(idx + 1, img_in_lr.shape[0]))

            # skip if the results exists
            patch_sr_path = osp.join(tmp_path, 'sr_{:05d}_{:03d}.png'.format(i, idx))
            patch_swap_path = osp.join(tmp_path, 'swap_{:05d}_{:03d}.png'.format(i, idx))
            if osp.exists(patch_swap_path):
                continue

            patch_sr = scipy_misc.imresize(patch_lr, 4., interp='bicubic')
            # patch_sr = (net_upscale.outputs.eval({tf_input: [patch_lr]})[0] + 1) * 127.5
            # map_patch_sr = net_vgg19.get_layer_output(sess=sess, feed_image=patch_sr, layer_name=matching_layer[0])

            if args.swap_target == 'ref_res':
                style = [img_ref_hr.astype(np.float32) - img_ref_sr.astype(np.float32)]
            else:
                style = [img_ref_hr.astype(np.float32)]

            maps, weights, correspondence = swaper.conditional_swap_multi_layer(
                content=patch_sr.astype(np.float32),
                style=style,
                condition=[img_ref_sr.astype(np.float32)],
                patch_size=args.swap_size, stride=args.swap_stride
            )

            if args.swap_target == 'ref_res':
                patch_swap = patch_sr.astype(np.float32) + maps[0]
            else:
                patch_swap = patch_sr.astype(np.float32)

            scipy_misc.imsave(patch_sr_path, patch_sr.round().clip(0, 255).astype(np.uint8))
            scipy_misc.imsave(patch_swap_path, patch_swap.round().clip(0, 255).astype(np.uint8))

        patch_sr_files = sorted(glob(osp.join(tmp_path, 'sr_{:05d}_*.png'.format(i))))
        patch_swap_files = sorted(glob(osp.join(tmp_path, 'swap_{:05d}_*.png'.format(i))))
        if grids is not None:
            img_in_sr, img_in_swap = recon_patch(img_in_lr_copy, grids, patch_sr_files, patch_swap_files)
        else:
            img_in_sr = scipy_misc.imread(patch_sr_files[0], mode='RGB')
            img_in_swap = scipy_misc.imread(patch_swap_files[0], mode='RGB')

        scipy_misc.imsave(file_name.replace('SwapSR', 'in_hr'), img_in_hr)
        scipy_misc.imsave(file_name.replace('SwapSR', 'in_lr'), img_in_lr_copy)
        scipy_misc.imsave(file_name.replace('SwapSR', 'in_sr'), img_in_sr)
        scipy_misc.imsave(file_name.replace('SwapSR', 'ref_hr'), img_ref_hr)
        scipy_misc.imsave(file_name.replace('SwapSR', 'ref_lr'), img_ref_lr)
        scipy_misc.imsave(file_name.replace('SwapSR', 'ref_sr'), img_ref_sr)
        scipy_misc.imsave(file_name, img_in_swap)
        psnr = calculate_psnr(img_in_swap, img_in_hr)
        ssim = calculate_ssim(img_in_swap, img_in_hr)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        print('\tPSNR(dB): {:.3f}, SSIM: {:.3f}'.format(psnr, ssim))

print('Avg of {}, PSNR(dB): {:.3f}, SSIM: {:.3f}'.format(len(psnr_list), np.mean(psnr_list), np.mean(ssim_list)))


