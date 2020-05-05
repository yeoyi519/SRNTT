import os
from os.path import join, isdir, basename, splitext
from glob import glob
from SRNTT.model import *
from SRNTT.util import *
import argparse
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='SRNTT')

# init parameters
parser.add_argument('--is_train', type=str2bool, default=False)
parser.add_argument('--srntt_model_path', type=str, default='SRNTT/models/SRNTT')
parser.add_argument('--vgg19_model_path', type=str, default='SRNTT/models/VGG19/imagenet-vgg-verydeep-19.mat')
parser.add_argument('--save_dir', type=str, default=None, help='dir of saving intermediate training results')
parser.add_argument('--model_epoch', type=int, default=-1, help='')
parser.add_argument('--num_res_blocks', type=int, default=16, help='number of residual blocks')

# train parameters
parser.add_argument('--input_dir', type=str, default='data/train/input', help='dir of input images')
parser.add_argument('--ref_dir', type=str, default='data/train/ref', help='dir of reference images')
parser.add_argument('--map_dir', type=str, default='data/train/map_321', help='dir of texture maps of reference images')
parser.add_argument('--batch_size', type=int, default=9)
parser.add_argument('--num_init_epochs', type=int, default=5)
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--beta1', type=float, default=0.9)
### train: 指定 save_dir 可继续训练, 但是只允许从当前最新的模型继续, 模型保存的序号继续增加
## use_init_model_only: 基于只使用 loss_init 训练的模型, 继续进行二阶段训练(loss_init, loss)
## use_pretrained_model: 基于使用 loss_init, loss 训练的模型, 继续进行二阶段训练(loss_init, loss)
### test: 指定 save_dir 和 model_epoch, 加载指定迭代的模型进行测试, model_epoch=-1 表示使用最新模型
## use_init_model_only: 只使用 loss_init 训练的模型进行测试
## use_pretrained_model: 使用 loss_init, loss 训练的模型进行测试
parser.add_argument('--use_pretrained_model', type=str2bool, default=True)
parser.add_argument('--use_init_model_only', type=str2bool, default=False, help='effect if use_pretrained_model is true')
parser.add_argument('--w_per', type=float, default=1e-4, help='weight of perceptual loss between output and ground truth')
parser.add_argument('--w_tex', type=float, default=1e-4, help='weight of texture loss between output and texture map')
parser.add_argument('--w_adv', type=float, default=1e-6, help='weight of adversarial loss')
parser.add_argument('--w_bp', type=float, default=0.0, help='weight of back projection loss')
parser.add_argument('--w_rec', type=float, default=1.0, help='weight of reconstruction loss')
parser.add_argument('--vgg_perceptual_loss_layer', type=str, default='relu5_1', help='the VGG19 layer name to compute perceptrual loss')
parser.add_argument('--is_WGAN_GP', type=str2bool, default=True, help='whether use WGAN-GP')
parser.add_argument('--is_L1_loss', type=str2bool, default=True, help='whether use L1 norm')
parser.add_argument('--param_WGAN_GP', type=float, default=10, help='parameter for WGAN-GP')
parser.add_argument('--input_size', type=int, default=40)
parser.add_argument('--use_ref_directly', type=str2bool, default=False, help='use ref vgg feature directly')
parser.add_argument('--use_weight_map', type=str2bool, default=False)
parser.add_argument('--use_lower_layers_in_per_loss', type=str2bool, default=False)

# test parameters
parser.add_argument('--result_dir', type=str, default='result', help='dir of saving testing results')
parser.add_argument('--ref_scale', type=float, default=1.0)
parser.add_argument('--is_original_image', type=str2bool, default=True)
parser.add_argument('--clip_fea', type=str2bool, default=False, help='clip swapped or hr feature which for srntt')
parser.add_argument('--noise_target', type=str, default='swapped', choices=['swapped', 'hr'], help='add Gaussian noise on swapped or hr feature when test with ref')
parser.add_argument('--noise_mean', type=float, default=0, help='add Gaussian noise on swapped or hr feature when test with ref')
parser.add_argument('--noise_sigma', type=float, default=0, help='add Gaussian noise on swapped or hr feature when test with ref')
parser.add_argument('--visual_fea', type=str2bool, default=True, help='visualize swapped and hr feature')
parser.add_argument('--srntt_only', type=str2bool, default=False, help='save srntt result and calculate srntt metric only')

args = parser.parse_args()

if args.is_train:
#### start from train.sh
### --is_train                  True
### --input_dir                 data/train/CUFED/input
### --ref_dir                   data/train/CUFED/ref
### --map_dir                   data/train/CUFED/map_321
### --use_pretrained_model      False
### --num_init_epochs           2
### --num_epochs                2
### --save_dir                  demo_training_srntt

#### default parameters ####
### --srntt_model_path          SRNTT/models/SRNTT
### --vgg19_model_path          SRNTT/models/VGG19/imagenet-vgg-verydeep-19.mat
### --num_res_blocks            16
#==
### --batch_size                9
### --learning_rate             1e-4
### --beta1                     0.9
### --use_init_model_only       False
### --w_per                     1e-4
### --w_tex                     1e-4
### --w_adv                     1e-6
### --w_bp                      0.0
### --w_rec                     1.0
### --vgg_perceptual_loss_layer         relu5_1
### --is_WGAN_GP                True
### --is_L1_loss                True
### --param_WGAN_GP             10
### --input_size                40
### --use_weight_map            False
### --use_lower_layers_in_per_loss      False
#==
### --result_dir                result
### --ref_scale                 1.0
### --is_original_image         True

    # record parameters to file
    if args.save_dir is None:
        args.save_dir = 'default_save_dir'
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    with open(os.path.join(args.save_dir, 'arguments.txt'), 'w') as f:
        for arg in sorted(vars(args)):
            line = '{:>30}\t{:<10}\n'.format(arg, getattr(args, arg))
            bar = ''
            f.write(line)
        f.close()

    srntt = SRNTT(
        srntt_model_path=args.srntt_model_path,
        vgg19_model_path=args.vgg19_model_path,
        save_dir=args.save_dir,
        num_res_blocks=args.num_res_blocks
    )
    srntt.train(
        input_dir=args.input_dir,
        ref_dir=args.ref_dir,
        map_dir=args.map_dir,
        batch_size=args.batch_size,
        num_init_epochs=args.num_init_epochs,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        use_pretrained_model=args.use_pretrained_model,
        use_init_model_only=args.use_init_model_only,
        weights=(args.w_per, args.w_tex, args.w_adv, args.w_bp, args.w_rec),
        vgg_perceptual_loss_layer=args.vgg_perceptual_loss_layer,
        is_WGAN_GP=args.is_WGAN_GP,
        is_L1_loss=args.is_L1_loss,
        param_WGAN_GP=args.param_WGAN_GP,
        input_size=args.input_size,
        use_ref_directly=args.use_ref_directly,
        use_weight_map=args.use_weight_map,
        use_lower_layers_in_per_loss=args.use_lower_layers_in_per_loss
    )
else:
#### start from test.sh
### --is_train                  False
### --input_dir                 data/test/CUFED5/001_0.png
### --ref_dir                   data/test/CUFED5/001_2.png
### --result_dir                demo_testing_srntt

#### default parameters ####
### --srntt_model_path          SRNTT/models/SRNTT
### --vgg19_model_path          SRNTT/models/VGG19/imagenet-vgg-verydeep-19.mat
### --save_dir                  None
### --num_res_blocks            16
#==
### --map_dir                   data/train/map_321
### --batch_size                9
### --num_init_epochs           5
### --num_epochs                50
### --learning_rate             1e-4
### --beta1                     0.9
### --use_pretrained_model      True
### --use_init_model_only       False
### --w_per                     1e-4
### --w_tex                     1e-4
### --w_adv                     1e-6
### --w_bp                      0.0
### --w_rec                     1.0
### --vgg_perceptual_loss_layer         relu5_1
### --is_WGAN_GP                True
### --is_L1_loss                True
### --param_WGAN_GP             10
### --input_size                40
### --use_weight_map            False
### --use_lower_layers_in_per_loss      False
#==
### --ref_scale                 1.0
### --is_original_image         True
    if args.save_dir is not None:
        # read recorded arguments
        fixed_arguments = ['srntt_model_path', 'vgg19_model_path', 'save_dir', 'num_res_blocks', 'use_weight_map']
        if os.path.exists(os.path.join(args.save_dir, 'arguments.txt')):
            with open(os.path.join(args.save_dir, 'arguments.txt'), 'r') as f:
                for arg, line in zip(sorted(vars(args)), f.readlines()):
                    arg_name, arg_value = line.strip().split('\t')
                    if arg_name in fixed_arguments:
                        fixed_arguments.remove(arg_name)
                        try:
                            if isinstance(getattr(args, arg_name), bool):
                                setattr(args, arg_name, str2bool(arg_value))
                            else:
                                setattr(args, arg_name, type(getattr(args, arg_name))(arg_value))
                        except:
                            print('Unmatched arg_name: %s!' % arg_name)

    srntt = SRNTT(
        srntt_model_path=args.srntt_model_path,
        vgg19_model_path=args.vgg19_model_path,
        save_dir=args.save_dir,
        num_res_blocks=args.num_res_blocks,
    )

    logger = logging.getLogger('')
    if isdir(args.input_dir) and isdir(args.ref_dir):
        imgs = sorted(glob(join(args.input_dir, '*')))
        refs = sorted(glob(join(args.ref_dir, '*')))
        if not args.srntt_only:
            bic_psnr, bic_ssim = [], []
            upscale_psnr, upscale_ssim = [], []
        srntt_psnr, srntt_ssim = [], []
        for img, ref in zip(imgs, refs):
            img_name = splitext(basename(img))[0]
            ref_name = splitext(basename(ref))[0]
            img_hr, img_lr, img_bic, img_upscale, img_srntt = srntt.test(
                input_dir=img,
                ref_dir=ref,
                use_pretrained_model=args.use_pretrained_model,
                use_init_model_only=args.use_init_model_only,
                model_epoch=args.model_epoch,
                use_ref_directly=args.use_ref_directly,
                use_weight_map=args.use_weight_map,
                result_dir=join(args.result_dir, img_name, ref_name),
                ref_scale=args.ref_scale,
                is_original_image=args.is_original_image,
                clip_fea=args.clip_fea,
                noise_target=args.noise_target,
                noise_mean=args.noise_mean,
                noise_sigma=args.noise_sigma,
                visual_fea=args.visual_fea,
                srntt_only=args.srntt_only
            )
            logger.info('Img: {}, Ref: {}'.format(img_name, ref_name))
            if not args.srntt_only:
                bic_psnr.append(calculate_psnr(img_bic, img_hr))
                bic_ssim.append(calculate_ssim(img_bic, img_hr))
                logger.info('\tBicubic - PSNR(dB): {:.3f}, SSIM: {:.3f}'.format(bic_psnr[-1], bic_ssim[-1]))
                upscale_psnr.append(calculate_psnr(img_upscale, img_hr))
                upscale_ssim.append(calculate_ssim(img_upscale, img_hr))
                logger.info('\tUpscale - PSNR(dB): {:.3f}, SSIM: {:.3f}'.format(upscale_psnr[-1], upscale_ssim[-1]))
            srntt_psnr.append(calculate_psnr(img_srntt, img_hr))
            srntt_ssim.append(calculate_ssim(img_srntt, img_hr))
            logger.info('\tSRNTT   - PSNR(dB): {:.3f}, SSIM: {:.3f}'.format(srntt_psnr[-1], srntt_ssim[-1]))
        logger.info('Average:')
        if not args.srntt_only:
            logger.info('\tBicubic - PSNR(dB): {:.3f}, SSIM: {:.3f}'.format(np.mean(bic_psnr), np.mean(bic_ssim)))
            logger.info('\tUpscale - PSNR(dB): {:.3f}, SSIM: {:.3f}'.format(np.mean(upscale_psnr), np.mean(upscale_ssim)))
        logger.info('\tSRNTT   - PSNR(dB): {:.3f}, SSIM: {:.3f}'.format(np.mean(srntt_psnr), np.mean(srntt_ssim)))
    else:
        img_hr, img_lr, img_bic, img_upscale, img_srntt = srntt.test(
            input_dir=args.input_dir,
            ref_dir=args.ref_dir,
            use_pretrained_model=args.use_pretrained_model,
            use_init_model_only=args.use_init_model_only,
            model_epoch=args.model_epoch,
            use_ref_directly=args.use_ref_directly,
            use_weight_map=args.use_weight_map,
            result_dir=args.result_dir,
            ref_scale=args.ref_scale,
            is_original_image=args.is_original_image,
            clip_fea=args.clip_fea,
            noise_target=args.noise_target,
            noise_mean=args.noise_mean,
            noise_sigma=args.noise_sigma,
            visual_fea=args.visual_fea,
            srntt_only=args.srntt_only
        )
        logger.info('Img: {}, Ref: {}'.format(args.input_dir, args.ref_dir))
        if not args.srntt_only:
            logger.info('\tBicubic - PSNR(dB): {:.3f}, SSIM: {:.3f}'.format(calculate_psnr(img_bic, img_hr), calculate_ssim(img_bic, img_hr)))
            logger.info('\tUpscale - PSNR(dB): {:.3f}, SSIM: {:.3f}'.format(calculate_psnr(img_upscale, img_hr), calculate_ssim(img_upscale, img_hr)))
        logger.info('\tSRNTT   - PSNR(dB): {:.3f}, SSIM: {:.3f}'.format(calculate_psnr(img_srntt, img_hr), calculate_ssim(img_srntt, img_hr)))

