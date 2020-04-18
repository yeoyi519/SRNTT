import tensorflow as tf
from SRNTT.tensorlayer import *
import numpy as np
from glob import glob
from os.path import exists, join, split, realpath, dirname
from os import makedirs
from SRNTT.model import *
from SRNTT.vgg19 import *
from SRNTT.swap import *
from scipy.misc import imread, imresize
import argparse

tf.logging.set_verbosity(tf.logging.ERROR)

#### start from train.sh
parser = argparse.ArgumentParser('offline_patchMatch_textureSwap')
parser.add_argument('--data_folder', type=str, default='data/train/CUFED', help='The dir of dataset: CUFED or DIV2K')
args = parser.parse_args()

#### input size (LR)
### CUFED: 40
### DIV2K: 80
data_folder = args.data_folder
if 'CUFED' in data_folder:
    input_size = 40
elif 'DIV2K' in data_folder:
    input_size = 80
else:
    raise Exception('Unrecognized dataset!')

#### dir
### ref_dir:     data/train/CUFED/ref
### input_dir:   data/train/CUFED/input
### map_dir:     data/train/CUFED/map_321
input_path = join(data_folder, 'input')
ref_path = join(data_folder, 'ref')
matching_layer = ['relu3_1', 'relu2_1', 'relu1_1']
save_path = join(data_folder, 'map_321')
if not exists(save_path):
    makedirs(save_path)

#### image paths
input_files = sorted(glob(join(input_path, '*.png')))
ref_files = sorted(glob(join(ref_path, '*.png')))
n_files = len(input_files)
assert n_files == len(ref_files)

#### models: SRNTT(only content extractor), VGG19, Swap
### SRNTT input: [0, 255] -> [-1, 1]
### SRNTT ouput: tanh [-1, 1]
vgg19_model_path = 'SRNTT/models/VGG19/imagenet-vgg-verydeep-19.mat'
tf_input = tf.placeholder(dtype=tf.float32, shape=[1, input_size, input_size, 3])
srntt = SRNTT(vgg19_model_path=vgg19_model_path)
net_upscale, _ = srntt.model(tf_input / 127.5 - 1, is_train=False)
net_vgg19 = VGG19(model_path=vgg19_model_path)
swaper = Swap()

#### content extractor(pretrained SRGAN): SRNTT/SRNTT/models/SRNTT/upscale.npz
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    #### load SRNTT content extractor/pretrained SRGAN
    tf.global_variables_initializer().run()
    model_path = join(dirname(realpath(__file__)), 'SRNTT', 'models', 'SRNTT', 'upscale.npz')
    if files.load_and_assign_npz(
            sess=sess,
            name=model_path,
            network=net_upscale) is False:
        raise Exception('FAILED load %s' % model_path)
    else:
        print('SUCCESS load %s' % model_path)

    print_format = '%%0%dd/%%0%dd' % (len(str(n_files)), len(str(n_files)))
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
        file_name = join(save_path, split(input_files[i])[-1].replace('.png', '.npz'))
        if exists(file_name):
            continue
        print(print_format % (i + 1, n_files))
        img_in_lr = imresize(imread(input_files[i], mode='RGB'), (input_size, input_size), interp='bicubic')
        img_ref = imresize(imread(ref_files[i], mode='RGB'), (input_size * 4, input_size * 4), interp='bicubic')
        img_ref_lr = imresize(img_ref, (input_size, input_size), interp='bicubic')
        img_in_sr = (net_upscale.outputs.eval({tf_input: [img_in_lr]})[0] + 1) * 127.5
        img_ref_sr = (net_upscale.outputs.eval({tf_input: [img_ref_lr]})[0] + 1) * 127.5

        # get feature maps via VGG19
        map_in_sr = net_vgg19.get_layer_output(sess=sess, feed_image=img_in_sr, layer_name=matching_layer[0])
        map_ref = net_vgg19.get_layer_output(sess=sess, feed_image=img_ref, layer_name=matching_layer)
        map_ref_sr = net_vgg19.get_layer_output(sess=sess, feed_image=img_ref_sr, layer_name=matching_layer[0])

        # patch matching and swapping
        other_style = []
        for m in map_ref[1:]:
            other_style.append([m])

        maps, weights, correspondence = swaper.conditional_swap_multi_layer(
            content=map_in_sr,
            style=[map_ref[0]],
            condition=[map_ref_sr],
            other_styles=other_style
        )

        # save maps
        np.savez(file_name, target_map=maps, weights=weights, correspondence=correspondence)
