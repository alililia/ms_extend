# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===========================================================================
"""Parse parameters function."""

import argparse


def parse_args():
    """
    Parse parameters.

    Help description for each configuration:
        train_data_dir: file path of training input data.
        device_target: device id of GPU or Ascend, default is None.
        train_fakeimg_dir: file path of stored fake img in training.
        loss_show_dir: file path of stored loss img in training.
        ckpt_dir: file path of stored checkpoint file in training.
        epoch_num: epoch number for training,different datasets have different values, default=200.
        batch_size: batch size,different size datasets have different values, default=1.
        run_distribute: whether to run distribute, default is False.
        beta1: Adam beta1, default=0.5.
        beta2: Adam beta2, default=0.999.

        val_data_dir: file path of validation input data.
        ckpt: file path of checking point file used in validation.
        predict_dir: file path of generated image in validation.

        load_size: scale images to this size, default=286.
        train_pic_size: train image size, default=256.
        val_pic_size: train image size, default=256.

        lambda_dis: weight for discriminator loss, default=0.5.
        lambda_gan: weight for GAN loss, default=0.5.
        lambda_l1: weight for L1 loss, default=100.

        lr: initial learning rate, default=0.1.
        dataset_size: training dataset size, different size datasets have different values. default=400.
        n_epochs: number of epochs with the initial learning rate, default=100.
        n_epochs_decay: number of epochs with the dynamic learning rate, default=100.

        g_in_planes (int): the number of channels in input images, default=3.
        g_out_planes (int): the number of channels in output images, default=3.
        g_ngf (int): the number of filters in the last conv layer, default=64.
        g_layers (int): the number of downsamplings in UNet, default=8.
        d_in_planes (int): Input channel, default=6.
        d_ndf (int): the number of filters in the last conv layer, default=64.
        d__layers (int): The number of ConvnormRelu blocks, default=3.
        alpha (float): LeakyRelu slope, default= 0.2.
        init_gain: scaling factor for normal xavier and orthogonal, default=0.02.
        pad_mode: pad mode, default is CONSTANT".
        init_type: network init type, default is "normal".

     Returns:
        parsed parameters.
    """
    #train
    parser = argparse.ArgumentParser(description='config')
    parser.add_argument('--train_data_dir', default='../data/maps/train/', type=str)
    parser.add_argument('--device_target', default='GPU', choices=['GPU', 'Ascend'], type=str)
    parser.add_argument('--train_fakeimg_dir', default='results/fake_img/', type=str)
    parser.add_argument('--loss_show_dir', default='results/loss_show', type=str)
    parser.add_argument('--ckpt_dir', default='results/ckpt', type=str)
    parser.add_argument('--epoch_num', default=200, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--run_distribute', default=False, type=bool)
    parser.add_argument('--beta1', default=0.5, type=float)
    parser.add_argument('--beta2', default=0.999, type=float)

    #eval
    parser.add_argument('--val_data_dir', default='../data/maps/val/', type=str)
    parser.add_argument('--ckpt', default='results/ckpt/Generator_200.ckpt', type=str)
    parser.add_argument('--predict_dir', default='results/predict/', type=str)

    #dataset
    parser.add_argument('--load_size', default=286, type=int)
    parser.add_argument('--train_pic_size', default=256, type=int)
    parser.add_argument('--val_pic_size', default=256, type=int)

    #loss
    parser.add_argument('--lambda_dis', default=0.5, type=float)
    parser.add_argument('--lambda_gan', default=0.5, type=float)
    parser.add_argument('--lambda_l1', default=100, type=int)

    #tools
    parser.add_argument('--lr', default=0.0002, type=float)
    parser.add_argument('--dataset_size', default=400, type=int)
    parser.add_argument('--n_epochs', default=100, type=int)
    parser.add_argument('--n_epochs_decay', default=100, type=int)

    #network
    parser.add_argument('--g_in_planes', default=3, type=int)
    parser.add_argument('--g_out_planes', default=3, type=int)
    parser.add_argument('--g_ngf', default=64, type=int)
    parser.add_argument('--g_layers', default=8, type=int)
    parser.add_argument('--d_in_planes', default=6, type=int)
    parser.add_argument('--d_ndf', default=64, type=int)
    parser.add_argument('--d_layers', default=3, type=int)
    parser.add_argument('--alpha', default=0.2, type=float)
    parser.add_argument('--init_gain', default=0.02, type=float)
    parser.add_argument('--pad_mode', default='CONSTANT', type=str)
    parser.add_argument('--init_type', default='normal', type=str)

    return parser.parse_args()


pix2pix_config = parse_args()
