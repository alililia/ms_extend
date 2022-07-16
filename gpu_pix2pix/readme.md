# Pix2Pix

Pix2Pix is a deep learning image conversion model based on conditional generative adversarial network. The model can realize semantic or label to real image, grayscale image to color image, aerial image to map, day to night, line drawing to physical image conversion. Pix2Pix is a general framework for image translation based on conditional GAN, which realizes the generalization of model structure and loss function, and has achieved remarkable results on many image translation datasets.In Pix2Pix, the generator and the discriminator play against each other to optimize the two models at the same time.

## Pretrained model

Model trained by MindSpore, Six datasets correspond to six ckpt files.

|  dataset  |  ckpt  |
| ------- | ------ |
| maps | [ckpt](https://download.mindspore.cn/vision/pix2pix/maps/Generator_200.ckpt) |
| cityscapes |[ckpt](https://download.mindspore.cn/vision/pix2pix/cityscapes/Generator_200.ckpt) |
| facades |[ckpt](https://download.mindspore.cn/vision/pix2pix/facades/Generator_200.ckpt) |
| night2day |[ckpt](https://download.mindspore.cn/vision/pix2pix/night2day/Generator_17.ckpt) |
| edges2shoes |[ckpt](https://download.mindspore.cn/vision/pix2pix/edge2shoes/Generator_15.ckpt) |
| edges2handbags |[ckpt](https://download.mindspore.cn/vision/pix2pix/edge2handbags/Generator_15.ckpt) |

## Training Parameter description

| Parameter | Default | Description |
|:-----|:---------|:--------|
| device_target | GPU | Device id of GPU or Ascend |
| epoch_num | 200 | Epoch number for training |
| batch_size | 1 | Batch size |
| beta1 | 0.5 | Adam beta1 |
| beta2 | 0.999 | Adam beta2 |
| load_size | 286 | Scale images to this size |
| train_pic_size | 256 | Train image size |
| val_pic_size | 256 | Eval image size |
| lambda_dis | 0.5 | Weight for discriminator loss |
| lambda_gan | 0.5 | Weight for GAN loss |
| lambda_l1 | 100 | Weight for L1 loss |
| lr | 0.0002 | Initial learning rate |
| n_epochs | 256 | Number of epochs with the initial learning rate |
| g_in_planes | 3 | The number of channels in input images |
| g_out_planes | 3 | The number of channels in output images |
| g_ngf | 64 | The number of filters in the last conv layer |
| g_layers | 8 | The number of downsamplings in UNet |
| d_in_planes | 6 | Input channel |
| d_ndf | 64 | The number of filters in the last conv layer |
| d_layers | 3 | The number of ConvNormRelu blocks |
| alpha | 0.2 | LeakyRelu slope |
| init_gain | 0.02 | Scaling factor for normal xavier and orthogonal |
| train_fakeimg_dir | results/fake_img/ | File path of stored fake img in training |
| loss_show_dir | results/loss_show | File path of stored loss img in training |
| ckpt_dir | results/ckpt | File path of stored checkpoint file in training |
| ckpt | results/ckpt/Generator_200.ckpt | File path of checking point file used in validation |
| predict_dir | results/predict/ | File path of generated image in validation |

## Example

Here, how to use Pix2Pix model will be introduec as following.

***

### Train

- The following configuration uses 1 GPUs for training. We select edges2handbags.tar.gz. The trained for  15 epochs, and the batch size 4.

```shell
  python train.py
```

output:

```text
================start===================  
Date time:  2022-07-06 23:29:31.486978  
ms per step : 573.098  
epoch:  1 / 15  
step:  0 / 34641  
Dloss:  0.8268157  
Gloss:  96.03406  
=================end====================  
================start===================  
Date time:  2022-07-06 23:29:42.813233  
ms per step : 48.086  
epoch:  1 / 15  
step:  100 / 34641  
Dloss:  0.08200404  
Gloss:  14.9787245  
=================end====================  
================start===================  
Date time:  2022-07-06 23:29:47.699990  
ms per step : 49.513  
epoch:  1 / 15  
step:  200 / 34641  
Dloss:  1.7021327  
Gloss:  11.134652  
=================end====================  
···  
================start===================  
Date time:  2022-07-07 06:29:31.024008  
ms per step : 48.005  
epoch:  15 / 15  
step:  34400 / 34641  
Dloss:  6.0905662e-05  
Gloss:  13.91369  
=================end====================  
================start===================  
Date time:  2022-07-07 06:29:35.837478  
ms per step : 48.055  
epoch:  15 / 15  
step:  34500 / 34641  
Dloss:  6.202688e-05  
Gloss:  17.475426  
=================end====================  
================start===================  
Date time:  2022-07-07 06:29:40.601357  
ms per step : 47.031  
epoch:  15 / 15  
step:  34600 / 34641  
Dloss:  1.3251489e-06  
Gloss:  17.247965  
=================end====================
```

### Infer

- The following configuration for infer.

```shell
python eval.py --platform GPU --ckpt results/ckpt/Generator_15.ckpt
```

**Result**

![5.png](./images/5.png)