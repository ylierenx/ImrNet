import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import model
import dataloder
from math import log10
import sys
import time
# import seaborn as sns
# import matplotlib.pyplot as plt
import os
from torchvision.transforms import Compose, ToTensor
import cv2
from os.path import join
from os import listdir
import math
import numpy as np
import matplotlib.image as mpimg
import seaborn as sns
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import xlwt
from scipy.signal import convolve2d
import xlwt

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from torchvision.utils import save_image

from torch.utils.data import Dataset, DataLoader

rgb = False
flag = True
block_size = 32
device = torch.device("cuda:0")
gop_size = 8
image_width = 192
image_height = 160
img_w = 176
img_h = 144
num_gop = 12

test_video_name = 'test_video'
sr = 0.1
# nonkey initial reconstruction
sample_and_inirecon_nonkey = model.sample_and_inirecon(num_filters=int(1024 * sr), B_size=32)
sample_and_inirecon_nonkey.to(device)
Biv_Shr1_nonkey = model.Biv_Shr()
Biv_Shr1_nonkey.to(device)
Biv_Shr2_nonkey = model.Biv_Shr()
Biv_Shr2_nonkey.to(device)
Biv_Shr3_nonkey = model.Biv_Shr()
Biv_Shr3_nonkey.to(device)
wiener_net1_nonkey = model.wiener_net()
wiener_net1_nonkey.to(device)
wiener_net2_nonkey = model.wiener_net()
wiener_net2_nonkey.to(device)
wiener_net3_nonkey = model.wiener_net()
wiener_net3_nonkey.to(device)

# key initial reconstruction
sample_and_inirecon_key = model.sample_and_inirecon(num_filters=512, B_size=32)
sample_and_inirecon_key.to(device)
Biv_Shr1_key = model.Biv_Shr()
Biv_Shr1_key.to(device)
Biv_Shr2_key = model.Biv_Shr()
Biv_Shr2_key.to(device)
Biv_Shr3_key = model.Biv_Shr()
Biv_Shr3_key.to(device)
wiener_net1_key = model.wiener_net()
wiener_net1_key.to(device)
wiener_net2_key = model.wiener_net()
wiener_net2_key.to(device)
wiener_net3_key = model.wiener_net()
wiener_net3_key.to(device)

# spynet 1_forward
flownet_5_forward_1 = model.flownet_first()
flownet_5_forward_1.to(device)
flownet_4_forward_1 = model.flownet()
flownet_4_forward_1.to(device)
flownet_3_forward_1 = model.flownet()
flownet_3_forward_1.to(device)
flownet_2_forward_1 = model.flownet()
flownet_2_forward_1.to(device)
flownet_1_forward_1 = model.flownet()
flownet_1_forward_1.to(device)

# spynet 1_backward
flownet_5_backward_1 = model.flownet_first()
flownet_5_backward_1.to(device)
flownet_4_backward_1 = model.flownet()
flownet_4_backward_1.to(device)
flownet_3_backward_1 = model.flownet()
flownet_3_backward_1.to(device)
flownet_2_backward_1 = model.flownet()
flownet_2_backward_1.to(device)
flownet_1_backward_1 = model.flownet()
flownet_1_backward_1.to(device)

# warp
FlowBackWarp5 = model.backWarp(W=image_width / (pow(2, 4)), H=image_height / (pow(2, 4)), device=device)
FlowBackWarp5.to(device)
FlowBackWarp4 = model.backWarp(W=image_width / (pow(2, 3)), H=image_height / (pow(2, 3)), device=device)
FlowBackWarp4.to(device)
FlowBackWarp3 = model.backWarp(W=image_width / (pow(2, 2)), H=image_height / (pow(2, 2)), device=device)
FlowBackWarp3.to(device)
FlowBackWarp2 = model.backWarp(W=image_width / (pow(2, 1)), H=image_height / (pow(2, 1)), device=device)
FlowBackWarp2.to(device)
FlowBackWarp1 = model.backWarp(W=image_width, H=image_height, device=device)
FlowBackWarp1.to(device)

# probnet
Prob_net_1 = model.Prob_net()
Prob_net_1.to(device)

# residual reconstruction
sample_and_inirecon_2 = model.sample_and_inirecon_2(num_filters=int(1024 * sr), B_size=32,
                                                    init_weight=sample_and_inirecon_nonkey.sample.weight)
sample_and_inirecon_2.to(device)
Biv_Shr1_2 = model.Biv_Shr()
Biv_Shr1_2.to(device)
Biv_Shr2_2 = model.Biv_Shr()
Biv_Shr2_2.to(device)
Biv_Shr3_2 = model.Biv_Shr()
Biv_Shr3_2.to(device)
wiener_net1_2 = model.wiener_net()
wiener_net1_2.to(device)
wiener_net2_2 = model.wiener_net()
wiener_net2_2.to(device)
wiener_net3_2 = model.wiener_net()
wiener_net3_2.to(device)

# spynet 2_forward
flownet_5_forward_2 = model.flownet_first()
flownet_5_forward_2.to(device)
flownet_4_forward_2 = model.flownet()
flownet_4_forward_2.to(device)
flownet_3_forward_2 = model.flownet()
flownet_3_forward_2.to(device)
flownet_2_forward_2 = model.flownet()
flownet_2_forward_2.to(device)
flownet_1_forward_2 = model.flownet()
flownet_1_forward_2.to(device)

# spynet 2_backward
flownet_5_backward_2 = model.flownet_first()
flownet_5_backward_2.to(device)
flownet_4_backward_2 = model.flownet()
flownet_4_backward_2.to(device)
flownet_3_backward_2 = model.flownet()
flownet_3_backward_2.to(device)
flownet_2_backward_2 = model.flownet()
flownet_2_backward_2.to(device)
flownet_1_backward_2 = model.flownet()
flownet_1_backward_2.to(device)

# probnet
Prob_net_2 = model.Prob_net()
Prob_net_2.to(device)

# residual reconstruction
sample_and_inirecon_3 = model.sample_and_inirecon_2(num_filters=int(1024 * sr), B_size=32,
                                                    init_weight=sample_and_inirecon_nonkey.sample.weight)
sample_and_inirecon_3.to(device)
Biv_Shr1_3 = model.Biv_Shr()
Biv_Shr1_3.to(device)
Biv_Shr2_3 = model.Biv_Shr()
Biv_Shr2_3.to(device)
Biv_Shr3_3 = model.Biv_Shr()
Biv_Shr3_3.to(device)
wiener_net1_3 = model.wiener_net()
wiener_net1_3.to(device)
wiener_net2_3 = model.wiener_net()
wiener_net2_3.to(device)
wiener_net3_3 = model.wiener_net()
wiener_net3_3.to(device)

if flag:
    dict1 = torch.load('./check_point1/imrnet_05_01_gop_08.ckpt')
    sample_and_inirecon_nonkey.load_state_dict(dict1['state_dict_sample_and_inirecon_nonkey'])
    Biv_Shr1_nonkey.load_state_dict(dict1['state_dict_Biv_Shr1_nonkey'])
    Biv_Shr2_nonkey.load_state_dict(dict1['state_dict_Biv_Shr2_nonkey'])
    Biv_Shr3_nonkey.load_state_dict(dict1['state_dict_Biv_Shr3_nonkey'])
    wiener_net1_nonkey.load_state_dict(dict1['state_dict_wiener_net1_nonkey'])
    wiener_net2_nonkey.load_state_dict(dict1['state_dict_wiener_net2_nonkey'])
    wiener_net3_nonkey.load_state_dict(dict1['state_dict_wiener_net3_nonkey'])
    # sample_and_inirecon_nonkey_params.load_state_dict(dict1['state_dict_sample_and_inirecon_nonkey'])
    sample_and_inirecon_key.load_state_dict(dict1['state_dict_sample_and_inirecon_key'])
    Biv_Shr1_key.load_state_dict(dict1['state_dict_Biv_Shr1_key'])
    Biv_Shr2_key.load_state_dict(dict1['state_dict_Biv_Shr2_key'])
    Biv_Shr3_key.load_state_dict(dict1['state_dict_Biv_Shr3_key'])
    wiener_net1_key.load_state_dict(dict1['state_dict_wiener_net1_key'])
    wiener_net2_key.load_state_dict(dict1['state_dict_wiener_net2_key'])
    wiener_net3_key.load_state_dict(dict1['state_dict_wiener_net3_key'])

    flownet_1_forward_1.load_state_dict(dict1['state_dict_flownet_1_forward_1'])
    flownet_2_forward_1.load_state_dict(dict1['state_dict_flownet_2_forward_1'])
    flownet_3_forward_1.load_state_dict(dict1['state_dict_flownet_3_forward_1'])
    flownet_4_forward_1.load_state_dict(dict1['state_dict_flownet_4_forward_1'])
    flownet_5_forward_1.load_state_dict(dict1['state_dict_flownet_5_forward_1'])

    flownet_1_backward_1.load_state_dict(dict1['state_dict_flownet_1_backward_1'])
    flownet_2_backward_1.load_state_dict(dict1['state_dict_flownet_2_backward_1'])
    flownet_3_backward_1.load_state_dict(dict1['state_dict_flownet_3_backward_1'])
    flownet_4_backward_1.load_state_dict(dict1['state_dict_flownet_4_backward_1'])
    flownet_5_backward_1.load_state_dict(dict1['state_dict_flownet_5_backward_1'])

    Prob_net_1.load_state_dict(dict1['state_dict_Prob_net_1'])

    sample_and_inirecon_2.load_state_dict(dict1['state_dict_sample_and_inirecon_2'])
    Biv_Shr1_2.load_state_dict(dict1['state_dict_Biv_Shr1_2'])
    Biv_Shr2_2.load_state_dict(dict1['state_dict_Biv_Shr2_2'])
    Biv_Shr3_2.load_state_dict(dict1['state_dict_Biv_Shr3_2'])
    wiener_net1_2.load_state_dict(dict1['state_dict_wiener_net1_2'])
    wiener_net2_2.load_state_dict(dict1['state_dict_wiener_net2_2'])
    wiener_net3_2.load_state_dict(dict1['state_dict_wiener_net3_2'])

    flownet_1_forward_2.load_state_dict(dict1['state_dict_flownet_1_forward_2'])
    flownet_2_forward_2.load_state_dict(dict1['state_dict_flownet_2_forward_2'])
    flownet_3_forward_2.load_state_dict(dict1['state_dict_flownet_3_forward_2'])
    flownet_4_forward_2.load_state_dict(dict1['state_dict_flownet_4_forward_2'])
    flownet_5_forward_2.load_state_dict(dict1['state_dict_flownet_5_forward_2'])

    flownet_1_backward_2.load_state_dict(dict1['state_dict_flownet_1_backward_2'])
    flownet_2_backward_2.load_state_dict(dict1['state_dict_flownet_2_backward_2'])
    flownet_3_backward_2.load_state_dict(dict1['state_dict_flownet_3_backward_2'])
    flownet_4_backward_2.load_state_dict(dict1['state_dict_flownet_4_backward_2'])
    flownet_5_backward_2.load_state_dict(dict1['state_dict_flownet_5_backward_2'])

    Prob_net_2.load_state_dict(dict1['state_dict_Prob_net_2'])

    sample_and_inirecon_3.load_state_dict(dict1['state_dict_sample_and_inirecon_3'])
    Biv_Shr1_3.load_state_dict(dict1['state_dict_Biv_Shr1_3'])
    Biv_Shr2_3.load_state_dict(dict1['state_dict_Biv_Shr2_3'])
    Biv_Shr3_3.load_state_dict(dict1['state_dict_Biv_Shr3_3'])
    wiener_net1_3.load_state_dict(dict1['state_dict_wiener_net1_3'])
    wiener_net2_3.load_state_dict(dict1['state_dict_wiener_net2_3'])
    wiener_net3_3.load_state_dict(dict1['state_dict_wiener_net3_3'])

else:
    dict1 = {'epoch': -1}


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)


def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):
    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")

    M, N = im1.shape
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    window = matlab_style_gauss2D(shape=(win_size, win_size), sigma=1.5)
    window = window / np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1 * im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2 * im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1 * im2, window, 'valid') - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigmal2 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return np.mean(np.mean(ssim_map))


def SPL_Block_1_nonkey(x, y, flag):
    x = wiener_net1_nonkey(x)
    x = sample_and_inirecon_nonkey(x, y, flag)
    x = Biv_Shr1_nonkey(x, flag)
    x = sample_and_inirecon_nonkey(x, y, flag)
    return x


def SPL_Block_2_nonkey(x, y, flag):
    x = wiener_net2_nonkey(x)
    x = sample_and_inirecon_nonkey(x, y, flag)
    x = Biv_Shr2_nonkey(x, flag)
    x = sample_and_inirecon_nonkey(x, y, flag)
    return x


def SPL_Block_3_nonkey(x, y, flag):
    x = wiener_net3_nonkey(x)
    x = sample_and_inirecon_nonkey(x, y, flag)
    x = Biv_Shr3_nonkey(x, flag)
    x = sample_and_inirecon_nonkey(x, y, flag)
    return x


def SPL_Block_1_key(x, y, flag):
    x = wiener_net1_key(x)
    x = sample_and_inirecon_key(x, y, flag)
    x = Biv_Shr1_key(x, flag)
    x = sample_and_inirecon_key(x, y, flag)
    return x


def SPL_Block_2_key(x, y, flag):
    x = wiener_net2_key(x)
    x = sample_and_inirecon_key(x, y, flag)
    x = Biv_Shr2_key(x, flag)
    x = sample_and_inirecon_key(x, y, flag)
    return x


def SPL_Block_3_key(x, y, flag):
    x = wiener_net3_key(x)
    x = sample_and_inirecon_key(x, y, flag)
    x = Biv_Shr3_key(x, flag)
    x = sample_and_inirecon_key(x, y, flag)
    return x


def SPL_Block_1_2(x, y, flag):
    x = wiener_net1_2(x)
    x = sample_and_inirecon_2(x, y, flag)
    x = Biv_Shr1_2(x, flag)
    x = sample_and_inirecon_2(x, y, flag)
    return x


def SPL_Block_2_2(x, y, flag):
    x = wiener_net2_2(x)
    x = sample_and_inirecon_2(x, y, flag)
    x = Biv_Shr2_2(x, flag)
    x = sample_and_inirecon_2(x, y, flag)
    return x


def SPL_Block_3_2(x, y, flag):
    x = wiener_net3_2(x)
    x = sample_and_inirecon_2(x, y, flag)
    x = Biv_Shr3_2(x, flag)
    x = sample_and_inirecon_2(x, y, flag)
    return x


def SPL_Block_1_3(x, y, flag):
    x = wiener_net1_3(x)
    x = sample_and_inirecon_3(x, y, flag)
    x = Biv_Shr1_3(x, flag)
    x = sample_and_inirecon_3(x, y, flag)
    return x


def SPL_Block_2_3(x, y, flag):
    x = wiener_net2_3(x)
    x = sample_and_inirecon_3(x, y, flag)
    x = Biv_Shr2_3(x, flag)
    x = sample_and_inirecon_3(x, y, flag)
    return x


def SPL_Block_3_3(x, y, flag):
    x = wiener_net3_3(x)
    x = sample_and_inirecon_3(x, y, flag)
    x = Biv_Shr3_3(x, flag)
    x = sample_and_inirecon_3(x, y, flag)
    return x


def spynet_compensation_forward_1(x1, x2):
    x1_1 = x1
    x2_1 = x2
    x1_2 = F.interpolate(x1, scale_factor=0.5, mode='bilinear')
    x2_2 = F.interpolate(x2, scale_factor=0.5, mode='bilinear')
    x1_3 = F.interpolate(x1_2, scale_factor=0.5, mode='bilinear')
    x2_3 = F.interpolate(x2_2, scale_factor=0.5, mode='bilinear')
    x1_4 = F.interpolate(x1_3, scale_factor=0.5, mode='bilinear')
    x2_4 = F.interpolate(x2_3, scale_factor=0.5, mode='bilinear')
    x1_5 = F.interpolate(x1_4, scale_factor=0.5, mode='bilinear')
    x2_5 = F.interpolate(x2_4, scale_factor=0.5, mode='bilinear')

    fl_5 = flownet_5_forward_1(x1_5, x2_5)

    fl_5_up = F.interpolate(fl_5, scale_factor=2, mode='bilinear')
    x2_4_warp = FlowBackWarp4(x2_4, fl_5_up)
    fl_4_re = flownet_4_forward_1(x1_4, x2_4_warp, fl_5_up)
    fl_4 = fl_4_re + fl_5_up

    fl_4_up = F.interpolate(fl_4, scale_factor=2, mode='bilinear')
    x2_3_warp = FlowBackWarp3(x2_3, fl_4_up)
    fl_3_re = flownet_3_forward_1(x1_3, x2_3_warp, fl_4_up)
    fl_3 = fl_3_re + fl_4_up

    fl_3_up = F.interpolate(fl_3, scale_factor=2, mode='bilinear')
    x2_2_warp = FlowBackWarp2(x2_2, fl_3_up)
    fl_2_re = flownet_2_forward_1(x1_2, x2_2_warp, fl_3_up)
    fl_2 = fl_2_re + fl_3_up

    fl_2_up = F.interpolate(fl_2, scale_factor=2, mode='bilinear')
    x2_1_warp = FlowBackWarp1(x2_1, fl_2_up)
    fl_1_re = flownet_1_forward_1(x1_1, x2_1_warp, fl_2_up)
    fl_1 = fl_1_re + fl_2_up
    x1_1_pred = FlowBackWarp1(x2_1, fl_1)

    return x1_1_pred, fl_1


def spynet_compensation_backward_1(x1, x2):
    x1_1 = x1
    x2_1 = x2
    x1_2 = F.interpolate(x1, scale_factor=0.5, mode='bilinear')
    x2_2 = F.interpolate(x2, scale_factor=0.5, mode='bilinear')
    x1_3 = F.interpolate(x1_2, scale_factor=0.5, mode='bilinear')
    x2_3 = F.interpolate(x2_2, scale_factor=0.5, mode='bilinear')
    x1_4 = F.interpolate(x1_3, scale_factor=0.5, mode='bilinear')
    x2_4 = F.interpolate(x2_3, scale_factor=0.5, mode='bilinear')
    x1_5 = F.interpolate(x1_4, scale_factor=0.5, mode='bilinear')
    x2_5 = F.interpolate(x2_4, scale_factor=0.5, mode='bilinear')

    fl_5 = flownet_5_backward_1(x1_5, x2_5)

    fl_5_up = F.interpolate(fl_5, scale_factor=2, mode='bilinear')
    x2_4_warp = FlowBackWarp4(x2_4, fl_5_up)
    fl_4_re = flownet_4_backward_1(x1_4, x2_4_warp, fl_5_up)
    fl_4 = fl_4_re + fl_5_up

    fl_4_up = F.interpolate(fl_4, scale_factor=2, mode='bilinear')
    x2_3_warp = FlowBackWarp3(x2_3, fl_4_up)
    fl_3_re = flownet_3_backward_1(x1_3, x2_3_warp, fl_4_up)
    fl_3 = fl_3_re + fl_4_up

    fl_3_up = F.interpolate(fl_3, scale_factor=2, mode='bilinear')
    x2_2_warp = FlowBackWarp2(x2_2, fl_3_up)
    fl_2_re = flownet_2_backward_1(x1_2, x2_2_warp, fl_3_up)
    fl_2 = fl_2_re + fl_3_up

    fl_2_up = F.interpolate(fl_2, scale_factor=2, mode='bilinear')
    x2_1_warp = FlowBackWarp1(x2_1, fl_2_up)
    fl_1_re = flownet_1_backward_1(x1_1, x2_1_warp, fl_2_up)
    fl_1 = fl_1_re + fl_2_up
    x1_1_pred = FlowBackWarp1(x2_1, fl_1)

    return x1_1_pred, fl_1


def spynet_compensation_forward_2(x1, x2):
    x1_1 = x1
    x2_1 = x2
    x1_2 = F.interpolate(x1, scale_factor=0.5, mode='bilinear')
    x2_2 = F.interpolate(x2, scale_factor=0.5, mode='bilinear')
    x1_3 = F.interpolate(x1_2, scale_factor=0.5, mode='bilinear')
    x2_3 = F.interpolate(x2_2, scale_factor=0.5, mode='bilinear')
    x1_4 = F.interpolate(x1_3, scale_factor=0.5, mode='bilinear')
    x2_4 = F.interpolate(x2_3, scale_factor=0.5, mode='bilinear')
    x1_5 = F.interpolate(x1_4, scale_factor=0.5, mode='bilinear')
    x2_5 = F.interpolate(x2_4, scale_factor=0.5, mode='bilinear')

    fl_5 = flownet_5_forward_2(x1_5, x2_5)

    fl_5_up = F.interpolate(fl_5, scale_factor=2, mode='bilinear')
    x2_4_warp = FlowBackWarp4(x2_4, fl_5_up)
    fl_4_re = flownet_4_forward_2(x1_4, x2_4_warp, fl_5_up)
    fl_4 = fl_4_re + fl_5_up

    fl_4_up = F.interpolate(fl_4, scale_factor=2, mode='bilinear')
    x2_3_warp = FlowBackWarp3(x2_3, fl_4_up)
    fl_3_re = flownet_3_forward_2(x1_3, x2_3_warp, fl_4_up)
    fl_3 = fl_3_re + fl_4_up

    fl_3_up = F.interpolate(fl_3, scale_factor=2, mode='bilinear')
    x2_2_warp = FlowBackWarp2(x2_2, fl_3_up)
    fl_2_re = flownet_2_forward_2(x1_2, x2_2_warp, fl_3_up)
    fl_2 = fl_2_re + fl_3_up

    fl_2_up = F.interpolate(fl_2, scale_factor=2, mode='bilinear')
    x2_1_warp = FlowBackWarp1(x2_1, fl_2_up)
    fl_1_re = flownet_1_forward_2(x1_1, x2_1_warp, fl_2_up)
    fl_1 = fl_1_re + fl_2_up
    x1_1_pred = FlowBackWarp1(x2_1, fl_1)

    return x1_1_pred, fl_1


def spynet_compensation_backward_2(x1, x2):
    x1_1 = x1
    x2_1 = x2
    x1_2 = F.interpolate(x1, scale_factor=0.5, mode='bilinear')
    x2_2 = F.interpolate(x2, scale_factor=0.5, mode='bilinear')
    x1_3 = F.interpolate(x1_2, scale_factor=0.5, mode='bilinear')
    x2_3 = F.interpolate(x2_2, scale_factor=0.5, mode='bilinear')
    x1_4 = F.interpolate(x1_3, scale_factor=0.5, mode='bilinear')
    x2_4 = F.interpolate(x2_3, scale_factor=0.5, mode='bilinear')
    x1_5 = F.interpolate(x1_4, scale_factor=0.5, mode='bilinear')
    x2_5 = F.interpolate(x2_4, scale_factor=0.5, mode='bilinear')

    fl_5 = flownet_5_backward_2(x1_5, x2_5)

    fl_5_up = F.interpolate(fl_5, scale_factor=2, mode='bilinear')
    x2_4_warp = FlowBackWarp4(x2_4, fl_5_up)
    fl_4_re = flownet_4_backward_2(x1_4, x2_4_warp, fl_5_up)
    fl_4 = fl_4_re + fl_5_up

    fl_4_up = F.interpolate(fl_4, scale_factor=2, mode='bilinear')
    x2_3_warp = FlowBackWarp3(x2_3, fl_4_up)
    fl_3_re = flownet_3_backward_2(x1_3, x2_3_warp, fl_4_up)
    fl_3 = fl_3_re + fl_4_up

    fl_3_up = F.interpolate(fl_3, scale_factor=2, mode='bilinear')
    x2_2_warp = FlowBackWarp2(x2_2, fl_3_up)
    fl_2_re = flownet_2_backward_2(x1_2, x2_2_warp, fl_3_up)
    fl_2 = fl_2_re + fl_3_up

    fl_2_up = F.interpolate(fl_2, scale_factor=2, mode='bilinear')
    x2_1_warp = FlowBackWarp1(x2_1, fl_2_up)
    fl_1_re = flownet_1_backward_2(x1_1, x2_1_warp, fl_2_up)
    fl_1 = fl_1_re + fl_2_up
    x1_1_pred = FlowBackWarp1(x2_1, fl_1)

    return x1_1_pred, fl_1


def load_img(filepath):
    img = cv2.imread(filepath, flags=cv2.IMREAD_GRAYSCALE)
    # img = img[:, :, np.newaxis]
    # img = np.pad(img, ((0, 16), (0, 16)), 'constant', constant_values=0)
    return img


def get_single_image(image_path):
    # print(image_path)
    if rgb:
        Img = cv2.imread(image_path, 1)
        Img_yuv = cv2.cvtColor(Img, cv2.COLOR_BGR2YCrCb)
        Img_rec_yuv = Img_yuv.copy()
        Iorg_y = Img_yuv[:, :, 0]
    else:
        # print(image_path)
        Img_yuv = cv2.imread(image_path, flags=cv2.IMREAD_GRAYSCALE)
        Img_rec_yuv = Img_yuv.copy()
        Iorg_y = Img_yuv

    # image = load_img(image_path)
    input_compose = Compose([ToTensor()])
    # image = input_compose(image)
    image = input_compose(Iorg_y)

    inputs = image[0, :, :]
    ih = inputs.shape[0]
    iw = inputs.shape[1]

    if np.mod(iw, block_size) != 0:
        col_pad = block_size - np.mod(iw, block_size)
        inputs = torch.cat((inputs, torch.zeros([ih, col_pad])), axis=1)
    else:
        col_pad = 0
        inputs = inputs
    if np.mod(ih, block_size) != 0:
        row_pad = block_size - np.mod(ih, block_size)
        inputs = torch.cat((inputs, torch.zeros([row_pad, iw + col_pad])), axis=0)
    else:
        row_pad = 0
    inputs = inputs.cuda()
    # I = I.cuda()

    # print(inputs.shape)
    # print(torch.max(image)*255)
    # sys.exit()
    return inputs, Img_rec_yuv


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", "tif"])


def PSNR(data, tar):
    data = torch.squeeze(data)
    tar = torch.squeeze(tar)
    data = data.cpu().numpy()
    tar = tar.cpu().numpy()

    data = data.astype(np.float32)
    tar = tar.astype(np.float32)

    max_gray = 1.
    mse = np.mean(np.power(data - tar, 2))

    return 10. * np.log10(max_gray ** 2 / mse)


if __name__ == '__main__':
    workbook = xlwt.Workbook(encoding='utf-8')
    video_index = 0
    worksheet = workbook.add_sheet('text_video_psnr_ssim')
    worksheet_1 = workbook.add_sheet('text_video_psnr_ssim_ave')
    psnr_ave = []
    ssim_ave = []

    # temp_psnr = []
    # temp_ssim = []
    for video_name in sorted(os.listdir('./' + test_video_name)):
        psnr_ave = []
        ssim_ave = []
        worksheet.write(0, video_index, video_name + '_psnr')
        worksheet.write(0, video_index + 1, video_name + '_ssim')
        print(video_name)
        file_path = './' + test_video_name + '/' + video_name
        if not os.path.exists('./' + test_video_name + '_output'):
            os.mkdir('./' + test_video_name + '_output')
        if not os.path.exists('./' + test_video_name + '_output/' + video_name + str(sr)):
            os.mkdir('./' + test_video_name + '_output/' + video_name + str(sr))
        if not os.path.exists('./' + test_video_name + '_output/psnr_ssim'):
            os.mkdir('./' + test_video_name + '_output/psnr_ssim')

        input_tensor = torch.zeros([1, gop_size + 1, 1, image_height, image_width])
        file = listdir(file_path)
        file.sort()
        image_filenames = [join(file_path, x) for x in file if is_image_file(x)]
        # num_gop = len(image_filenames) / gop_size

        img_index = 1
        with torch.no_grad():
            for i in range(int(num_gop)):
                if img_index == (len(image_filenames) - 8):
                    break
                yuv_recon = []
                start = time.time()
                for j in range(int(gop_size) + 1):
                    # according to the index format
                    # s = "%03d" % (j + i * gop_size + 1)
                    # s = 'im' + s
                    # image_name = s + '.png'
                    s = "%06d" % (j + i * gop_size)
                    image_name = s + '.tif'

                    image_path = os.path.join(file_path, image_name)
                    single_image, single_yuv = get_single_image(image_path)
                    yuv_recon.append(single_yuv)
                    input_tensor[0, j, 0, :, :] = single_image

                frame0 = input_tensor[:, 0, :, :, :]
                frame8 = input_tensor[:, 8, :, :, :]
                F_0 = frame0.to(device)
                F_8 = frame8.to(device)
                # key_initial reconstruction
                ini_x_key0, sample_F_0 = sample_and_inirecon_key(F_0, 0, False)
                spl1_x_key0 = SPL_Block_1_key(ini_x_key0, sample_F_0, True)
                spl2_x_key0 = SPL_Block_2_key(spl1_x_key0, sample_F_0, True)
                out_F_0 = SPL_Block_3_key(spl2_x_key0, sample_F_0, True)

                ini_x_key8, sample_F_8 = sample_and_inirecon_key(F_8, 0, False)
                spl1_x_key8 = SPL_Block_1_key(ini_x_key8, sample_F_8, True)
                spl2_x_key8 = SPL_Block_2_key(spl1_x_key8, sample_F_8, True)
                out_F_8 = SPL_Block_3_key(spl2_x_key8, sample_F_8, True)

                psnr_temp = PSNR(out_F_0[:, :, :img_h, :img_w], F_0[:, :, :img_h, :img_w])
                print(psnr_temp)

                # torchvision.utils.save_image(out_F_0.float(),
                #                              './test_video_cif_output/mobile_0.05/' + str(
                #                                  img_index) + '_' + str(
                #                                  psnr_temp) + '.png',
                #                              normalize=True)

                out_F_0numpy = out_F_0.cpu().numpy()
                out_F_0numpy = out_F_0numpy[0, 0, :img_h, :img_w]
                F_0numpy = F_0.cpu().numpy()
                F_0numpy = F_0numpy[0, 0, :img_h, :img_w]

                ssim_temp = compute_ssim(np.around(out_F_0numpy * 255), np.around(F_0numpy * 255))

                Prediction_value = out_F_0.cpu().data.numpy().squeeze()
                X_rec = np.clip(Prediction_value, 0, 1).astype(np.float64)
                if rgb:
                    yuv_recon[0][:, :, 0] = X_rec[:img_h, :img_w] * 255
                    im_rec_rgb = cv2.cvtColor(yuv_recon[0], cv2.COLOR_YCrCb2BGR)
                    im_rec_rgb = np.clip(im_rec_rgb, 0, 255).astype(np.uint8)

                    cv2.imwrite(test_video_name + "_output/" + video_name + str(sr) + '/' + str(img_index) + '_' + str(
                        psnr_temp) + str(ssim_temp) + '.png', im_rec_rgb)
                else:
                    yuv_recon[0][:, :] = X_rec[:img_h, :img_w] * 255
                    im_rec_y = np.clip(yuv_recon[0][:, :], 0, 255).astype(np.uint8)
                    cv2.imwrite(test_video_name + "_output/" + video_name + str(sr) + '/' + str(img_index) + '_' + str(
                        psnr_temp) + str(ssim_temp) + '.png', im_rec_y)

                psnr_ave.append(psnr_temp)
                ssim_ave.append(ssim_temp)
                worksheet.write(img_index + 1, video_index, psnr_temp)
                worksheet.write(img_index + 1, video_index + 1, ssim_temp)
                img_index = img_index + 1
                # f = open('./test_video/psnr_ssim/' + video_name + '_output/final/' + video_name + '_psnr_final.txt', 'a')
                # f.write(" %0.6f " % (psnr_temp))
                # f.write('\n')
                # f.close()
                # f = open('./test_video/psnr_ssim/' + video_name + '_output/final/' + video_name + '_ssim_final.txt', 'a')
                # f.write(" %0.6f " % (ssim_temp))
                # f.write('\n')
                # f.close()

                for k in range(int(gop_size) - 1):
                    frame1 = input_tensor[:, k + 1, :, :, :]
                    F_1 = frame1.to(device)
                    # non_key_initial reconstruction
                    ini_x_nonkey, y_F_1 = sample_and_inirecon_nonkey(F_1, 0, False)
                    spl1_x_nonkey = SPL_Block_1_nonkey(ini_x_nonkey, y_F_1, True)
                    spl2_x_nonkey = SPL_Block_2_nonkey(spl1_x_nonkey, y_F_1, True)
                    out_F_1_pre = SPL_Block_3_nonkey(spl2_x_nonkey, y_F_1, True)

                    # Forward and backward optical flow estimation and motion compensation
                    f1_1_pred_fw_1, flow_forward_output_1 = spynet_compensation_forward_1(
                        out_F_1_pre, out_F_0)
                    f1_1_pred_bw_1, flow_backward_output_1 = spynet_compensation_backward_1(
                        out_F_1_pre, out_F_8)

                    # fusion
                    prob_1 = Prob_net_1(f1_1_pred_fw_1, f1_1_pred_bw_1, out_F_1_pre)
                    out_F_1_1 = prob_1[:, [0], :, :] * f1_1_pred_fw_1 + prob_1[:, [1], :, :] * f1_1_pred_bw_1 + prob_1[
                                                                                                                :,
                                                                                                                [2],
                                                                                                                :,
                                                                                                                :] * out_F_1_pre

                    # residual2 reconstruction
                    ini_x_2, y_residual_2 = sample_and_inirecon_2(out_F_1_1, y_F_1, False)
                    spl1_x_2 = SPL_Block_1_2(ini_x_2, y_residual_2, True)
                    spl2_x_2 = SPL_Block_2_2(spl1_x_2, y_residual_2, True)
                    out_F_1_residual = SPL_Block_3_2(spl2_x_2, y_residual_2, True)
                    out_F_1_2 = out_F_1_1 + out_F_1_residual

                    # Forward and backward optical flow estimation and motion compensation
                    f1_1_pred_fw_2, flow_forward_output_2 = spynet_compensation_forward_2(
                        out_F_1_2, out_F_0)
                    f1_1_pred_bw_2, flow_backward_output_2 = spynet_compensation_backward_2(
                        out_F_1_2, out_F_8)

                    # fusion
                    prob_2 = Prob_net_2(f1_1_pred_fw_2, f1_1_pred_bw_2, out_F_1_2)
                    out_F_1_3 = prob_2[:, [0], :, :] * f1_1_pred_fw_2 + prob_2[:, [1], :, :] * f1_1_pred_bw_2 + prob_2[
                                                                                                                :,
                                                                                                                [2],
                                                                                                                :,
                                                                                                                :] * out_F_1_2

                    # residual3 reconstruction
                    ini_x_3, y_residual_3 = sample_and_inirecon_3(out_F_1_3, y_F_1, False)
                    spl1_x_3 = SPL_Block_1_3(ini_x_3, y_residual_3, True)
                    spl2_x_3 = SPL_Block_2_3(spl1_x_3, y_residual_3, True)
                    out_F_1_residual_3 = SPL_Block_3_3(spl2_x_3, y_residual_3, True)
                    out_F_1_4 = out_F_1_3 + out_F_1_residual_3

                    psnr_temp = PSNR(out_F_1_4[:, :, :img_h, :img_w], F_1[:, :, :img_h, :img_w])
                    print(psnr_temp)
                    # torchvision.utils.save_image(out_F_1_4.float(),
                    #                              './test_video_cif_output/mobile_0.05/' + str(
                    #                                  img_index) + '_' + str(
                    #                                  psnr_temp) + '.png',
                    #                              normalize=True)

                    # ssim_temp = pytorch_ssim.ssim(out_F_1_4, F_1)
                    # ssim_temp_cpu = ssim_temp.cpu()

                    out_F_1_4numpy = out_F_1_4.cpu().numpy()
                    out_F_1_4numpy = out_F_1_4numpy[0, 0, :img_h, :img_w]
                    F_1numpy = F_1.cpu().numpy()
                    F_1numpy = F_1numpy[0, 0, :img_h, :img_w]

                    ssim_temp = compute_ssim(np.around(out_F_1_4numpy * 255), np.around(F_1numpy * 255))

                    Prediction_value = out_F_1_4.cpu().data.numpy().squeeze()
                    X_rec = np.clip(Prediction_value, 0, 1).astype(np.float64)
                    if rgb:
                        yuv_recon[0][:, :, 0] = X_rec[:img_h, :img_w] * 255
                        im_rec_rgb = cv2.cvtColor(yuv_recon[0], cv2.COLOR_YCrCb2BGR)
                        im_rec_rgb = np.clip(im_rec_rgb, 0, 255).astype(np.uint8)

                        cv2.imwrite(
                            test_video_name + "_output/" + video_name + str(sr) + '/' + str(img_index) + '_' + str(
                                psnr_temp) + str(ssim_temp) + '.png', im_rec_rgb)
                    else:
                        yuv_recon[0][:, :] = X_rec[:img_h, :img_w] * 255
                        im_rec_y = np.clip(yuv_recon[0][:, :], 0, 255).astype(np.uint8)
                        cv2.imwrite(
                            test_video_name + "_output/" + video_name + str(sr) + '/' + str(img_index) + '_' + str(
                                psnr_temp) + str(ssim_temp) + '.png', im_rec_y)

                    # f = open('./test_video/psnr_ssim/' + video_name + '_output/final/' + video_name + '_psnr_final.txt', 'a')
                    # f.write(" %0.6f " % (psnr_temp))
                    # f.write('\n')
                    # f.close()
                    # f = open('./test_video/psnr_ssim/' + video_name + '_output/final/' + video_name + '_ssim_final.txt', 'a')
                    # f.write(" %0.6f " % (ssim_temp))
                    # f.write('\n')
                    # f.close()

                    worksheet.write(img_index + 1, video_index, psnr_temp)
                    worksheet.write(img_index + 1, video_index + 1, ssim_temp)
                    psnr_ave.append(psnr_temp)
                    ssim_ave.append(ssim_temp)
                    img_index = img_index + 1
            psnr_ave = psnr_ave[0:gop_size * num_gop]
            print(psnr_ave)
            ssim_ave = ssim_ave[0:gop_size * num_gop]
            worksheet_1.write(0, video_index, video_name + '_psnr')
            worksheet_1.write(0, video_index + 1, video_name + '_ssim')
            worksheet_1.write(1, video_index, np.mean(psnr_ave))
            worksheet_1.write(1, video_index + 1, np.mean(ssim_ave))
            video_index = video_index + 2
    workbook.save('./' + test_video_name + '_output/psnr_ssim/psnr_ssim_0.5_' + str(sr) + '.xls')
    sys.exit()