import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

# model.py and dataloder1.py are model code and data loading code
import model
import dataloder

from math import log10
import sys
import time
# import seaborn as sns
# import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.image as mpimg

# set GPU ID
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from torchvision.utils import save_image

from torch.utils.data import Dataset, DataLoader

flag = True

device = torch.device("cuda:0")

# nonkey initial reconstruction
sample_and_inirecon_nonkey = model.sample_and_inirecon(num_filters=102, B_size=32)
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
FlowBackWarp5 = model.backWarp(W=160 / (pow(2, 4)), H=160 / (pow(2, 4)), device=device)
FlowBackWarp5.to(device)
FlowBackWarp4 = model.backWarp(W=160 / (pow(2, 3)), H=160 / (pow(2, 3)), device=device)
FlowBackWarp4.to(device)
FlowBackWarp3 = model.backWarp(W=160 / (pow(2, 2)), H=160 / (pow(2, 2)), device=device)
FlowBackWarp3.to(device)
FlowBackWarp2 = model.backWarp(W=160 / (pow(2, 1)), H=160 / (pow(2, 1)), device=device)
FlowBackWarp2.to(device)
FlowBackWarp1 = model.backWarp(W=160, H=160, device=device)
FlowBackWarp1.to(device)

# probnet
Prob_net_1 = model.Prob_net()
Prob_net_1.to(device)

trainset = dataloder.UCF101(gop_size=9, image_size=160)
train_loader = dataloder.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=1, pin_memory=False)
params_flow_forward = list(flownet_1_forward_1.parameters()) + list(flownet_2_forward_1.parameters()) + list(flownet_3_forward_1.parameters()) + list(flownet_4_forward_1.parameters()) + list(flownet_5_forward_1.parameters())
params_flow_backward = list(flownet_1_backward_1.parameters()) + list(flownet_2_backward_1.parameters()) + list(flownet_3_backward_1.parameters()) + list(flownet_4_backward_1.parameters()) + list(flownet_5_backward_1.parameters())
params_prob = list(Prob_net_1.parameters())
params_all_1 = params_flow_forward + params_flow_backward + params_prob
optimizer_all_1 = optim.Adam(params_all_1, lr=0.0001)

if flag:
    dict1 = torch.load('./check_point/splnet_0.5.ckpt')
    sample_and_inirecon_nonkey.load_state_dict(dict1['state_dict_sample_and_inirecon_nonkey'])
    Biv_Shr1_nonkey.load_state_dict(dict1['state_dict_Biv_Shr1_nonkey'])
    Biv_Shr2_nonkey.load_state_dict(dict1['state_dict_Biv_Shr2_nonkey'])
    Biv_Shr3_nonkey.load_state_dict(dict1['state_dict_Biv_Shr3_nonkey'])
    wiener_net1_nonkey.load_state_dict(dict1['state_dict_wiener_net1_nonkey'])
    wiener_net2_nonkey.load_state_dict(dict1['state_dict_wiener_net2_nonkey'])
    wiener_net3_nonkey.load_state_dict(dict1['state_dict_wiener_net3_nonkey'])
    dict1 = torch.load('./check_point/splnet_0.1.ckpt')
    sample_and_inirecon_key.load_state_dict(dict1['state_dict_sample_and_inirecon_key'])
    Biv_Shr1_key.load_state_dict(dict1['state_dict_Biv_Shr1_key'])
    Biv_Shr2_key.load_state_dict(dict1['state_dict_Biv_Shr2_key'])
    Biv_Shr3_key.load_state_dict(dict1['state_dict_Biv_Shr3_key'])
    wiener_net1_key.load_state_dict(dict1['state_dict_wiener_net1_key'])
    wiener_net2_key.load_state_dict(dict1['state_dict_wiener_net2_key'])
    wiener_net3_key.load_state_dict(dict1['state_dict_wiener_net3_key'])
else:
    dict1 = {'epoch': -1}

checkpoint_counter = 1

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

if __name__ == '__main__':

    for epoch in range(dict1['epoch'] + 1, 3601):
        print("Epoch: ", epoch)

        start = time.time()
        for i, (inputs, f_index) in enumerate(train_loader):

            frame0 = inputs[:, 0, :, :, :]
            frame1 = inputs[:, 1, :, :, :]
            frame8 = inputs[:, 2, :, :, :]
            F_0 = frame0.to(device)
            F_1 = frame1.to(device)
            F_8 = frame8.to(device)
            optimizer_all_1.zero_grad()

            # non_key_initial reconstruction
            ini_x_nonkey, y_F_1 = sample_and_inirecon_nonkey(F_1, 0, False)
            spl1_x_nonkey = SPL_Block_1_nonkey(ini_x_nonkey, y_F_1, True)
            spl2_x_nonkey = SPL_Block_2_nonkey(spl1_x_nonkey, y_F_1, True)
            out_F_1_pre = SPL_Block_3_nonkey(spl2_x_nonkey, y_F_1, True)

            # key_initial reconstruction
            ini_x_key0, sample_F_0 = sample_and_inirecon_key(F_0, 0, False)
            spl1_x_key0 = SPL_Block_1_key(ini_x_key0, sample_F_0, True)
            spl2_x_key0 = SPL_Block_2_key(spl1_x_key0, sample_F_0, True)
            out_F_0 = SPL_Block_3_key(spl2_x_key0, sample_F_0, True)

            ini_x_key8, sample_F_8 = sample_and_inirecon_key(F_8, 0, False)
            spl1_x_key8 = SPL_Block_1_key(ini_x_key8, sample_F_8, True)
            spl2_x_key8 = SPL_Block_2_key(spl1_x_key8, sample_F_8, True)
            out_F_8 = SPL_Block_3_key(spl2_x_key8, sample_F_8, True)

            # Forward and backward optical flow estimation and motion compensation
            f1_1_pred_fw_1, flow_forward_output_1 = spynet_compensation_forward_1(
                out_F_1_pre, out_F_0)
            f1_1_pred_bw_1, flow_backward_output_1 = spynet_compensation_backward_1(
                out_F_1_pre, out_F_8)

            #fusion
            prob_1 = Prob_net_1(f1_1_pred_fw_1, f1_1_pred_bw_1, out_F_1_pre)
            out_F_1_1 = prob_1[:, [0], :, :] * f1_1_pred_fw_1 + prob_1[:, [1], :, :] * f1_1_pred_bw_1 + prob_1[:, [2],
                                                                                                        :,
                                                                                                        :] * out_F_1_pre

            recnLoss_8 = torch.mean(
                torch.norm((F_8 - out_F_8), p=2, dim=(2, 3)) * torch.norm((F_8 - out_F_8), p=2, dim=(2, 3)))
            recnLoss_0 = torch.mean(
                torch.norm((F_0 - out_F_0), p=2, dim=(2, 3)) * torch.norm((F_0 - out_F_0), p=2, dim=(2, 3)))
            recnLoss_1_pre = torch.mean(
                torch.norm((F_1 - out_F_1_pre), p=2, dim=(2, 3)) * torch.norm((F_1 - out_F_1_pre), p=2, dim=(2, 3)))
            recnLoss_1_1 = torch.mean(
                torch.norm((F_1 - out_F_1_1), p=2, dim=(2, 3)) * torch.norm((F_1 - out_F_1_1), p=2, dim=(2, 3)))
            recnLoss_1_forward_1 = torch.mean(
                torch.norm((F_1 - f1_1_pred_fw_1), p=2, dim=(2, 3)) * torch.norm((F_1 - f1_1_pred_fw_1), p=2,
                                                                                 dim=(2, 3)))
            recnLoss_1_backward_1 = torch.mean(
                torch.norm((F_1 - f1_1_pred_bw_1), p=2, dim=(2, 3)) * torch.norm((F_1 - f1_1_pred_bw_1), p=2,
                                                                                 dim=(2, 3)))

            all_loss = recnLoss_1_forward_1 + recnLoss_1_backward_1 + recnLoss_1_1
            all_loss.backward()
            optimizer_all_1.step()

            if ((i % 100) == 0):
                print(f_index + 1)
                print(
                    " L0: %0.6f  L_i_pre: %0.6f L_i_forward: %0.6f L_i_backward: %0.6f  L8: %0.6f L_fusion: %0.6f Iterations: %4d/%4d  " % (
                        recnLoss_0.item(), recnLoss_1_pre.item(), recnLoss_1_forward_1.item(),
                        recnLoss_1_backward_1.item(),
                        recnLoss_8.item(), recnLoss_1_1.item(), i, len(train_loader)))
                f = open('train_0.5_0.01_step1.txt', 'a')
                f.write(" %0.6f %0.6f %d " % (recnLoss_1_forward_1.item(), recnLoss_1_backward_1.item(), epoch))
                f.write('\n')
                f.close()

        end = time.time()
        print(end - start)
        if ((epoch % 100) == 0 and epoch > 0):
            dict1 = {
                'Detail': "splnet_optical_video",
                'epoch': epoch,
                'state_dict_sample_and_inirecon_key': sample_and_inirecon_key.state_dict(),
                'state_dict_Biv_Shr1_key': Biv_Shr1_key.state_dict(),
                'state_dict_Biv_Shr2_key': Biv_Shr2_key.state_dict(),
                'state_dict_Biv_Shr3_key': Biv_Shr3_key.state_dict(),
                'state_dict_wiener_net1_key': wiener_net1_key.state_dict(),
                'state_dict_wiener_net2_key': wiener_net2_key.state_dict(),
                'state_dict_wiener_net3_key': wiener_net3_key.state_dict(),
                'state_dict_optimizer_all_1': optimizer_all_1.state_dict(),
                'state_dict_sample_and_inirecon_nonkey': sample_and_inirecon_nonkey.state_dict(),
                'state_dict_Biv_Shr1_nonkey': Biv_Shr1_nonkey.state_dict(),
                'state_dict_Biv_Shr2_nonkey': Biv_Shr2_nonkey.state_dict(),
                'state_dict_Biv_Shr3_nonkey': Biv_Shr3_nonkey.state_dict(),
                'state_dict_wiener_net1_nonkey': wiener_net1_nonkey.state_dict(),
                'state_dict_wiener_net2_nonkey': wiener_net2_nonkey.state_dict(),
                'state_dict_wiener_net3_nonkey': wiener_net3_nonkey.state_dict(),

                'state_dict_flownet_1_forward_1': flownet_1_forward_1.state_dict(),
                'state_dict_flownet_2_forward_1': flownet_2_forward_1.state_dict(),
                'state_dict_flownet_3_forward_1': flownet_3_forward_1.state_dict(),
                'state_dict_flownet_4_forward_1': flownet_4_forward_1.state_dict(),
                'state_dict_flownet_5_forward_1': flownet_5_forward_1.state_dict(),
                'state_dict_flownet_1_backward_1': flownet_1_backward_1.state_dict(),
                'state_dict_flownet_2_backward_1': flownet_2_backward_1.state_dict(),
                'state_dict_flownet_3_backward_1': flownet_3_backward_1.state_dict(),
                'state_dict_flownet_4_backward_1': flownet_4_backward_1.state_dict(),
                'state_dict_flownet_5_backward_1': flownet_5_backward_1.state_dict(),
                'state_dict_Prob_net_1': Prob_net_1.state_dict()

            }
            torch.save(dict1, './check_point1' + "/spl_optical_flow_video_step1_0.5_0.1_" + str(checkpoint_counter) + ".ckpt")
            checkpoint_counter += 1

