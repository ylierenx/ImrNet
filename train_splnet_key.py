import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

# model.py and dataloder1.py are model code and data loading code
import model
import dataloder1

from math import log10
import sys
import time
# import seaborn as sns
# import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.image as mpimg

# set GPU ID
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda:0")

# Whether to load the model
flag = False

# sample_and_inirecon, Biv_Shr and wiener_net are sampling and initial reconstruction,
# bivariate shrinkage and Wiener filter, respectively
# num_filters indicates the number of sampling points of the BCS sampling block, and B_size indicates the size of the sampling block
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

# Get a list of training parameters
params_ininet = list(sample_and_inirecon_key.parameters())
params_Biv_Shr = list(Biv_Shr1_key.parameters()) + list(Biv_Shr2_key.parameters()) + list(Biv_Shr3_key.parameters())
params_wienet = list(wiener_net1_key.parameters()) + list(wiener_net2_key.parameters()) + list(
    wiener_net3_key.parameters())
params_splnet = params_ininet + params_Biv_Shr + params_wienet

# Choose optim as the optimizer and set the initial learning rate to 0.0001
optimizer_splnet = optim.Adam(params_splnet, lr=0.0001)

# load model
if flag:
    dict1 = torch.load('./check_point1/splnet_0.5_1.ckpt')
    sample_and_inirecon_key.load_state_dict(dict1['state_dict_sample_and_inirecon_nonkey'])
    Biv_Shr1_key.load_state_dict(dict1['state_dict_Biv_Shr1_nonkey'])
    Biv_Shr2_key.load_state_dict(dict1['state_dict_Biv_Shr2_nonkey'])
    Biv_Shr3_key.load_state_dict(dict1['state_dict_Biv_Shr3_nonkey'])
    wiener_net1_key.load_state_dict(dict1['state_dict_wiener_net1_nonkey'])
    wiener_net2_key.load_state_dict(dict1['state_dict_wiener_net2_nonkey'])
    wiener_net3_key.load_state_dict(dict1['state_dict_wiener_net3_nonkey'])
    optimizer_splnet.load_state_dict(dict1['state_dict_optimizer_splnet'])
else:
    dict1 = {'epoch': -1}

# Set the training set parameters, gop_size=1 means for the key frame network, image_size means the size of the training image,
# batch_size means the number of batch samples, and shuffle means whether the overall training samples are randomly sorted each time
# num_workers indicates the number of data reading threads during training, generally set to 2,
# adding more can increase the speed of reading data, and will take up more cpu resources
# pin_memory indicates whether to use virtual memory, set to True
trainset = dataloder1.UCF101(gop_size=1, image_size=160)
train_loader = dataloder1.DataLoader(trainset, batch_size=2, shuffle=True, num_workers=1, pin_memory=True)

# Indicates the model number when saving the model
checkpoint_counter = 1


def psnr(img_rec, img_ori):
    img_rec = img_rec.astype(np.float32)
    img_ori = img_ori.astype(np.float32)
    max_gray = 1.
    mse = np.mean(np.power(img_rec - img_ori, 2))
    return 10. * np.log10(max_gray ** 2 / mse)


# The specific implementation of each BLOCK of SPLNet
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


# Adjust learning rate
def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# Test model during training
def test_splnet(device):
    I_ini = mpimg.imread('./test1/000001.tif')
    I_ini = I_ini / 255.
    I = np.pad(I_ini, ((0, 16), (0, 16)), 'constant', constant_values=0)
    I = I[np.newaxis, np.newaxis, :, :]
    with torch.no_grad():
        inputs = torch.tensor(I, device=device).float()
        ini_x, phi_x = sample_and_inirecon_key(inputs, 0, False)
        spl1_x = SPL_Block_1_key(ini_x, phi_x, True)
        spl2_x = SPL_Block_2_key(spl1_x, phi_x, True)
        spl3_x = SPL_Block_3_key(spl2_x, phi_x, True)
        spl3_x = spl3_x.cpu().numpy()
        recon_img = spl3_x[0, 0, :, :]
        recon_img = recon_img[0:144, 0:176]
        p1 = psnr(recon_img, I_ini)
    return p1


if __name__ == '__main__':

    start = time.time()
    for epoch in range(0, 501):
        print("Epoch: ", epoch)

        # Reduce the learning rate at a certain epoch
        if epoch >= 300 and epoch < 400:
            lr = 0.00001
            adjust_learning_rate(optimizer_splnet, lr)
        if epoch >= 400:
            lr = 0.000001
            adjust_learning_rate(optimizer_splnet, lr)

        # training
        for i, inputs in enumerate(train_loader):

            inputs = inputs[:, 0, :, :, :]
            inputs = inputs.to(device)
            optimizer_splnet.zero_grad()
            ini_x, phi_x = sample_and_inirecon_key(inputs, 0, False)
            spl1_x = SPL_Block_1_key(ini_x, phi_x, True)
            spl2_x = SPL_Block_2_key(spl1_x, phi_x, True)
            spl3_x = SPL_Block_3_key(spl2_x, phi_x, True)
            biv_1 = Biv_Shr1_key(inputs, False)
            biv_2 = Biv_Shr2_key(inputs, False)
            biv_3 = Biv_Shr3_key(inputs, False)

            recnLoss = torch.mean(
                torch.norm((inputs - spl3_x), p=2, dim=(2, 3)) * torch.norm((inputs - spl3_x), p=2, dim=(2, 3)))

            constrant_loss1 = torch.mean(
                torch.norm((inputs - biv_1), p=2, dim=(2, 3)) * torch.norm((inputs - biv_1), p=2, dim=(2, 3)))
            constrant_loss2 = torch.mean(
                torch.norm((inputs - biv_2), p=2, dim=(2, 3)) * torch.norm((inputs - biv_2), p=2, dim=(2, 3)))
            constrant_loss3 = torch.mean(
                torch.norm((inputs - biv_3), p=2, dim=(2, 3)) * torch.norm((inputs - biv_3), p=2, dim=(2, 3)))
            constrant_loss4 = torch.mean(
                torch.norm((inputs - ini_x), p=2, dim=(2, 3)) * torch.norm((inputs - ini_x), p=2, dim=(2, 3)))

            recnLoss_all = recnLoss + 0.01 * (constrant_loss1 + constrant_loss2 + constrant_loss3 + constrant_loss4)

            recnLoss_all.backward()
            optimizer_splnet.step()

            # print and save loss
            if ((i % 40) == 0):
                print('test')
                p1 = test_splnet(device)
                print(p1)
                print(" train_loss: %0.6f  Iterations: %4d/%4d epoch:%d " % (
                    recnLoss_all.item(), i, len(train_loader), epoch))

                f = open('train_loss_0.5.txt', 'a')
                f.write(" %0.6f %d " % (recnLoss_all.item(), epoch))
                f.write('\n')
                f.close()

                end = time.time()
                print(end - start)
                start = time.time()

        # save model
        if ((epoch % 50) == 0 and epoch > 0):
            dict1 = {
                'Detail': "splnet_key",
                'epoch': epoch,

                'state_dict_sample_and_inirecon_key': sample_and_inirecon_key.state_dict(),
                'state_dict_Biv_Shr1_key': Biv_Shr1_key.state_dict(),
                'state_dict_Biv_Shr2_key': Biv_Shr2_key.state_dict(),
                'state_dict_Biv_Shr3_key': Biv_Shr3_key.state_dict(),
                'state_dict_wiener_net1_key': wiener_net1_key.state_dict(),
                'state_dict_wiener_net2_key': wiener_net2_key.state_dict(),
                'state_dict_wiener_net3_key': wiener_net3_key.state_dict(),
                'state_dict_optimizer_splnet': optimizer_splnet.state_dict()

            }
            torch.save(dict1, './check_point1' + "/splnet_0.5_" + str(checkpoint_counter) + ".ckpt")
            checkpoint_counter += 1













