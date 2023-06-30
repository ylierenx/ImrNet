import torch

import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import sys


def tensor2image(x):
    [batch_size, c_size, h_size, w_size] = list(x.shape)
    x = x.permute(0, 2, 3, 1)
    B_size = int(math.sqrt(float(c_size)))
    subset_x = []
    x = x.flatten(start_dim=1)
    for i in range(h_size * w_size):
        slice_x = x[:, i * c_size:(i + 1) * c_size]
        slice_x = slice_x.reshape(-1, B_size, B_size)
        subset_x.append(slice_x)
    x1 = torch.cat(subset_x, dim=-1)
    subset_x = []
    for i in range(h_size):
        slice_x = x1[:, :, i * (B_size * w_size):(i + 1) * (B_size * w_size)]
        subset_x.append(slice_x)
    x1 = torch.cat(subset_x, dim=-2)
    x1 = torch.unsqueeze(x1, dim=3)
    x1 = x1.permute(0, 3, 1, 2)
    return x1



class sample_net_transpose(nn.Module):
    def __init__(self, num_filters, B_size, init_weight):
        super(sample_net_transpose, self).__init__()

        init_weight = torch.reshape(init_weight, (num_filters, 1, (B_size * B_size)))
        init_weight = init_weight.permute(2, 0, 1)
        init_weight = torch.unsqueeze(init_weight, 3)
        init_weight = torch.nn.Parameter(init_weight)
        self.sample_transpose = nn.Conv2d(in_channels=num_filters, out_channels=(B_size * B_size), kernel_size=1, stride=1, padding=0,
                                bias=False)
        self.sample_transpose.weight = init_weight

    def forward(self, x):
        self.output = self.sample_transpose(x)
        self.output = tensor2image(self.output)
        return self.output



class prob_net_BW(nn.Module):
    def __init__(self):
        super(prob_net_BW, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = self.conv1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = F.sigmoid(x)
        return x


class CSNet(nn.Module):
    def __init__(self, num_filters, B_size):
        super(CSNet, self).__init__()

        self.ini_rec = nn.Conv2d(in_channels=num_filters, out_channels=(B_size * B_size), kernel_size=1, stride=1,
                                 padding=0, bias=False)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x = x
        self.output = self.ini_rec(x)
        self.output = tensor2image(self.output)
        self.output = self.conv1(self.output)
        self.output = F.relu(self.output, inplace=True)
        self.output = self.conv2(self.output)
        self.output = F.relu(self.output, inplace=True)
        self.output = self.conv3(self.output)
        self.output = F.relu(self.output, inplace=True)
        self.output = self.conv4(self.output)
        self.output = F.relu(self.output, inplace=True)
        self.output = self.conv5(self.output)
        return self.output

class flownet_first(nn.Module):
    def __init__(self):
        super(flownet_first, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3)
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = self.conv1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = F.relu(x, inplace=True)
        x = self.conv3(x)
        x = F.relu(x, inplace=True)
        x = self.conv4(x)
        x = F.relu(x, inplace=True)
        x = self.conv5(x)
        return x

class flownet_first_new(nn.Module):
    def __init__(self):
        super(flownet_first_new, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3)
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
    def forward(self, x1, x2, x3, x4):
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = F.relu(x, inplace=True)
        x = self.conv3(x)
        x = F.relu(x, inplace=True)
        x = self.conv4(x)
        x = F.relu(x, inplace=True)
        x = self.conv5(x)
        return x

class flownet(nn.Module):
    def __init__(self):
        super(flownet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3)
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
    def forward(self, x1, x2_warp, fl):
        x = torch.cat((x1, x2_warp, fl), dim=1)
        x = self.conv1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = F.relu(x, inplace=True)
        x = self.conv3(x)
        x = F.relu(x, inplace=True)
        x = self.conv4(x)
        x = F.relu(x, inplace=True)
        x = self.conv5(x)
        return x
class flownet_new(nn.Module):
    def __init__(self):
        super(flownet_new, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=32, kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3)
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
    def forward(self, x1, x2_warp, fl, x3, x4):
        x = torch.cat((x1, x2_warp, fl, x3, x4), dim=1)
        x = self.conv1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = F.relu(x, inplace=True)
        x = self.conv3(x)
        x = F.relu(x, inplace=True)
        x = self.conv4(x)
        x = F.relu(x, inplace=True)
        x = self.conv5(x)
        return x

class backWarp(nn.Module):

    def __init__(self, W, H, device):
        super(backWarp, self).__init__()
        # create a grid
        gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))
        self.W = W
        self.H = H
        self.gridX = torch.tensor(gridX, requires_grad=False, device=device)
        self.gridY = torch.tensor(gridY, requires_grad=False, device=device)

    def forward(self, img, flow):
        # Extract horizontal and vertical flows.
        u = flow[:, 0, :, :]
        v = flow[:, 1, :, :]
        x = self.gridX.unsqueeze(0).expand_as(u).float() + u
        y = self.gridY.unsqueeze(0).expand_as(v).float() + v
        # range -1 to 1
        x = 2 * (x / self.W - 0.5)
        y = 2 * (y / self.H - 0.5)
        # stacking X and Y
        grid = torch.stack((x, y), dim=3)
        # Sample pixels using bilinear interpolation.
        imgOut = torch.nn.functional.grid_sample(img, grid)
        return imgOut

class Prob_net(nn.Module):
    def __init__(self):
        super(Prob_net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=7, stride=1, padding=3)

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.conv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.conv7 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.conv9 = nn.Conv2d(in_channels=(128+64), out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.conv11 = nn.Conv2d(in_channels=(64+32), out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.conv13 = nn.Conv2d(in_channels=(32+16), out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv14 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)

        self.conv15 = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, stride=1, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x1, x2, x3):
        x = torch.cat((x1, x2, x3), dim=1)
        self.c1 = self.conv1(x)
        self.c1 = F.relu(self.c1, inplace=True)
        self.c1 = self.conv2(self.c1)
        self.c1 = F.relu(self.c1, inplace=True)
        self.p1 = F.max_pool2d(self.c1, 2)

        self.c2 = self.conv3(self.p1)
        self.c2 = F.relu(self.c2, inplace=True)
        self.c2 = self.conv4(self.c2)
        self.c2 = F.relu(self.c2, inplace=True)
        self.p2 = F.max_pool2d(self.c2, 2)

        self.c3 = self.conv5(self.p2)
        self.c3 = F.relu(self.c3, inplace=True)
        self.c3 = self.conv6(self.c3)
        self.c3 = F.relu(self.c3, inplace=True)
        self.p3 = F.max_pool2d(self.c3, 2)

        self.c4 = self.conv7(self.p3)
        self.c4 = F.relu(self.c4, inplace=True)
        self.c4 = self.conv8(self.c4)
        self.c4 = F.relu(self.c4, inplace=True)

        self.up5 = F.interpolate(self.c4, scale_factor=2, mode='bilinear')
        self.merge5= torch.cat((self.c3, self.up5), dim=1)
        self.c5 = self.conv9(self.merge5)
        self.c5 = F.relu(self.c5, inplace=True)
        self.c5 = self.conv10(self.c5)
        self.c5 = F.relu(self.c5, inplace=True)

        self.up6 = F.interpolate(self.c5, scale_factor=2, mode='bilinear')
        self.merge6 = torch.cat((self.c2, self.up6), dim=1)
        self.c6 = self.conv11(self.merge6)
        self.c6 = F.relu(self.c6, inplace=True)
        self.c6 = self.conv12(self.c6)
        self.c6 = F.relu(self.c6, inplace=True)

        self.up7 = F.interpolate(self.c6, scale_factor=2, mode='bilinear')
        self.merge7 = torch.cat((self.c1, self.up7), dim=1)
        self.c7 = self.conv13(self.merge7)
        self.c7 = F.relu(self.c7, inplace=True)
        self.c7 = self.conv14(self.c7)
        self.c7 = F.relu(self.c7, inplace=True)

        self.c8 = self.conv15(self.c7)
        self.output = F.softmax(self.c8, dim=1)

        return self.output

class generator_net(nn.Module):
    def __init__(self):
        super(generator_net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=5, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x1, x2, x3, x4):
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = F.relu(x, inplace=True)
        x = self.conv3(x)
        x = F.relu(x, inplace=True)
        x = self.conv4(x)
        x = F.relu(x, inplace=True)
        x = self.conv5(x)
        return x


class discrimate_net(nn.Module):
    def __init__(self):
        super(discrimate_net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=7, stride=1, padding=3)

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.conv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.conv7 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.conv9 = nn.Conv2d(in_channels=(128 + 64), out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.conv11 = nn.Conv2d(in_channels=(64 + 32), out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.conv13 = nn.Conv2d(in_channels=(32 + 16), out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv14 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)

        self.conv15 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        #x = torch.cat((x1, x2, x3, x4), dim=1)
        self.c1 = self.conv1(x)
        self.c1 = F.relu(self.c1, inplace=True)
        self.c1 = self.conv2(self.c1)
        self.c1 = F.relu(self.c1, inplace=True)
        self.p1 = F.max_pool2d(self.c1, 2)

        self.c2 = self.conv3(self.p1)
        self.c2 = F.relu(self.c2, inplace=True)
        self.c2 = self.conv4(self.c2)
        self.c2 = F.relu(self.c2, inplace=True)
        self.p2 = F.max_pool2d(self.c2, 2)

        self.c3 = self.conv5(self.p2)
        self.c3 = F.relu(self.c3, inplace=True)
        self.c3 = self.conv6(self.c3)
        self.c3 = F.relu(self.c3, inplace=True)
        self.p3 = F.max_pool2d(self.c3, 2)

        self.c4 = self.conv7(self.p3)
        self.c4 = F.relu(self.c4, inplace=True)
        self.c4 = self.conv8(self.c4)
        self.c4 = F.relu(self.c4, inplace=True)

        self.up5 = F.interpolate(self.c4, scale_factor=2, mode='bilinear')
        self.merge5 = torch.cat((self.c3, self.up5), dim=1)
        self.c5 = self.conv9(self.merge5)
        self.c5 = F.relu(self.c5, inplace=True)
        self.c5 = self.conv10(self.c5)
        self.c5 = F.relu(self.c5, inplace=True)

        self.up6 = F.interpolate(self.c5, scale_factor=2, mode='bilinear')
        self.merge6 = torch.cat((self.c2, self.up6), dim=1)
        self.c6 = self.conv11(self.merge6)
        self.c6 = F.relu(self.c6, inplace=True)
        self.c6 = self.conv12(self.c6)
        self.c6 = F.relu(self.c6, inplace=True)

        self.up7 = F.interpolate(self.c6, scale_factor=2, mode='bilinear')
        self.merge7 = torch.cat((self.c1, self.up7), dim=1)
        self.c7 = self.conv13(self.merge7)
        self.c7 = F.relu(self.c7, inplace=True)
        self.c7 = self.conv14(self.c7)
        self.c7 = F.relu(self.c7, inplace=True)

        self.c8 = self.conv15(self.c7)
        self.output = F.sigmoid(self.c8)
        return self.output

class Prob_net_2(nn.Module):
    def __init__(self):
        super(Prob_net_2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=5, out_channels=16, kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=7, stride=1, padding=3)

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.conv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.conv7 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.conv9 = nn.Conv2d(in_channels=(128+64), out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.conv11 = nn.Conv2d(in_channels=(64+32), out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.conv13 = nn.Conv2d(in_channels=(32+16), out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv14 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)

        self.conv15 = nn.Conv2d(in_channels=16, out_channels=5, kernel_size=3, stride=1, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x1, x2, x3, x4, x5):
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        self.c1 = self.conv1(x)
        self.c1 = F.relu(self.c1, inplace=True)
        self.c1 = self.conv2(self.c1)
        self.c1 = F.relu(self.c1, inplace=True)
        self.p1 = F.max_pool2d(self.c1, 2)

        self.c2 = self.conv3(self.p1)
        self.c2 = F.relu(self.c2, inplace=True)
        self.c2 = self.conv4(self.c2)
        self.c2 = F.relu(self.c2, inplace=True)
        self.p2 = F.max_pool2d(self.c2, 2)

        self.c3 = self.conv5(self.p2)
        self.c3 = F.relu(self.c3, inplace=True)
        self.c3 = self.conv6(self.c3)
        self.c3 = F.relu(self.c3, inplace=True)
        self.p3 = F.max_pool2d(self.c3, 2)

        self.c4 = self.conv7(self.p3)
        self.c4 = F.relu(self.c4, inplace=True)
        self.c4 = self.conv8(self.c4)
        self.c4 = F.relu(self.c4, inplace=True)

        self.up5 = F.interpolate(self.c4, scale_factor=2, mode='bilinear')
        self.merge5= torch.cat((self.c3, self.up5), dim=1)
        self.c5 = self.conv9(self.merge5)
        self.c5 = F.relu(self.c5, inplace=True)
        self.c5 = self.conv10(self.c5)
        self.c5 = F.relu(self.c5, inplace=True)

        self.up6 = F.interpolate(self.c5, scale_factor=2, mode='bilinear')
        self.merge6 = torch.cat((self.c2, self.up6), dim=1)
        self.c6 = self.conv11(self.merge6)
        self.c6 = F.relu(self.c6, inplace=True)
        self.c6 = self.conv12(self.c6)
        self.c6 = F.relu(self.c6, inplace=True)

        self.up7 = F.interpolate(self.c6, scale_factor=2, mode='bilinear')
        self.merge7 = torch.cat((self.c1, self.up7), dim=1)
        self.c7 = self.conv13(self.merge7)
        self.c7 = F.relu(self.c7, inplace=True)
        self.c7 = self.conv14(self.c7)
        self.c7 = F.relu(self.c7, inplace=True)

        self.c8 = self.conv15(self.c7)
        self.output = F.softmax(self.c8, dim=1)

        return self.output
class Prob_net_3(nn.Module):
    def __init__(self):
        super(Prob_net_3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=7, stride=1, padding=3)

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.conv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.conv7 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.conv9 = nn.Conv2d(in_channels=(128+64), out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.conv11 = nn.Conv2d(in_channels=(64+32), out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.conv13 = nn.Conv2d(in_channels=(32+16), out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv14 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)

        self.conv15 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, stride=1, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        self.c1 = self.conv1(x)
        self.c1 = F.relu(self.c1, inplace=True)
        self.c1 = self.conv2(self.c1)
        self.c1 = F.relu(self.c1, inplace=True)
        self.p1 = F.max_pool2d(self.c1, 2)

        self.c2 = self.conv3(self.p1)
        self.c2 = F.relu(self.c2, inplace=True)
        self.c2 = self.conv4(self.c2)
        self.c2 = F.relu(self.c2, inplace=True)
        self.p2 = F.max_pool2d(self.c2, 2)

        self.c3 = self.conv5(self.p2)
        self.c3 = F.relu(self.c3, inplace=True)
        self.c3 = self.conv6(self.c3)
        self.c3 = F.relu(self.c3, inplace=True)
        self.p3 = F.max_pool2d(self.c3, 2)

        self.c4 = self.conv7(self.p3)
        self.c4 = F.relu(self.c4, inplace=True)
        self.c4 = self.conv8(self.c4)
        self.c4 = F.relu(self.c4, inplace=True)

        self.up5 = F.interpolate(self.c4, scale_factor=2, mode='bilinear')
        self.merge5= torch.cat((self.c3, self.up5), dim=1)
        self.c5 = self.conv9(self.merge5)
        self.c5 = F.relu(self.c5, inplace=True)
        self.c5 = self.conv10(self.c5)
        self.c5 = F.relu(self.c5, inplace=True)

        self.up6 = F.interpolate(self.c5, scale_factor=2, mode='bilinear')
        self.merge6 = torch.cat((self.c2, self.up6), dim=1)
        self.c6 = self.conv11(self.merge6)
        self.c6 = F.relu(self.c6, inplace=True)
        self.c6 = self.conv12(self.c6)
        self.c6 = F.relu(self.c6, inplace=True)

        self.up7 = F.interpolate(self.c6, scale_factor=2, mode='bilinear')
        self.merge7 = torch.cat((self.c1, self.up7), dim=1)
        self.c7 = self.conv13(self.merge7)
        self.c7 = F.relu(self.c7, inplace=True)
        self.c7 = self.conv14(self.c7)
        self.c7 = F.relu(self.c7, inplace=True)

        self.c8 = self.conv15(self.c7)
        self.output = F.softmax(self.c8, dim=1)

        return self.output


class gen_net(nn.Module):
    def __init__(self):
        super(gen_net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x1, x2, x3):
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.conv1(x)
        x_1 = F.relu(x, inplace=True)
        x = self.conv2(x_1)
        x = F.relu(x, inplace=True)
        x = self.conv3(x)
        x_1 = x + x_1
        x_1 = F.relu(x_1, inplace=True)
        x = self.conv4(x_1)
        x = F.relu(x, inplace=True)
        x = self.conv5(x)
        x_1 = x + x_1
        x = F.relu(x_1, inplace=True)
        x = self.conv6(x)
        return x

t = np.linspace(0.125, 0.875, 7)

def getFlowCoeff (indices, device):
    """
    Gets flow coefficients used for calculating intermediate optical
    flows from optical flows between I0 and I1: F_0_1 and F_1_0.

    F_t_0 = C00 x F_0_1 + C01 x F_1_0
    F_t_1 = C10 x F_0_1 + C11 x F_1_0

    where,
    C00 = -(1 - t) x t
    C01 = t x t
    C10 = (1 - t) x (1 - t)
    C11 = -t x (1 - t)

    Parameters
    ----------
        indices : tensor
            indices corresponding to the intermediate frame positions
            of all samples in the batch.
        device : device
                computation device (cpu/cuda).

    Returns
    -------
        tensor
            coefficients C00, C01, C10, C11.
    """


    # Convert indices tensor to numpy array
    ind = indices.detach().numpy()
    ind = indices
    C11 = C00 = - (1 - (t[ind])) * (t[ind])
    C01 = (t[ind]) * (t[ind])
    C10 = (1 - (t[ind])) * (1 - (t[ind]))
    return torch.Tensor(C00)[None, None, None, :].permute(3, 0, 1, 2).to(device), torch.Tensor(C01)[None, None, None, :].permute(3, 0, 1, 2).to(device), torch.Tensor(C10)[None, None, None, :].permute(3, 0, 1, 2).to(device), torch.Tensor(C11)[None, None, None, :].permute(3, 0, 1, 2).to(device)

def getWarpCoeff (indices, device):
    """
    Gets coefficients used for calculating final intermediate
    frame `It_gen` from backwarped images using flows F_t_0 and F_t_1.

    It_gen = (C0 x V_t_0 x g_I_0_F_t_0 + C1 x V_t_1 x g_I_1_F_t_1) / (C0 x V_t_0 + C1 x V_t_1)

    where,
    C0 = 1 - t
    C1 = t

    V_t_0, V_t_1 --> visibility maps
    g_I_0_F_t_0, g_I_1_F_t_1 --> backwarped intermediate frames

    Parameters
    ----------
        indices : tensor
            indices corresponding to the intermediate frame positions
            of all samples in the batch.
        device : device
                computation device (cpu/cuda).

    Returns
    -------
        tensor
            coefficients C0 and C1.
    """


    # Convert indices tensor to numpy array
    ind = indices.detach().numpy()
    ind = indices
    C0 = 1 - t[ind]
    C1 = t[ind]
    return torch.Tensor(C0)[None, None, None, :].permute(3, 0, 1, 2).to(device), torch.Tensor(C1)[None, None, None, :].permute(3, 0, 1, 2).to(device)


class sample_and_inirecon(nn.Module):
    def __init__(self, num_filters, B_size):
        super(sample_and_inirecon, self).__init__()
        self.sample = nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=B_size, stride=B_size, padding=0,
                                bias=False)
        self.sample_transpose = nn.Conv2d(in_channels=num_filters, out_channels=(B_size * B_size), kernel_size=1,
                                          stride=1, padding=0,
                                          bias=False)

    def forward(self, x_ini, y, flag):
        if flag:
            x = self.sample(x_ini)
            x = y - x
            x = self.sample_transpose(x)
            x = tensor2image(x)
            x = x + x_ini
            return x
        else:
            phi_x = self.sample(x_ini)
            x = self.sample_transpose(phi_x)
            x = tensor2image(x)
            return x, phi_x


class sample_net(nn.Module):
    def __init__(self, num_filters, B_size):
        super(sample_net, self).__init__()
        self.sample = nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=B_size, stride=B_size, padding=0,
                                bias=False)

    def forward(self, x):
        self.output = self.sample(x)
        return self.output


class ini_recon_net(nn.Module):
    def __init__(self, num_filters, B_size):
        super(ini_recon_net, self).__init__()
        self.sample_transpose = nn.Conv2d(in_channels=num_filters, out_channels=(B_size * B_size), kernel_size=1,
                                          stride=1, padding=0,
                                          bias=False)

    def forward(self, phi_x):
        x = self.sample_transpose(phi_x)
        x = tensor2image(x)
        return x

class sample_and_inirecon_2(nn.Module):
    def __init__(self, num_filters, B_size, init_weight):
        super(sample_and_inirecon_2, self).__init__()
        self.sample = nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=B_size, stride=B_size, padding=0,
                                bias=False)
        self.sample_transpose = nn.Conv2d(in_channels=num_filters, out_channels=(B_size * B_size), kernel_size=1,
                                          stride=1, padding=0,
                                          bias=False)

        init_weight = torch.nn.Parameter(init_weight)
        self.sample.weight = init_weight

    def forward(self, x_ini, y, flag):
        if flag:
            x = self.sample(x_ini)
            x = y - x
            x = self.sample_transpose(x)
            x = tensor2image(x)
            x = x + x_ini
            return x
        else:
            phi_x = self.sample(x_ini)
            delta_y = y - phi_x
            x = self.sample_transpose(delta_y)
            x = tensor2image(x)
            return x, delta_y



class sample_and_inirecon_params(nn.Module):
    def __init__(self, num_filters, B_size):
        super(sample_and_inirecon_params, self).__init__()
        self.sample = nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=B_size, stride=B_size, padding=0,
                                bias=False)
        self.sample_transpose = nn.Conv2d(in_channels=num_filters, out_channels=(B_size * B_size), kernel_size=1,
                                          stride=1, padding=0,
                                          bias=False)

    def forward(self, x_ini, y, flag, params):
        if flag:
            x = self.sample(x_ini)
            x = y - x
            x = self.sample_transpose(x)
            x = tensor2image(x)
            x = params * x + x_ini
            return x
        else:
            phi_x = self.sample(x_ini)
            x = self.sample_transpose(phi_x)
            x = tensor2image(x)
            return x


class Biv_Shr(nn.Module):
    def __init__(self):
        super(Biv_Shr, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x, flag):
        x1 = self.conv1(x)
        x1 = F.relu(x1, inplace=True)
        x1 = self.conv2(x1)
        if flag:
            x2 = self.conv3(x1)
            x2 = F.sigmoid(x2)
            x1 = x1 * x2
        x2 = self.conv4(x1)
        x2 = F.relu(x2, inplace=True)
        x2 = self.conv5(x2)

        return x2


class wiener_net(nn.Module):
    def __init__(self):
        super(wiener_net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        return x

