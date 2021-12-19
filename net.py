import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2


L1_NORM = lambda a: torch.sum(a + 1e-8)



# 基础卷积层
class ConvLayer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, pad, is_last=False):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, pad)          # pad 是卷积前
        self.is_last = is_last

    def forward(self, x):
        out = self.conv(x)
        if self.is_last is False:
            out = F.relu(out, inplace=True)
        return out


class InceptionResNet(nn.Module):
    def __init__(self, in_channels, out_channels ):
        super(InceptionResNet, self).__init__()
        self.conv1= ConvLayer(in_channels, out_channels, kernel_size=3, stride=1, pad=1)        # 1x1
        self.conv2 = nn.Sequential(
            ConvLayer(in_channels, out_channels, kernel_size=3, stride=1, pad=1),                # 1x1
            ConvLayer(out_channels, out_channels, kernel_size=3, stride=1, pad=1),                # 3x3
        )
        self.conv3 = nn.Sequential(
            ConvLayer(in_channels, out_channels, kernel_size=3, stride=1, pad=1),                 # 1x1
            ConvLayer(out_channels, out_channels, kernel_size=3, stride=1, pad=1),                 # 3x3
            ConvLayer(out_channels, out_channels, kernel_size=3, stride=1, pad=1)                  # 3x3
        )
        self.conv4 = ConvLayer(3*out_channels, in_channels, kernel_size=1, stride=1, pad=0)         # 1x1

    def forward(self, x):
     # Inception 结构
         x1 = self.conv1(x)
         x2 = self.conv2(x)
         x3 = self.conv3(x)
         x123 = torch.cat([x1, x2, x3], 1)
         x4 = self.conv4(x123)
         out = x4 + x
         return out

class InceptionResNet(nn.Module):
    def __init__(self, in_channels, out_channels ):
        super(InceptionResNet, self).__init__()
        self.conv1= ConvLayer(in_channels, out_channels, kernel_size=3, stride=1, pad=1)        # 1x1
        self.conv2 = nn.Sequential(
            ConvLayer(in_channels, out_channels, kernel_size=3, stride=1, pad=1),                # 1x1
            ConvLayer(out_channels, out_channels, kernel_size=3, stride=1, pad=1),                # 3x3
        )
        self.conv3 = nn.Sequential(
            ConvLayer(in_channels, out_channels, kernel_size=3, stride=1, pad=1),                 # 1x1
            ConvLayer(out_channels, out_channels, kernel_size=3, stride=1, pad=1),                 # 3x3
            ConvLayer(out_channels, out_channels, kernel_size=3, stride=1, pad=1)                  # 3x3
        )
        self.conv4 = ConvLayer(3*out_channels, in_channels, kernel_size=1, stride=1, pad=0)         # 1x1

    def forward(self, x):
     # Inception 结构
         x1 = self.conv1(x)
         x2 = self.conv2(x)
         x3 = self.conv3(x)
         x123 = torch.cat([x1, x2, x3], 1)
         x4 = self.conv4(x123)
         out = x4 + x
         return out


class AverageFusion(nn.Module):
    def forward(self, x, y):
        return (x + y)/2.0


class MaxFusion(nn.Module):
    def forward(self, x, y):
        u = torch.max(x, y)
        return u


class NetWork(nn.Module):
    def __init__(self, input_nc=1, output_nc=1):
        super(NetWork, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        ms_resnet = InceptionResNet
        nb_filter = [32, 64, 128, 256]
        pad = 1
        kernel_size = 3
        stride = 1
        # encoder
        self.conv1 = ConvLayer(input_nc, nb_filter[0], kernel_size, stride, pad)                 # input:1 output:32
        self.conv2 = ConvLayer(nb_filter[0], nb_filter[1], kernel_size, stride, pad)             # input:32 output:64
        self.conv3 = ms_resnet(nb_filter[1], nb_filter[2])                                       # input:64 output:64
        self.conv4 = ms_resnet(nb_filter[1], nb_filter[2])                                       # input:64 output:64

        # decoder
        self.decoder = nn.Sequential(
            ConvLayer(nb_filter[1], nb_filter[1], kernel_size, stride, pad),
            ConvLayer(nb_filter[1], nb_filter[0], kernel_size, stride, pad),
            ConvLayer(nb_filter[0], output_nc, kernel_size, stride, pad, True)
        )

        # fuse
        self.fuse = AverageFusion()
        #self.fuse = MaxFusion()

    def forward(self, img_ir, img_vis):
        # encoder
        c11 = self.conv1(img_ir)
        c12 = self.conv2(c11)

        c21 = self.conv1(img_vis)
        c22 = self.conv2(c21)

        c13 = self.conv3(c12)                        # torch.Size([1, 64, 256, 256])
        c14 = self.conv4(c13)

        c23 = self.conv3(c22)                     # torch.Size([1, 64, 256, 256])
        c24 = self.conv4(c23)

        # fusing layer
        cf = self.fuse(c14, c24)
        y_f = self.decoder(cf)

        return y_f

if __name__=="__main__":
    net_model = NetWork()
    net_model.cuda()
    x1 = torch.rand([1, 1, 256, 256]).cuda()
    x2 = torch.rand([1, 1, 256, 256]).cuda()
    y = net_model.forward(x1, x2)
    print(y.shape)
