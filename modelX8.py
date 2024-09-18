import torch.nn as nn
import torch
import math
import torch.nn.functional as F

import common as common
from channel_cluster_3 import ChannelClustering
from channel_dynamic_filter import CDF
from Dynamic_fusion3 import DFM
from PNAB import PNAB


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=True)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=True)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, res_scale=1):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)

    def forward(self, x):
        x1 = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out * self.res_scale + x1
        return out

# Super division module
class SDM(nn.Module):
    def __init__(self, in_channels, num_clusters, batchsize):
        super(SDM, self).__init__()
        self.in_channels = in_channels
        self.num_clusters = num_clusters
        self.batchsize = batchsize


        self.ChannelCluster = ChannelClustering(in_channels=self.in_channels, num_clusters=self.num_clusters)

        self.SD_1 = PNAB(in_channels=self.num_clusters)

        self.c1_c = nn.Conv2d(self.num_clusters, self.num_clusters, 1, 1, 0)

        self.CDF_1 = DFM(in_channels=self.in_channels, num_cluster=self.num_clusters, batchsize=self.batchsize)

        self.conv_fuse1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, padding=1)
        )
        self.conv_fuse2 = nn.Sequential(
            nn.Conv2d(self.in_channels + self.num_clusters, self.in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, padding=1)
        )

    def forward(self, LR):
        LR_filter = self.ChannelCluster(LR)

        HR = self.c1_c(self.SD_1(LR_filter))


        LR_fuse_1 = self.CDF_1(HR, LR) + LR

        LR_out = self.conv_fuse1(LR_fuse_1)
        final_out = self.conv_fuse2(torch.cat([LR_out, HR], dim=1))

        return final_out

class main(nn.Module):
    def __init__(self, in_channels, num_clusters, batchsize):
        super(main, self).__init__()
        self.in_channels = in_channels
        self.num_clusters = num_clusters
        self.batchsize = batchsize

        self.RBS = nn.ModuleList()
        for _ in range(4):
            self.RBS.append(nn.Sequential(
                ResBlock(self.in_channels, self.in_channels, 1),
                ResBlock(self.in_channels, self.in_channels, 1),
                ResBlock(self.in_channels, self.in_channels, 1),
                #ResBlock(self.in_channels, self.in_channels, 1),
                # ResBlock(self.in_channels, self.in_channels, 1),
            ))

        self.SDM1 = SDM(self.in_channels, self.num_clusters, batchsize=self.batchsize)
        self.SDM2 = SDM(self.in_channels, self.num_clusters, batchsize=self.batchsize)
        self.SDM3 = SDM(self.in_channels, self.num_clusters, batchsize=self.batchsize)
        self.SDM4 = SDM(self.in_channels, self.num_clusters, batchsize=self.batchsize)
        # self.SDM5 = SDM(self.in_channels, self.num_clusters, batchsize=self.batchsize)
        # self.SDM6 = SDM(self.in_channels, self.num_clusters, batchsize=self.batchsize)

        self.final_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, padding=0),
        )

        self.upsampleX2 = nn.Upsample(scale_factor=2, mode='bicubic')
        self.upsampleX4 = nn.Upsample(scale_factor=4, mode='bicubic')
        self.upsampleX8 = nn.Upsample(scale_factor=8, mode='bicubic')

    def forward(self, LR):
        LRX2 = self.upsampleX2(LR)
        LRX4 = self.upsampleX4(LR)
        LRX8 = self.upsampleX8(LR)

        LR_out_1 = self.SDM1(LR) + LR
        # LR_out_1 = self.SDM1(LR) + LR
        out_1 = self.RBS[0](LR_out_1)


        out_1X2 = self.upsampleX2(out_1) + LRX2

        LR_out_2 = self.SDM2(out_1X2) + out_1X2
        out_2 = self.RBS[1](LR_out_2)


        out_3_up = self.upsampleX2(out_2) + LRX4

        LR_out_4 = self.SDM3(out_3_up) + out_3_up
        out_4 = self.RBS[2](LR_out_4)

        out_4_up = self.upsampleX2(out_4) + LRX8

        LR_out_5 = self.SDM4(out_4_up) + out_4_up
        final_out= self.RBS[3](LR_out_5)



        final_out = self.final_conv(final_out) + LRX8

        return final_out, out_4, out_2, out_1