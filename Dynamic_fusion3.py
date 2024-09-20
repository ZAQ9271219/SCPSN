import torch.nn as nn
import torch
import numpy as np

# Dynamic fusion module
class DFM(nn.Module):
    def __init__(self, in_channels, num_cluster, batchsize):
        super(DFM, self).__init__()
        self.in_channels = in_channels
        self.num_clusters = num_cluster
        self.K = num_cluster // 2
        self.batchsize = batchsize

        self.conv = nn.Sequential(
            nn.Conv2d(1 + self.K, 1, kernel_size=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1, 1, kernel_size=1, padding=0)
        )

    def forward(self, HR, LR):
        HR_flatten = HR.view(HR.shape[0], HR.shape[1], -1)
        LR_flatten = LR.view(LR.shape[0], LR.shape[1], -1)

        correlation_matrix = torch.matmul(LR_flatten, HR_flatten.permute(0, 2, 1))

        max_values, max_corr_indices = torch.topk(correlation_matrix, self.K, dim=2)

        result_img = torch.ones_like(LR)

        for i in range(LR.shape[0]):
            selected_channel_indices = max_corr_indices[i, :, :]
            selected_channels = HR[i, selected_channel_indices, :, :]
            LR_selected = LR[i, :, :, :].unsqueeze(1)
            img_cat = torch.cat([LR_selected, selected_channels], dim=1)
            max_weight = torch.max(selected_channels, dim=1, keepdim=True)[0]
            weight = torch.sigmoid(max_weight)
            fusion_img = self.conv(img_cat)
            add_img = weight * fusion_img
            result_img[i, :, :, :] = (fusion_img + add_img).squeeze(1)


        return result_img

