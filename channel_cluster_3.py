import torch.nn as nn
import torch
import numpy as np
from sklearn.cluster import KMeans,MiniBatchKMeans
import cv2
import scipy.io
import h5py
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

'''
通道聚类模块
输入图片为[bs, num_c, w, h], 展开为[bs, num_c, w×h],一张[num_c, w×h]与[w×h, num_c]计算得到通道相关系数图
'''
class CorrelationModel(nn.Module):
    def __init__(self, in_channels):
        super(CorrelationModel, self).__init__()
        self.num_channels = in_channels
        # self.g = nn.Conv2d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=3, stride=1,
        #                    padding=1)

    def forward(self, image):
        # image = self.g(image)
        # 计算图像的均值，可以指定要减去的维度
        mean_image = image.mean(dim=(2,3), keepdim=True)
        channel_std = image.std(dim=(2, 3), keepdim=True)
        image = (image - mean_image) / (channel_std + 1e-8)
        # 将两个图像展平成一维向量
        image1 = image.view(image.shape[0], image.shape[1], -1)
        image2 = image.view(image.shape[0], image.shape[1], -1)

        # 计算相关系数矩阵
        correlation_matrix = torch.matmul(image1, image2.permute(0, 2, 1))

        # 计算每个通道的模长
        channel_magnitude1 = torch.sqrt(torch.sum(image1 ** 2, dim=(2,)))
        channel_magnitude2 = torch.sqrt(torch.sum(image2 ** 2, dim=(2,)))

        # 标准化相关系数矩阵
        correlation_matrix /= torch.unsqueeze(channel_magnitude1, 2)
        correlation_matrix /= torch.unsqueeze(channel_magnitude2, 1)

        # 将相关性矩阵保存为 NumPy 数组
        # correlation_matrix_np = correlation_matrix.cpu().detach().numpy()  # 如果使用 PyTorch，首先将其移到 CPU 上
        # print(correlation_matrix_np.shape)
        # # 选择要保存的文件路径
        # file_path = 'D:\Program File\Pytorch1.11 Program\MY\channel-cluster\\files\\pavia_correlation_matrix_np'
        # # 使用 NumPy 保存相关性矩阵到文件
        # np.save(file_path, correlation_matrix_np)
        # print(f"Correlation matrix saved to {file_path}")
        #
        # print("corrmatrix has been caculated")
        return correlation_matrix


class ChannelClustering(nn.Module):
    def __init__(self, in_channels, num_clusters):
        super(ChannelClustering, self).__init__()
        self.in_channels = in_channels
        self.num_clusters = num_clusters
        self.correlation_model = CorrelationModel(self.in_channels)



    def forward(self, image):
        b, c, h, w = image.shape

        correlation_matrix = self.correlation_model(image).to("cuda")


        center_elements_tensor = torch.zeros((b, self.num_clusters, h, w))
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=0, n_init='auto')
        for j in range(b):
            cluster_labels = torch.from_numpy(kmeans.fit_predict(correlation_matrix[j,:,:].cpu().detach().numpy()))
            image_temp = image[j, :, :, :]
            image_reshaped = image_temp.view(1, c, -1)

            center_elements = []
            for i in range(self.num_clusters):
                channels_in_cluster = torch.nonzero(cluster_labels == i, as_tuple=False).view(-1)

                cluster_image = image_reshaped[:, channels_in_cluster, :]

                distances = torch.sqrt(torch.sum((cluster_image.unsqueeze(1) - cluster_image.unsqueeze(2)) ** 2, dim=3))
                avg_distances = torch.mean(distances, dim=-1)

                center_index = torch.argmin(avg_distances, dim=1)

                center_elements.append(cluster_image.gather(1, center_index.view(-1, 1, 1).expand(-1, -1, cluster_image.size(2))))


            if center_elements:
                temp = torch.stack(center_elements, dim=1).squeeze(2)
                temp = temp.clone().detach().view(1, self.num_clusters, h, w)
                center_elements_tensor[j, :, :, :] = temp
                center_elements.clear()




        center_elements_tensor = center_elements_tensor.to("cuda")

        return center_elements_tensor




